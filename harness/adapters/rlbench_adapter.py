"""RLBench adapter: the simulator backend for this harness.

Owns the live RLBench Environment + TaskEnvironment; exposes
reset/observe/execute/query_action_status to the runtime, and the success
signal through adapters/judge.py only.

Perception mode is gt_debug: object poses come from the simulator scene via
named PyRep objects, and the canonical task object names (cup1 / rubbish /
bin) are published directly. This is the mode the 431-episode four-arm
experiment ran in, and it is the mode this repository ships.

A sensor_eval mode — objects sourced only from the vision chain, with no
ground-truth fallback — is the intended next step, because a ground-truth
fallback would leak oracle poses into the agent's view and invalidate any
conclusion drawn from it. That mode needs a perception stack (segmentation +
instance registration) that is not part of this repository; see the "Status &
Scope" section of the README. Constructing the adapter with
perception_mode="sensor_eval" therefore raises rather than silently falling
back to ground truth.
"""
from __future__ import annotations

import os
import time
from typing import Optional

from harness.skills.grasp import run_grasp
from harness.skills.place import run_place
from harness.skills.motion import MotionError

# task -> graspable object id, container id, PyRep object names, and the
# calibrated grasp palm offset (height above object frame center; from
# scripts/probe_grasp_heights.py sweeps — cup rim band +0.06, small block 0.0)
TASK_SPECS = {
    "PickUpCup": {"target": "cup1", "container": None,
                  "objects": {"cup1": "cup1", "cup2": "cup2"},
                  "grasp_palm_offset": 0.06},
    "PutRubbishInBin": {"target": "rubbish", "container": "bin",
                        "objects": {"rubbish": "rubbish", "bin": "success"},
                        "grasp_palm_offset": 0.0},
    # "bin" is exposed as the success-sensor frame (judge criterion region);
    # valid only in gt_debug mode, recorded in success_semantics.csv
}


class RlbenchAdapter:
    def __init__(self, task_name: str = "PickUpCup", robot: str = "panda",
                 camera_size=(640, 480), headless: bool = True,
                 grasp_z_top_offset: float = 0.02,
                 fault_f1_empty_first: bool = False,
                 fault_f2_timeout_receipt: bool = False,
                 perception_mode: str = "gt_debug"):
        if perception_mode != "gt_debug":
            raise NotImplementedError(
                f"perception_mode={perception_mode!r} needs the perception "
                "stack (segmentation + instance registration), which is not "
                "part of this repository. Only gt_debug is published here — "
                "see the README's 'Status & Scope' section.")
        self.task_name = task_name
        self.perception_mode = perception_mode
        self.spec = TASK_SPECS[task_name]
        self.robot = robot
        self.camera_size = camera_size
        self.headless = headless
        self.grasp_z_top_offset = grasp_z_top_offset
        # M3 fault-injection proxies (experiment plan section 6.3, injected at
        # the skill proxy layer, identical across experiment groups):
        #   F1: first grasp closes in the air (honest REACHED, no attach)
        #   F2: first grasp APPLIES fully but the receipt is lost (TIMEOUT) —
        #       the signature scenario; runtime must re-verify the SAME
        #       call_id, never re-send (repeated-action rate must stay 0)
        self.fault_f1_empty_first = fault_f1_empty_first
        self.fault_f2_timeout_receipt = fault_f2_timeout_receipt
        self._f1_used = False
        self._f2_used = False
        self._f2_calls: set = set()
        self._f2_hide_next = False
        self._last_world = None
        self._env = None
        self._task = None
        self._pyrep_objects: dict = {}
        self._version = 0
        self._last_quat = None
        self._timed_out_calls: set = set()
        self._ee_pose_cache = None
        self._gripper_open_cache = 1.0
        self._cup_top_offset = 0.05   # half cup height; refined in smoke

    # ------------------------------------------------------------- lifecycle
    def _ensure_env(self):
        if self._env is not None:
            return
        os.environ.setdefault("COPPELIASIM_ROOT", os.path.expanduser("~/CoppeliaSim410"))
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig
        import rlbench.tasks as rlbench_tasks

        obs_config = ObservationConfig()
        obs_config.set_all(True)
        for cam in ("front_camera", "left_shoulder_camera", "right_shoulder_camera",
                    "overhead_camera", "wrist_camera"):
            cc = getattr(obs_config, cam)
            cc.image_size = tuple(self.camera_size)
            cc.depth_in_meters = True

        action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaIK(collision_checking=False),
            gripper_action_mode=Discrete(detach_before_open=True))
        self._env = Environment(action_mode=action_mode, obs_config=obs_config,
                                headless=self.headless, robot_setup=self.robot)
        self._env.launch()
        task_cls = getattr(rlbench_tasks, self.task_name)
        self._task = self._env.get_task(task_cls)
        self._task.set_variation(0)
        self._bind_pyrep_objects()

    def _bind_pyrep_objects(self):
        from pyrep.objects.shape import Shape
        from pyrep.objects.proximity_sensor import ProximitySensor
        names = self.spec["objects"]
        for obj_id, pyrep_name in names.items():
            if pyrep_name == "success":
                self._pyrep_objects[obj_id] = ProximitySensor("success")
            else:
                self._pyrep_objects[obj_id] = Shape(pyrep_name)

    # ---------------------------------------------------------------- reset
    def reset(self, seed: int) -> None:
        """Rule 7: seeds FIRST, then any other state work."""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        self._ensure_env()
        desc, obs = self._task.reset()
        # the front camera home pose: captured after the first task load;
        # move_camera persists across episodes (scene reset does not touch
        # the sensor), so restore it here
        if getattr(self, "_cam_home", None) is None:
            cam = self._task._scene._cam_front
            pos = cam.get_position()
            ori = cam.get_orientation()
            self._cam_home = [float(pos[0]), float(pos[1]), float(pos[2]),
                              float(ori[0]), float(ori[1]), float(ori[2])]
        else:
            cam = self._task._scene._cam_front
            cam.set_position(self._cam_home[:3])
            cam.set_orientation(list(self._cam_home[3:6]))
        self._version = 1
        self._timed_out_calls = set()
        self._f1_used = False
        self._f2_used = False
        self._f2_calls = set()
        self._f2_hide_next = False
        self._last_world = None
        self._last_quat = None
        self._snapshot_obs(obs)

    def shutdown(self) -> None:
        if self._env is not None:
            try:
                self._env.shutdown()
            except Exception:
                pass
            self._env = None

    # ------------------------------------------------------------- observe
    def _snapshot_obs(self, obs) -> None:
        if obs is None or obs.gripper_pose is None:
            return
        self._ee_pose_cache = [float(v) for v in obs.gripper_pose[:7]]
        self._last_quat = tuple(float(v) for v in obs.gripper_pose[3:7])
        self._gripper_open_cache = float(obs.gripper_open)

    def observe(self) -> "WorldState":
        self._ensure_env()
        if self._f2_hide_next:
            # F2 receipt-loss proxy: first post-action frame predates the
            # physical effect (see _observe_hidden)
            self._f2_hide_next = False
            return self._observe_hidden()
        self._version += 1
        self._clock = time.time()
        from harness.spec.types import ObjInstance, Pose, WorldState

        objects = []
        for obj_id, pyrep_obj in self._pyrep_objects.items():
            pos = pyrep_obj.get_position()
            if hasattr(pyrep_obj, "get_orientation"):
                pass
            objects.append(ObjInstance(
                id=obj_id,
                label=obj_id,
                view="front",
                pose=Pose(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                sigma_m=0.004,
                size=(0.08, 0.08, 0.10),
                visible_ratio=1.0,        # gt_debug: omniscient
                quality="ok",
                source="cad",
                last_seen_t=self._clock))
        ee = self._ee_pose_cache or [0.0, 0.0, 0.3, 0, 0, 0, 1]
        ee_pose = Pose(x=ee[0], y=ee[1], z=ee[2], qx=ee[3], qy=ee[4], qz=ee[5],
                       qw=ee[6])
        world = WorldState(version=self._version, timestamp=self._clock,
                           ee_pose=ee_pose, gripper_open=self._gripper_open_cache,
                           joints={}, objects=tuple(objects), unsafe_flag=False,
                           evidence_refs=())
        self._last_world = world
        return world

    def _observe_hidden(self) -> "WorldState":
        """F2 receipt-loss proxy: the first post-action observation predates
        the physical effect (receipt lost AND fresh frame unavailable). The
        runtime's R0 re-verify path then asks query_action_status and gets a
        fresh observation."""
        self._version += 1
        base = self._last_world
        if base is None:
            return self.observe()
        from dataclasses import replace
        hidden = replace(base, version=self._version)
        self._last_world = hidden
        return hidden

    def execute(self, call):
        import time as _t
        from harness.spec.types import ExecutionResult, Status
        t0 = _t.perf_counter()
        result = self._dispatch(call)
        elapsed_ms = int(max(1, (_t.perf_counter() - t0) * 1000))
        return ExecutionResult(call_id=call.call_id, status=result[0],
                               feedback=result[1],
                               evidence_refs=(f"ev_{call.call_id}",),
                               elapsed_ms=elapsed_ms)

    def _dispatch(self, call):
        from harness.spec.types import Status
        name, args = call.name, call.args

        if name in ("observe", "where_is", "list_objects"):
            return Status.REACHED, {"objects": list(self._pyrep_objects.keys())}
        if name == "wait":
            return Status.REACHED, {}

        if name == "grasp":
            # M3 fault proxies (first grasp per episode only, identical
            # injection across experiment groups):
            if self.fault_f1_empty_first and not self._f1_used:
                self._f1_used = True
                # close in the air above the object: honest REACHED, no attach
                return self._run_grasp_skill(call, palm_extra=0.08)
            if self.fault_f2_timeout_receipt and not self._f2_used:
                self._f2_used = True
                out = self._run_grasp_skill(call)      # fully applies
                self._f2_calls.add(call.call_id)
                self._f2_hide_next = True              # receipt lost
                return Status.TIMEOUT, {"claimed": {"success": True}}
            return self._run_grasp_skill(call)
        if name == "place":
            return self._run_place_skill(call)
        if name == "move_camera":
            # active perception: reposition the front camera to look DOWN at
            # the named object (e.g. into the mesh bin). The camera is a
            # movable scene object; reset() restores its home pose.
            try:
                cam = self._task._scene._cam_front
                tx, ty, tz = (float(v) for v in call.args["target_xyz"])
                cam.set_position([tx, ty, max(tz + 0.35, 0.95)])
                import math as _m
                cam.set_orientation([-_m.pi / 2 + 0.35, 0.0, 0.0])
                return Status.REACHED, {"claimed": {"look_at":
                                            call.args.get("object_id", "")}}
            except Exception as exc:
                return Status.BLOCKED, {"error": str(exc)[:120]}

        return Status.NOT_IMPLEMENTED, {}

    # ------------------------------------------------------------ skills
    def _make_step_fn(self):
        """Returns step_fn(action_xyz, gripper) -> obs snapshot dict.

        Orientation: keep the current ee orientation (self._last_quat).
        Motion: interpolate from the current ee position toward the target in
        <=5cm increments — EndEffectorPoseViaIK's jacobian IK refuses far
        targets outright (RLBench InvalidActionError). Backend refusals are
        surfaced as MotionError so skills abort their phase (TIMEOUT) instead
        of crashing the FSM.
        """
        def step_fn(action_xyz, gripper_cmd):
            quat = self._last_quat or (0.0, 0.7071068, 0.0, 0.7071068)
            cur = list(self._ee_pose_cache[:3]) if self._ee_pose_cache \
                else [0.0, 0.0, 0.8]
            if action_xyz is None:
                tgt = list(cur)                 # keep position (open-in-place)
            else:
                tgt = [float(action_xyz[0]), float(action_xyz[1]),
                       float(action_xyz[2])]
            try:
                return self._interp_step(tgt, quat, gripper_cmd)
            except MotionError:
                raise
            except Exception as exc:
                raise MotionError(f"{type(exc).__name__}: {exc}") from exc
        return step_fn

    def _interp_step(self, tgt: list, quat: tuple, gripper_cmd: float) -> dict:
        try:
            return self._interp_step_inner(tgt, quat, gripper_cmd)
        except Exception as exc:
            # one bisection retry: split the remaining segment in half (a far
            # or steep IK target often succeeds when approached from midway)
            mid = [(a + b) / 2 for a, b in zip(
                self._ee_pose_cache[:3] if self._ee_pose_cache else [0.0, 0.0, 0.8],
                tgt)]
            try:
                self._interp_step_inner(mid, quat, gripper_cmd)
            except Exception:
                raise exc
            return self._interp_step_inner(tgt, quat, gripper_cmd)

    def _interp_step_inner(self, tgt: list, quat: tuple, gripper_cmd: float) -> dict:
        cur = list(self._ee_pose_cache[:3]) if self._ee_pose_cache \
            else [0.0, 0.0, 0.8]
        dist = ((tgt[0] - cur[0]) ** 2 + (tgt[1] - cur[1]) ** 2 +
                (tgt[2] - cur[2]) ** 2) ** 0.5
        n = max(1, int(dist / 0.05) + 1)
        obs = None
        for i in range(1, n + 1):
            t = i / n
            px = cur[0] + (tgt[0] - cur[0]) * t
            py = cur[1] + (tgt[1] - cur[1]) * t
            pz = cur[2] + (tgt[2] - cur[2]) * t
            action = [px, py, pz,
                      float(quat[0]), float(quat[1]), float(quat[2]),
                      float(quat[3]), float(gripper_cmd)]
            # stalling (silent joint-limit stop in EndEffectorPoseViaIK)
            # can leave the last increment short: re-issue up to 3 times
            for _retry in range(3):
                obs = self._task_step(action)
                self._snapshot_obs(obs)
                cur_now = self._ee_pose_cache[:3]
                moved = ((cur_now[0] - px) ** 2 + (cur_now[1] - py) ** 2 +
                         (cur_now[2] - pz) ** 2) ** 0.5
                if moved < 0.002:
                    break
            cur = list(self._ee_pose_cache[:3])
        return {"ee": self._ee_pose_cache,
                "gripper_open": self._gripper_open_cache}

    def _task_step(self, action):
        """TaskEnvironment.step returns (obs, reward, terminated) — we keep
        the observation only; reward is judge-channel and unread."""
        obs, _reward, _done = self._task.step(action)
        return obs

    def _run_grasp_skill(self, call, palm_extra: float = 0.0):
        from harness.spec.types import Status
        args = call.args
        obj_id = args.get("object_id", self.spec["target"])
        obj = self._pyrep_objects.get(obj_id)
        if obj is None:
            return Status.BLOCKED, {"claimed": {"success": False}}
        base_offset = self.spec.get("grasp_palm_offset", 0.0) + palm_extra
        regrasp_fn = self._make_regrasp_fn()
        # low objects near the base can make the calibrated palm height collide
        # with the table (fingers below surface -> IK abort mid-descend);
        # retry once with a raised palm before giving up
        last = None
        for offset in (base_offset, base_offset + 0.03):
            pos = obj.get_position()
            grasp_frame = [float(pos[0]), float(pos[1]),
                           float(pos[2]) + offset]
            try:
                out = run_grasp(self._make_step_fn(), grasp_frame,
                                regrasp_fn=regrasp_fn)
            except MotionError as exc:
                last = {"claimed": {"success": False, "error": str(exc)},
                        "aborted_at": "exception", "phases": {}}
                continue
            if not out.get("aborted_at"):
                return Status.REACHED, {"claimed": {"success": True},
                                        "phases": out.get("phases", {})}
            last = out
        self._timed_out_calls.add(call.call_id)
        return Status.TIMEOUT, {"claimed": {"success": False},
                                "aborted_at": last.get("aborted_at"),
                                "phases": last.get("phases", {})}

    def _run_place_skill(self, call):
        from harness.spec.types import Status
        target = (call.args or {}).get("target_xyz")
        if not target:
            return Status.BLOCKED, {"claimed": {"success": False}}
        release_fn = self._make_release_fn()
        try:
            out = run_place(self._make_step_fn(), [float(target[0]),
                                                   float(target[1]),
                                                   float(target[2])],
                            release_fn=release_fn)
        except MotionError as exc:
            return Status.IK_FAILED, {"claimed": {"success": False,
                                                  "error": str(exc)}}
        if out.get("aborted_at"):
            return Status.TIMEOUT, {"claimed": {"success": False},
                                    "aborted_at": out["aborted_at"]}
        return Status.REACHED, {"claimed": {"success": True}}

    def _make_release_fn(self):
        """Explicit detach primitive. The Discrete gripper mode skips its
        release branch during kinematic attach carry (open_amount already
        reads open), so the skill calls PyRep's gripper.release() directly.
        This is actuation, not judging."""
        def release_fn():
            self._task._task.robot.gripper.release()
        return release_fn

    def _make_regrasp_fn(self):
        """Re-fire the attach attempt on nearby graspables between settle
        steps. PyRep's gripper.grasp() checks the attach proximity sensor
        internally, so this is a no-op unless an object is genuinely at the
        gripper."""
        def regrasp_fn():
            try:
                for g_obj in self._task._task.get_graspable_objects():
                    try:
                        self._task._task.robot.gripper.grasp(g_obj)
                    except Exception:
                        continue
            except Exception:
                pass
        return regrasp_fn

    # --------------------------------------------------------- F2 support
    def query_action_status(self, call_id: str) -> str:
        """Synchronous backend. For an injected F2 call the skill DID apply
        (the receipt was lost, not the effect) -> answer honestly 'applied' so
        the runtime re-verifies the SAME call_id instead of re-sending. A
        non-injected abort leaves the outcome partially unknown -> 'unknown'
        forces evidence re-verification without re-dispatch."""
        if call_id in self._f2_calls:
            return "applied"
        if call_id in self._timed_out_calls:
            return "unknown"
        return "settled"

    # --------------------------------------------------------- judge bridge
    def judge_bridge_task(self):
        """Hand the task-environment handle to adapters/judge.py, the only
        module permitted to read the simulator judge (AGENTS rule 1). This
        method itself never touches the judge."""
        return self._task._task

    # --------------------------------------------------------- task anchors
    def grasp_target_xyz(self, obj_id: Optional[str] = None) -> list:
        obj_id = obj_id or self.spec["target"]
        pos = self._pyrep_objects[obj_id].get_position()
        return [float(pos[0]), float(pos[1]), float(pos[2]) + self._cup_top_offset]

    def place_target_xyz(self) -> list:
        obj = self._pyrep_objects[self.spec["container"]]
        pos = obj.get_position()
        return [float(pos[0]), float(pos[1]), float(pos[2])]
