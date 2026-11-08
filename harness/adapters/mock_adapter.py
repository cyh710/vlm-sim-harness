"""FakeAdapter: deterministic mock backend with injectable faults.

Covers the 10-scenario matrix (experiment plan section 2.3). The adapter owns
a tiny physics model:
  - red_cup at a seeded pose; bin container; ee at home pose
  - grasp(obj): closes gripper, attaches cup to ee (if target near cup),
    lifts -> holding + displacement effects real in world state
  - place: releases into bin -> contained effect real
  - queries: observe/where_is/list_objects/wait are knowledge-only

Fault injection (constructor flags):
  grasp_empty_first        first grasp closes on nothing (F1 / grasp_empty_once)
  grasp_timeout_applied    grasp APPLIES physically but returns TIMEOUT
                           (F2 / timeout_after_effect) — the signature fault
  occluded_target          target visible_ratio 0 until move_camera action runs
  place_bounce_rim         place releases but cup ends at rim, half visible
                           (partial scenario: contained unverifiable)
  external_disturbance     observe() injects a one-shot world bump to trigger
                           stale detection on a pre-built call
  unsafe_zone              grasp target inside a forbidden zone -> safety veto
"""
from __future__ import annotations

import random

from harness.spec.types import (ObjInstance, Pose, Status, ToolCall,
                                WorldState)

_HOME_EE = Pose(x=0.30, y=0.0, z=0.25)
_BIN = dict(id="bin", label="rubbish bin",
            pose=Pose(x=-0.10, y=0.05, z=0.02),
            size=(0.20, 0.20, 0.12))


class FakeAdapter:
    def __init__(self, *, grasp_empty_first: bool = False,
                 grasp_timeout_applied: bool = False,
                 occluded_target: bool = False,
                 place_bounce_rim: bool = False,
                 external_disturbance: bool = False,
                 unsafe_zone: bool = False,
                 grasp_failure_count: int = 0):
        self.grasp_empty_first = grasp_empty_first
        self.grasp_timeout_applied = grasp_timeout_applied
        self.occluded_target = occluded_target
        self.place_bounce_rim = place_bounce_rim
        self.external_disturbance = external_disturbance
        self.unsafe_zone = unsafe_zone
        self.grasp_failure_count = grasp_failure_count

        self._rng = random.Random(0)
        self._version = 0
        self._clock = 1000.0
        self._ee = _HOME_EE
        self._gripper_open = 1.0
        self._held: str | None = None
        self._cup_pos = Pose(x=0.10, y=-0.05, z=0.05)
        self._cup_displaced = False        # lifted clear of support?
        self._bin_contained = False
        self._camera_moved = not occluded_target
        self._disturbance_pending = external_disturbance
        self._grasp_calls = 0
        self._applied_timeout_calls: set[str] = set()
        self._joint_delta = 0.0

    # ------------------------------------------------------------ lifecycle
    def reset(self, seed: int) -> None:
        """Rule 7: seed set FIRST, before any other state work."""
        random.seed(seed)
        self._rng = random.Random(seed)
        self._version = 1
        self._clock = 1000.0
        self._ee = _HOME_EE
        self._gripper_open = 1.0
        self._held = None
        # deterministic small layout jitter from the seed
        self._cup_pos = Pose(x=round(0.10 + self._rng.uniform(-0.01, 0.01), 4),
                             y=round(-0.05 + self._rng.uniform(-0.01, 0.01), 4),
                             z=0.05)
        self._cup_displaced = False
        self._bin_contained = False
        self._camera_moved = not self.occluded_target
        self._disturbance_pending = self.external_disturbance
        self._grasp_calls = 0
        self._applied_timeout_calls = set()
        self._joint_delta = 0.0

    def shutdown(self) -> None:
        pass

    # ------------------------------------------------------------ observation
    def observe(self) -> WorldState:
        self._version += 1
        if self._disturbance_pending:
            # external nudge: cup slides slightly; bumps version & ages facts
            self._disturbance_pending = False
            self._cup_pos = Pose(self._cup_pos.x + 0.02, self._cup_pos.y,
                                 self._cup_pos.z)
        visible = 1.0 if self._camera_moved else 0.0
        objects = [
            ObjInstance(id="red_cup", label="red cup", view="front",
                        pose=self._cup_pos, sigma_m=0.004,
                        size=(0.07, 0.07, 0.10),
                        visible_ratio=visible,
                        quality="ok", source="cad",
                        last_seen_t=self._clock),
            ObjInstance(id="bin", label="rubbish bin", view="front",
                        pose=_BIN["pose"], sigma_m=0.004, size=_BIN["size"],
                        visible_ratio=1.0, quality="ok", source="cad",
                        last_seen_t=self._clock),
        ]
        joints = {"joint_a": round(self._joint_delta, 4),
                  "joint_b": 0.0}
        return WorldState(version=self._version, timestamp=self._clock,
                          ee_pose=self._ee, gripper_open=self._gripper_open,
                          joints=joints, objects=tuple(objects),
                          unsafe_flag=False, evidence_refs=())

    # ------------------------------------------------------------ execution
    def execute(self, call: ToolCall):
        import time as _t
        t0 = _t.perf_counter()
        self._clock += 0.5
        result = self._dispatch(call)
        elapsed_ms = int(((_t.perf_counter() - t0) * 1000) or 1)
        from harness.spec.types import ExecutionResult
        return ExecutionResult(call_id=call.call_id, status=result[0],
                               feedback=result[1],
                               evidence_refs=(f"ev_{call.call_id}",),
                               elapsed_ms=elapsed_ms)

    def _dispatch(self, call: ToolCall):
        name, args = call.name, call.args

        if name == "observe" or name == "where_is" or name == "list_objects":
            self._version += 0            # knowledge-only; observe() bumps
            return Status.REACHED, {"objects": ["red_cup", "bin"]}

        if name == "wait":
            return Status.REACHED, {}

        if name == "move_camera":
            self._camera_moved = True
            tgt = args.get("target_xyz")
            if isinstance(tgt, (list, tuple)):
                self._ee = Pose(tgt[0], tgt[1], tgt[2])
            return Status.REACHED, {"claimed": {"success": True}}

        if name == "grasp":
            self._grasp_calls += 1
            obj_id = args.get("object_id", "red_cup")
            first = self._grasp_calls == 1
            if self.unsafe_zone:
                # adapter-level guard mirrors S3; never happens post-validation
                return Status.UNSAFE, {"claimed": {"success": False}}
            if self.grasp_empty_first and first:
                # empty-close signature (RLBench calibrated): fingers shut on
                # nothing -> open_amount reads 0.0
                self._gripper_open = 0.0
                return Status.REACHED, {"claimed": {"success": True,
                                                    "note": "closed on nothing"}}
            if self.grasp_timeout_applied and first:
                # PHYSICAL EFFECT APPLIED, receipt lost (F2)
                self._apply_grasp(obj_id)
                self._applied_timeout_calls.add(call.call_id)
                return Status.TIMEOUT, {"claimed": {"success": True}}
            self._apply_grasp(obj_id)
            return Status.REACHED, {"claimed": {"success": True}}

        if name == "place":
            if self._held != args.get("object_id"):
                return Status.BLOCKED, {"claimed": {"success": False,
                                                    "reason": "not_holding"}}
            self._ee = _BIN["pose"] if not isinstance(args.get("target_xyz"), list) \
                else Pose(*args["target_xyz"])
            self._gripper_open = 1.0
            if self.place_bounce_rim:
                # release happened (gripper open) but cup bounced outside the
                # bin's sensor-equivalent region (y offset > half-extent 0.10)
                # -> contained FAILS (non-counter), gripper_open verified
                self._held = None
                self._cup_pos = Pose(_BIN["pose"].x, _BIN["pose"].y - 0.12, 0.09)
                return Status.REACHED, {"claimed": {"success": True}}
            self._held = None
            self._cup_pos = Pose(_BIN["pose"].x, _BIN["pose"].y, _BIN["pose"].z + 0.06)
            self._bin_contained = True
            return Status.REACHED, {"claimed": {"success": True}}

        if name == "push":
            obj_id = args.get("object_id", "red_cup")
            if obj_id == "red_cup" and not self._held:
                direction = args.get("dir", [1.0, 0.0])
                self._cup_pos = Pose(self._cup_pos.x + 0.05 * direction[0],
                                     self._cup_pos.y + 0.05 * direction[1],
                                     self._cup_pos.z)
            return Status.REACHED, {"claimed": {"success": True}}

        return Status.NOT_IMPLEMENTED, {}

    def _apply_grasp(self, obj_id: str) -> None:
        # attach-carry signature (RLBench calibrated): fingers stay OPEN
        # (open_amount ~1.0) while the kinematic constraint carries the cup
        self._gripper_open = 1.0
        self._held = obj_id
        self._ee = Pose(self._cup_pos.x, self._cup_pos.y, 0.18)
        self._cup_pos = Pose(self._ee.x, self._ee.y, self._ee.z - 0.09)
        self._cup_displaced = True

    # ------------------------------------------------------------ F2 support
    def query_action_status(self, call_id: str) -> str:
        """Runtime asks (never assumes): did the timed-out call actually land?"""
        if call_id in self._applied_timeout_calls:
            return "applied"
        return "unknown"

    # ------------------------------------------------------------ helpers
    def perturb_world(self) -> None:
        """External disturbance hook for stale_pose scenarios."""
        self._disturbance_pending = True
