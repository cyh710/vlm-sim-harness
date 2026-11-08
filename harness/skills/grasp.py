"""grasp skill: open -> pre-grasp hover -> descend to grasp frame -> close -> lift.

Zero success judgement (AGENTS rule 3): motion + phase log only.

The caller (adapter) passes the GRASP FRAME — object frame + per-object palm
offset from TASK_SPECS (cup rim band +0.06; small block center +0.0; both from
scripts/probe_grasp_heights.py sweeps). The skill adds NO offset of its own.

Calibration facts baked into constants:
  - pregrasp +0.12 above the grasp frame: higher pregrasps stretch the arm
    into configurations that silently stall on joint limits mid-descend
  - lift +0.30 above the frame: PickUpCup judge needs the cup outside the
    success-sensor zone (sensor resampled per episode)
"""
from __future__ import annotations

from harness.skills.motion import run_phases

PREGRASP_ABOVE_GRASP_M = 0.12
CLOSE_WHILE_DESCEND_FROM = 0.05  # start closing this far above the grasp frame
SETTLE_STEPS = 3
LIFT_ABOVE_GRASP_M = 0.30


def run_grasp(step_fn, grasp_xyz: list, regrasp_fn=None,
              lift_above: float = LIFT_ABOVE_GRASP_M) -> dict:
    """regrasp_fn: optional backend primitive that re-fires the gripper's
    attach attempt on nearby graspables (PyRep gripper.grasp(), which checks
    the attach proximity sensor internally — no-op when nothing is in range).
    The Discrete gripper mode only fires its attach loop on the open->close
    TRANSITION, so the settle steps between closes are the place to re-fire:
    this lifted the flaky-layout attach rate in M2 (physics is
    launch-nondeterministic, ~83% per attempt)."""
    x, y, z = grasp_xyz

    def _move_to(px, py, pz, gripper):
        def fn():
            return step_fn([px, py, pz], gripper)
        return fn

    def _open_in_place():
        # action_xyz=None means "keep current position" (adapter contract):
        # command gripper open without moving
        return lambda: step_fn(None, 1.0)

    def _regrasp():
        def fn():
            if regrasp_fn is not None:
                regrasp_fn()
            return step_fn(None, 0.0)
        return fn

    phases = [
        # open FIRST: a failed previous close leaves the gripper shut; retry
        # must start from open fingers (skill-level, not a validator concern)
        ("open", _open_in_place()),
        ("pregrasp", _move_to(x, y, z + PREGRASP_ABOVE_GRASP_M, 1.0)),
        # two-stage descend: the final 5cm closes WHILE descending — fingers
        # wrap the rim gradually instead of a static close after a full-speed
        # descent that pushes the cup aside (M2 A/B: major attach-rate lever)
        ("descend_hover", _move_to(x, y, z + CLOSE_WHILE_DESCEND_FROM, 1.0)),
        ("descend_close", _move_to(x, y, z, 0.0)),
    ]
    for i in range(SETTLE_STEPS):
        phases.append((f"settle_{i}", _regrasp() if regrasp_fn else _move_to(x, y, z, 0.0)))
    phases.append(("lift", _move_to(x, y, z + lift_above, 0.0)))
    # one re-fire after the lift too: the attach sometimes lands late
    if regrasp_fn is not None:
        phases.append(("lift_regrasp", _regrasp()))

    return run_phases(phases)
