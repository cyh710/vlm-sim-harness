"""Motion primitives shared by skills.

The adapter owns the live RLBench TaskEnvironment; skills issue end-effector
pose steps through it and never judge outcomes (AGENTS rule 3). Every step
returns the post-step observation so the adapter can snapshot evidence.
"""
from __future__ import annotations

from typing import Callable, Optional

STEP_TIMEOUT_S = 20.0

# Top-down grasp orientation for the Panda in RLBench scenes: gripper z-axis
# pointing down, fingers closing along world x. Empirically validated on
# CoppeliaSim 4.1 + Panda (M2 smoke); recalibrate per-arm at M6.
TOP_DOWN_QUAT_XYZW = (0.0, 0.7071068, 0.0, 0.7071068)


def step_ee(step_fn: Callable, x: float, y: float, z: float,
            quat_xyzw: Optional[tuple], gripper: float) -> dict:
    """One EndEffectorPoseViaIK step. Returns the observation dict snapshot or
    raises MotionError with the backend reason."""
    action = [float(x), float(y), float(z)]
    if quat_xyzw is None:
        # keep current orientation: adapter injects it via partial_pose
        action = None
    try:
        obs = step_fn(action, gripper)
    except Exception as exc:  # IK failure / OOB / invalid action
        raise MotionError(f"{type(exc).__name__}: {exc}") from exc
    return obs


class MotionError(Exception):
    pass


def run_phases(phase_list: list, log: Optional[list] = None) -> dict:
    """Execute [(name, fn)] in order; fn returns an obs snapshot dict.

    First failure aborts the sequence and returns {"phase": name, "error": ...}
    so the adapter can mark BLOCKED/IK_FAILED without judging success.
    """
    evidence_phases = {}
    for name, fn in phase_list:
        try:
            evidence_phases[name] = fn()
        except MotionError as exc:
            if log is not None:
                log.append({"phase": name, "error": str(exc)})
            return {"aborted_at": name, "error": str(exc),
                    "phases": evidence_phases}
        if log is not None:
            log.append({"phase": name, "ok": True})
    return {"aborted_at": None, "phases": evidence_phases}
