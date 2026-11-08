"""place skill: transport hover -> descend -> release -> retreat.

Zero success judgement (AGENTS rule 3). Release detection belongs to the
verifier (gripper_open effect); this skill only produces motion.
"""
from __future__ import annotations

from harness.skills.motion import TOP_DOWN_QUAT_XYZW, run_phases

APPROACH_ABOVE_M = 0.10
RELEASE_DESCEND_M = 0.03
RETREAT_ABOVE_M = 0.15
OPEN_STEPS = 3


def run_place(step_fn, target_xyz: list, release_fn=None) -> dict:
    """release_fn: backend primitive that detaches the attached object
    (PyRep gripper.release()). Required on RLBench: the Discrete gripper mode
    skips its release branch when per-finger open_amount already reads 'open'
    during kinematic attach carry — observed M2 2026-10-29."""
    x, y, z = target_xyz

    def _move_to(px, py, pz, gripper):
        def fn():
            return step_fn([px, py, pz], gripper)
        return fn

    def _release():
        def fn():
            if release_fn is not None:
                release_fn()
            return step_fn(None, 1.0)
        return fn

    phases = [
        ("approach_above", _move_to(x, y, z + APPROACH_ABOVE_M, 0.0)),
        ("descend", _move_to(x, y, z + RELEASE_DESCEND_M, 0.0)),
        ("release", _release()),
    ]
    for i in range(OPEN_STEPS):
        phases.append((f"open_settle_{i}", _move_to(x, y, z + RELEASE_DESCEND_M, 1.0)))
    phases.append(("retreat", _move_to(x, y, z + RETREAT_ABOVE_M, 1.0)))

    return run_phases([(n, f) for n, f in phases])
