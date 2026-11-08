"""Independent Safety Supervisor (design doc section 13, patch P6).

Vetoes actions on workspace bounds / forbidden zones / joint limits.
Deliberately dumb and synchronous: the FSM asks it before dispatch; the FSM is
NOT an e-stop (patch P6) — the adapter still enforces hard limits itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from harness.spec.types import ToolCall, WorldState


@dataclass
class SafetyConfig:
    workspace_min: tuple[float, float, float] = (-0.5, -0.5, 0.0)
    workspace_max: tuple[float, float, float] = (0.5, 0.5, 0.6)
    # forbidden zones: list of boxes; any motion target inside -> COLLISION_RISK
    forbidden_zones: list[dict] = field(default_factory=list)
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)


class SafetySupervisor:
    def __init__(self, config: SafetyConfig):
        self.cfg = config

    def check_target(self, target: tuple[float, float, float]) -> tuple[bool, str]:
        x, y, z = target
        mn, mx = self.cfg.workspace_min, self.cfg.workspace_max
        if not (mn[0] <= x <= mx[0] and mn[1] <= y <= mx[1] and mn[2] <= z <= mx[2]):
            return False, "COLLISION_RISK"
        for zone in self.cfg.forbidden_zones:
            lo, hi = zone["min"], zone["max"]
            if (lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1] and lo[2] <= z <= hi[2]):
                return False, "COLLISION_RISK"
        return True, ""

    def check_call(self, call: ToolCall, world: WorldState) -> tuple[bool, str]:
        """Return (allowed, reason_code). Only kinematic targets of actions are
        checked here; queries/control pass through (S3 still bounds wait())."""
        if call.kind != "action":
            return True, ""
        args = call.args or {}
        target = args.get("target_xyz") or args.get("position")
        if isinstance(target, (list, tuple)) and len(target) == 3:
            ok, code = self.check_target(tuple(float(v) for v in target))
            if not ok:
                return False, code
        for jid, value in (args.get("joint_targets") or {}).items():
            if jid in self.cfg.joint_limits:
                lo, hi = self.cfg.joint_limits[jid]
                if not (lo <= float(value) <= hi):
                    return False, "COLLISION_RISK"
        return True, ""
