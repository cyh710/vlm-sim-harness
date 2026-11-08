"""Planner interface + ScriptedPlanner (M3 reference; LLMPlanner lands at M4).

Accounting rule: EVERY decide()/replan()/recover_local() call charges one
llm_calls budget slot regardless of backend, so all four experiment groups pay
the same price per decision (AGENTS rule: 预算口径必须四组一致).
"""
from __future__ import annotations

from typing import Callable, Optional

from harness.spec.runtime import PlannerContext, ScriptedPlannerRanOut
from harness.spec.types import CallKind, ToolCall


class Planner:
    def decide(self, ctx: PlannerContext) -> ToolCall:
        raise NotImplementedError

    def replan(self, ctx: PlannerContext) -> Optional[ToolCall]:
        """Called from S8 after a failure; return None to give up."""
        raise NotImplementedError

    def recover_local(self, ctx: PlannerContext) -> Optional[ToolCall]:
        """Called from S7 (R1) for local fixes; None -> escalate to replan."""
        return None


# ---------------------------------------------------------------------------

class ScriptedPlanner(Planner):
    """Deterministic step list + optional reactive overrides.

    steps: list of dicts:
        {"kind": "query"|"action"|"control", "name": str,
         "args": dict | callable(world)->dict,
         "repeat_if": optional predicate(ctx)->bool}
    overrides: list of (predicate(ctx)->bool, build(ctx)->ToolCall|None),
        evaluated in order BEFORE the next step; a matching override supplies
        the call instead (used for occluded -> move_camera, etc.). When an
        override fires it is consumed (one-shot) by default.
    give_up_after: when replan() is asked more than this many times, return
        None (planner declares failure).
    """

    def __init__(self, steps: list[dict], overrides: list | None = None,
                 give_up_after: int = 2):
        self.steps = list(steps)
        self.overrides = list(overrides or [])
        self.give_up_after = give_up_after
        self._replans = 0
        self._step_idx = 0
        self._current_step: dict | None = None
        self._last_call_from_step = True

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _build_call(step: dict, ctx: PlannerContext,
                    call_id: str) -> ToolCall:
        args = step.get("args", {})
        if callable(args):
            args = args(ctx.world)
        else:
            # per-key callables resolve against the current world
            args = {k: (v(ctx.world) if callable(v) else v)
                    for k, v in args.items()}
        version = ctx.world.version
        # stale-pose simulation: a step may propose a call based on an outdated
        # world version for its first N attempts (apply_lag_times)
        lag = step.get("lag_versions")
        if lag:
            applied = step.setdefault("_lag_applied", 0)
            if applied < int(step.get("apply_lag_times", 1)):
                step["_lag_applied"] = applied + 1
                version = max(1, ctx.world.version - int(lag))
        return ToolCall(
            call_id=call_id,
            kind=step.get("kind", CallKind.ACTION.value),
            name=step["name"],
            args=dict(args),
            world_version=version,
            timeout_ms=step.get("timeout_ms", 5000))

    def _next_step(self, ctx: PlannerContext) -> dict:
        while self._step_idx < len(self.steps):
            step = self.steps[self._step_idx]
            repeat_if = step.get("repeat_if")
            if callable(repeat_if) and repeat_if(ctx):
                return step
            self._step_idx += 1
            return step
        raise ScriptedPlannerRanOut("scripted plan exhausted")

    def _override_call(self, ctx: PlannerContext,
                       call_id: str) -> Optional[ToolCall]:
        for i, (predicate, builder) in enumerate(self.overrides):
            try:
                hit = predicate(ctx)
            except Exception:
                hit = False
            if hit:
                self.overrides.pop(i)
                built = builder(ctx)
                if built is not None:
                    return ToolCall(call_id=call_id, kind=built.kind,
                                    name=built.name, args=built.args,
                                    world_version=ctx.world.version,
                                    timeout_ms=built.timeout_ms)
        return None

    # -- planner API -----------------------------------------------------------
    def decide(self, ctx: PlannerContext) -> ToolCall:
        call_id = f"c_sp_{self._step_idx:02d}_{ctx.world.version:03d}"
        overridden = self._override_call(ctx, call_id)
        if overridden is not None:
            # override calls (e.g. move_camera) don't consume the plan step;
            # the pending step is rebuilt on the next decide
            self._last_call_from_step = False
            return overridden
        # retry semantics: if the last validation REJECTED this step's call,
        # do not consume the step — rebuild it against the refreshed world
        last_val = ctx.last_event.get("last_validation") or {}
        rejected = (last_val.get("allowed") is False
                    and last_val.get("rejected_name") is not None)
        if self._current_step is None:
            try:
                self._current_step = self._next_step(ctx)
            except ScriptedPlannerRanOut:
                self._current_step = {"kind": "control", "name": "done", "args": {}}
        elif rejected and self._last_call_from_step and \
                last_val.get("rejected_name") == self._current_step.get("name"):
            pass                                    # retry the same step
        elif self._last_call_from_step:
            try:                                    # step accepted -> advance
                self._current_step = self._next_step(ctx)
            except ScriptedPlannerRanOut:
                self._current_step = {"kind": "control", "name": "done", "args": {}}
        # else: an override call intervened; keep the pending step
        self._last_call_from_step = True
        return self._build_call(self._current_step, ctx, call_id)

    def replan(self, ctx: PlannerContext) -> Optional[ToolCall]:
        self._replans += 1
        if self._replans > self.give_up_after:
            return None
        # keep executing remaining plan against the refreshed world
        call_id = f"c_rp_{self._replans:02d}_{ctx.world.version:03d}"
        overridden = self._override_call(ctx, call_id)
        if overridden is not None:
            return overridden
        try:
            step = self._next_step(ctx)
        except ScriptedPlannerRanOut:
            return None
        return self._build_call(step, ctx, call_id)

    def recover_local(self, ctx: PlannerContext) -> Optional[ToolCall]:
        """R1: retry the failed skill with adjusted args (e.g. shifted grasp
        point). Falls back to None (escalate to replan) if the failed skill
        already retried twice."""
        failed = ctx.last_event.get("failed_call")
        if failed is None:
            return None
        retries = ctx.last_event.get("r1_retries", 0)
        if retries >= 2:
            return None
        args = dict(failed.args)
        # alternate shift axis across retries (x then y): a single-axis nudge
        # can keep landing in the same dead zone
        shift = [0.02, 0.0] if retries % 2 == 0 else [0.0, 0.02]
        tgt = args.get("target_xyz")
        if isinstance(tgt, (list, tuple)) and len(tgt) == 3:
            args["target_xyz"] = [tgt[0] + shift[0], tgt[1] + shift[1], tgt[2]]
        ctx.last_event["r1_retries"] = retries + 1
        call_id = f"c_r1_{retries + 1:02d}_{ctx.world.version:03d}"
        return ToolCall(call_id=call_id, kind=failed.kind, name=failed.name,
                        args=args, world_version=ctx.world.version,
                        timeout_ms=failed.timeout_ms)
