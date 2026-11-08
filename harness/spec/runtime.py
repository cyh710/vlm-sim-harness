"""Nine-state runtime loop (S0..S8 + unified EXIT).

Invariants enforced here (AGENTS section 2 / runtime spec section 0):
  - no blind re-send: a call whose outcome is UNKNOWN is re-VERIFIED, never
    re-dispatched (F2 scenario must yield physical_grasp_attempts == 1)
  - done does NOT end the episode: it triggers global goal acceptance at S5;
    failure returns to S2
  - only S6 writes the Progress Ledger, and only verified effects
  - budgets charge before consumption; exhaustion exits with facts preserved
  - single-threaded, deterministic; wall clock only ever feeds the budget/report
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import harness.spec.dsl as dsl
from harness.spec.budget import Budget, BudgetExhausted
from harness.spec.ledger import AttemptLedger, ProgressLedger
from harness.spec.memory import Fact, Memory
from harness.spec.registry import Registry
from harness.spec.trace import Trace
from harness.spec.types import (CallKind, CallState, Evidence, ExitReason,
                                ExecutionResult, PlannerProtocolError,
                                ProgressDelta, ReasonCode, Status, Strategy,
                                ToolCall, Verdict, WorldState)
from harness.spec.validator import ValidationResult, Validator
from harness.spec.verifier import Verifier

QUERY_ITEM = {"observe": "queries", "where_is": "queries",
              "list_objects": "queries", "wait": "queries"}


@dataclass
class PlannerContext:
    world: WorldState
    memory_text: str
    progress_summaries: tuple[str, ...]
    budget_remaining: dict
    last_event: dict            # {"type": "verify"|"reject"|"accept", ...}
    recovery_hint: str = ""
    # §9.1 recent_attempts view (call_id/name/result), newest last; the
    # runtime builds it from the Attempt Ledger — planners never read it
    # directly, so scripted and LLM backends see the identical view.
    recent_attempts: tuple[dict, ...] = ()


class ScriptedPlannerRanOut(Exception):
    pass


class Runtime:
    def __init__(self, *, adapter, planner, registry: Registry, budget: Budget,
                 trace: Trace, validator: Validator, verifier: Verifier,
                 memory: Memory | None = None,
                 max_no_progress_cycles: int = 4,
                 hard_iteration_cap: int = 200,
                 # M5 arm switches (plan §6.1). Defaults = group C (full);
                 # arms differ ONLY on these code paths — model, skills,
                 # perception, budget caps, safety and seeds are shared.
                 s5_verifier: bool = True,       # A: False — self-report drives
                 recovery: bool = True,          # B: False — no S7 R0..R2
                 replan_enabled: bool = True,    # C1: False — no S8
                 progress_ledger: bool = True):  # B/C1: False — no S6 commits
        self.adapter = adapter
        self.planner = planner
        self.registry = registry
        self.budget = budget
        self.trace = trace
        self.validator = validator
        self.verifier = verifier
        self.memory = memory or Memory()
        self.max_no_progress_cycles = max_no_progress_cycles
        self.hard_iteration_cap = hard_iteration_cap
        self.s5_verifier = s5_verifier
        self.recovery = recovery
        self.replan_enabled = replan_enabled
        self.progress_ledger = progress_ledger

        self.attempts = AttemptLedger()
        self.progress = ProgressLedger()
        self.world: WorldState | None = None
        self.cycle = 0
        self._stall_cycles = 0
        self._pending_call: ToolCall | None = None
        self._recover_handled_refresh = False
        self._protocol_error_streak = 0

    def _recent_attempts_view(self, n: int = 5) -> tuple[dict, ...]:
        """§9.1 recent_attempts: last n attempts, attempt order preserved."""
        out = []
        for rec in self.attempts.all_records()[-n:]:
            settled = rec.state is CallState.SETTLED and rec.status is not None
            out.append({"call_id": rec.call_id, "name": rec.name,
                        "result": rec.status.value if settled
                        else rec.state.value})
        return tuple(out)

    # ------------------------------------------------------------------ run
    def run(self, seed: int) -> "RunSummary":
        t0 = time.time()
        # ---- S0 INIT ------------------------------------------------------
        self.trace.log("S0", "init", name="reset")
        self.adapter.reset(seed)                 # adapter sets seed FIRST (rule 7)
        self.world = self._observe()             # also seeds memory facts

        exit_reason = ExitReason.FAILED
        online_claim_success = False
        last_event: dict = {}
        last_commit_cycle = 0

        while self.cycle < self.hard_iteration_cap:
            self.cycle += 1

            # no-progress watchdog: N cycles without any new verified commit
            if self.cycle - last_commit_cycle >= self.max_no_progress_cycles \
                    and self.cycle > 1:
                self.trace.log("EXIT", "no_progress", name="watchdog",
                               budget=self.budget.remaining())
                exit_reason, online_claim_success = self._exit(ExitReason.FAILED)
                break

            # ---- S1 OBSERVE (refresh when flagged) -------------------------
            if last_event.get("refresh"):
                self.world = self._observe()

            # ---- S2 DECIDE ---------------------------------------------------
            try:
                call = self._decide(last_event)
            except PlannerProtocolError as exc:
                self.trace.log("S2", "planner_protocol_error", call_id="-",
                               name="decide",
                               reason_code="PLANNER_PROTOCOL_ERROR",
                               detail=str(exc)[:200],
                               budget=self.budget.remaining())
                if self._protocol_error_streak >= 3 or \
                        not self.budget.can_charge("recoveries"):
                    self.trace.log("EXIT", "planner_gave_up",
                                   name="protocol_error",
                                   budget=self.budget.remaining())
                    exit_reason, online_claim_success = \
                        self._exit(ExitReason.FAILED)
                    break
                self.budget.charge("recoveries")
                last_event = {"type": "reject", "refresh": True}
                continue
            self._protocol_error_streak = 0

            # ---- S3 VALIDATE ---------------------------------------------------
            result = self.validator.validate(call, self.world, self.budget,
                                             self.memory)
            self.trace.log("S3", "validate", call_id=call.call_id, kind=call.kind,
                           name=call.name, verdict="allowed" if result.allowed else "rejected",
                           reason_code=result.reason_code,
                           budget=self.budget.remaining())
            if not result.allowed:
                branch = self._on_rejection(call, result, last_event)
                if branch is not None:      # an EXIT was decided
                    exit_reason, online_claim_success = branch
                    break
                last_event = {"type": "reject", "call": call,
                              "validation": result,
                              "refresh": result.suggested_next == "S1"}
                continue

            # ---- S4 EXECUTE (queries fast-path back to S1) -----------------
            if call.kind == CallKind.QUERY.value:
                self._execute_query(call)
                last_event = {"type": "query", "name": call.name}
                continue

            if call.kind == CallKind.CONTROL.value:
                branch = self._on_control(call, last_event)
                if branch is None:
                    last_event = {"type": "control", "name": call.name,
                                  "accepted": False}
                    continue
                exit_reason, online_claim_success = branch
                break

            evidence = self._execute_action(call)
            self.budget.add_wall_time(evidence.metrics.get("elapsed_s", 0.0))

            # ---- arm A: no S5 — executor self-report drives continue/end ----
            if not self.s5_verifier:
                claimed = evidence.metrics.get("status", "unknown")
                if claimed in ("reached", "success"):
                    # self-report IS A's progress signal; keeps the
                    # no-progress watchdog equivalent across arms
                    last_commit_cycle = self.cycle
                self.trace.log("S4", "self_report", call_id=call.call_id,
                               kind=call.kind, name=call.name,
                               verdict=str(claimed),
                               budget=self.budget.remaining())
                last_event = {"type": "verify", "claimed": claimed,
                              "evidence": evidence, "refresh": True}
                continue

            # ---- S5 VERIFY (arms B / C1 / C) ---------------------------------
            verification = self.verifier.verify(
                evidence, self.registry.expand_effects(call.name, call.args))
            if evidence.timeout_exceeded:
                # audit trail: outcome was UNKNOWN at receipt; this SUCCESS is
                # evidence-based re-verification, not a blind re-send
                self.trace.log("S5", "reverify_after_timeout",
                               call_id=call.call_id, name=call.name,
                               verdict=verification.verdict.value,
                               reason_code=verification.reason_code)
            self.trace.log("S5", "verify", call_id=call.call_id, kind=call.kind,
                           name=call.name, verdict=verification.verdict.value,
                           reason_code=verification.reason_code,
                           evidence_refs=evidence.evidence_refs,
                           latency_s=verification.latency_s,
                           budget=self.budget.remaining())

            if verification.verdict in (Verdict.SUCCESS, Verdict.PARTIAL):
                # ---- S6 COMMIT (verified subset only; arm C only) ----------
                if self.progress_ledger:
                    committed = self._commit(call, evidence, verification)
                    if committed > 0:
                        last_commit_cycle = self.cycle
                else:
                    # no-ledger arms (A/B/C1): the VERIFIED verdict itself is
                    # the progress signal feeding the no-progress watchdog —
                    # keeps watchdog semantics equivalent across arms
                    last_commit_cycle = self.cycle
                last_event = {"type": "verify", "verdict": verification,
                              "evidence": evidence, "refresh": True}
                continue

            if verification.verdict is Verdict.UNSAFE:
                exit_reason, online_claim_success = self._exit(ExitReason.SAFETY_STOP)
                break

            # FAILED / INCONCLUSIVE -> arm C1/C: S7 RECOVER; arm B: the honest
            # verdict goes back to the planner as plain feedback (no S7)
            if self.recovery:
                branch = self._recover(call, evidence, verification, last_event)
                if branch is not None:
                    exit_reason, online_claim_success = branch
                    break
                last_event = {"type": "verify", "verdict": verification,
                              "evidence": evidence,
                              "refresh": not self._recover_handled_refresh}
                self._recover_handled_refresh = False
            else:
                last_event = {"type": "verify", "verdict": verification,
                              "evidence": evidence, "refresh": True}

        else:
            exit_reason = ExitReason.FAILED   # hard cap: treat as failure

        wall = time.time() - t0
        self.budget.add_wall_time(wall)
        self.trace.log("EXIT", "exit", name=exit_reason.value,
                       verdict="success" if online_claim_success else None,
                       budget=self.budget.remaining())
        return RunSummary(
            exit_reason=exit_reason,
            online_claim_success=online_claim_success,
            attempts=self.attempts, progress=self.progress,
            budget=self.budget, trace=self.trace, wall_time_s=wall)

    # ------------------------------------------------------------- helpers
    def _exit(self, reason: ExitReason, claim: bool = False):
        return reason, claim

    def _observe(self) -> WorldState:
        world = self.adapter.observe()
        self.memory.upsert_fact(Fact(key="gripper", value={"open": world.gripper_open},
                                     world_version=world.version))
        for obj in world.objects:
            self.memory.upsert_fact(Fact(key=f"pose:{obj.id}",
                                         value={"x": obj.pose.x, "y": obj.pose.y,
                                                "z": obj.pose.z},
                                         world_version=world.version))
        return world

    def _decide(self, last_event: dict) -> ToolCall:
        self.budget.check_available("llm_calls")
        self.budget.charge("llm_calls")
        if self._pending_call is not None:
            # R1 recovery call produced by planner.recover_local in S7
            call = self._pending_call
            self._pending_call = None
            return self._with_call_id(self._restamp(call))
        ctx = PlannerContext(
            world=self.world,
            memory_text=self.memory.render(),
            progress_summaries=self.progress.verified_effect_summaries(),
            budget_remaining=self.budget.remaining(),
            last_event={k: v for k, v in last_event.items()
                        if k not in ("verdict", "evidence")},
            recent_attempts=self._recent_attempts_view(),
            )
        # give scripted/LLM planner the full last verdict for reactive rules
        ctx.last_event["last_verdict"] = _verdict_view(last_event.get("verdict"))
        ctx.last_event["last_validation"] = _validation_view(
            last_event.get("validation"), last_event.get("call"))
        try:
            call = self.planner.decide(ctx)
        except PlannerProtocolError as exc:
            # design §9.1: one in-planner repair already happened; this is the
            # S7 escalation. Budget stays charged (decision consumed a slot).
            self._protocol_error_streak += 1
            raise
        self._protocol_error_streak = 0
        call = self._restamp(call)
        return self._with_call_id(call)

    def _restamp(self, call: ToolCall) -> ToolCall:
        if call.world_version == 0:            # planner did not stamp version
            return ToolCall(call_id=call.call_id, kind=call.kind, name=call.name,
                            args=call.args, world_version=self.world.version,
                            expected_effects=call.expected_effects,
                            timeout_ms=call.timeout_ms)
        return call

    def _with_call_id(self, call: ToolCall) -> ToolCall:
        if call.call_id:
            return call
        return ToolCall(call_id=f"c_{self.cycle:04d}", kind=call.kind,
                        name=call.name, args=call.args,
                        world_version=call.world_version,
                        expected_effects=call.expected_effects,
                        timeout_ms=call.timeout_ms)

    def _on_rejection(self, call: ToolCall, result: ValidationResult,
                      last_event: dict):
        """Returns (exit_reason, claim) tuple when the loop must EXIT, else
        None to continue (optionally with refresh flag)."""
        code = result.reason_code
        if code == "BUDGET_EXHAUSTED":
            self.trace.log("EXIT", "budget_exhausted", call_id=call.call_id,
                           name=call.name, reason_code=code,
                           budget=self.budget.remaining())
            return self._exit(ExitReason.BUDGET_EXHAUSTED)
        if code == ReasonCode.COLLISION_RISK and not result.recoverable:
            self.trace.log("EXIT", "safety_stop", call_id=call.call_id,
                           name=call.name, reason_code=code)
            return self._exit(ExitReason.SAFETY_STOP)
        # SCHEMA_INVALID / PRECONDITION_UNMET / PERCEPTION_STALE: planner retry
        self.budget.charge("recoveries")
        return None

    def _execute_query(self, call: ToolCall) -> None:
        self.budget.check_available("queries")
        self.budget.charge(QUERY_ITEM.get(call.name, "queries"))
        self.attempts.record_dispatch(call.call_id, call.name, call.args,
                                      started_at=time.time())
        result = self.adapter.execute(call)
        self.attempts.settle(call.call_id, result.status, ended_at=time.time())
        self.world = self.adapter.observe()
        self.trace.log("S4", "dispatch", call_id=call.call_id, kind=call.kind,
                       name=call.name, verdict=result.status.value,
                       args=dict(call.args))

    def _on_control(self, call: ToolCall, last_event: dict):
        if call.name == "abort":
            self.trace.log("S5", "abort", call_id=call.call_id, name="abort")
            return self._exit(ExitReason.ABORTED)
        if call.name == "replan":
            if not self.replan_enabled:
                # arm B/C1: S8 is not part of the arm; tell the planner no
                self.trace.log("S8", "replan_disabled", call_id=call.call_id,
                               name="replan", budget=self.budget.remaining())
                return None
            self.budget.check_available("replans")
            self.budget.charge("replans")
            self.memory.invalidate_all_pose_facts()
            self.trace.log("S8", "replan", call_id=call.call_id, name="replan")
            return None
        if call.name == "done":
            if not self.s5_verifier:
                # arm A (plan §6.1): executor self-report drives the end —
                # done ends the episode with the agent's own claim; the
                # independent global acceptance below IS the B+ delta
                self.trace.log("EXIT", "done_selfreport", call_id=call.call_id,
                               name="done", budget=self.budget.remaining())
                return self._exit(ExitReason.SUCCESS, claim=True)
            # invariant 5: done does NOT end the episode; global acceptance now.
            # Transition predicates (displacement_*) are honoured via committed
            # Progress entries (design section 14: verified progress is
            # history); state predicates re-verify on the CURRENT world.
            goal = self.registry.task_goal
            if goal is None:
                raise ValueError("done issued but registry has no task_goal")
            accepted = self._goal_accepted(goal)
            self.trace.log("S5", "global_accept", call_id=call.call_id,
                           name="done", verdict="success" if accepted else "failed",
                           budget=self.budget.remaining())
            if accepted:
                self.memory.write_summary("task goal verified at done")
                return self._exit(ExitReason.SUCCESS, claim=True)
            # unmet -> back to S2 with explicit unmet feedback (patch: done ->
            # global verification -> S2/S7, never direct exit)
            self.budget.charge("recoveries")
            last_event["goal_unmet"] = dsl.summarize(goal)
            return None
        raise ScriptedPlannerRanOut(f"unknown control {call.name}")

    def _execute_action(self, call: ToolCall) -> Evidence:
        before = self.world
        self.budget.check_available("physical_actions")
        self.budget.charge("physical_actions")
        rec = self.attempts.record_dispatch(call.call_id, call.name, call.args,
                                            started_at=time.time())
        self.trace.log("S4", "dispatch", call_id=call.call_id,
                       attempt_id=rec.attempt_id, kind=call.kind, name=call.name,
                       args=dict(call.args),
                       budget=self.budget.remaining())
        result: ExecutionResult = self.adapter.execute(call)
        after = self.adapter.observe()

        timed_out = result.status is Status.TIMEOUT
        if timed_out:
            # outcome unknowable -> UNKNOWN; NEVER re-send this call_id
            self.attempts.mark_unknown(call.call_id, ended_at=time.time())
        else:
            self.attempts.settle(call.call_id, result.status, ended_at=time.time())

        self.trace.log("S4", "feedback", call_id=call.call_id,
                       attempt_id=rec.attempt_id, kind=call.kind, name=call.name,
                       verdict=result.status.value)

        return Evidence(
            call_id=call.call_id, action=call.name, args=dict(call.args),
            expected_effects=tuple(call.expected_effects),
            before=before, after=after,
            metrics={"elapsed_s": result.elapsed_ms / 1000.0,
                     "status": result.status.value,
                     "timeout_exceeded": timed_out},
            claimed=dict(result.feedback.get("claimed", {})),
            timeout_exceeded=timed_out,
            evidence_refs=result.evidence_refs)

    def _commit(self, call: ToolCall, evidence: Evidence,
                verification: Verdict) -> int:
        """S6: commit only LEAF effects verified True (PARTIAL must never
        smuggle unverified effects into the Progress Ledger)."""
        committed = 0
        for ast in self.registry.expand_effects(call.name, call.args):
            for leaf in dsl.leaf_nodes(ast):
                _, overall = dsl.evaluate(leaf, evidence.before, evidence.after)
                if overall is True:
                    self.progress.commit(ProgressDelta(
                        verified_effects=(leaf,),
                        source_call_id=call.call_id,
                        evidence_refs=evidence.evidence_refs))
                    committed += 1
        self.memory.write_summary(
            f"{call.name}({call.args.get('object_id', '')}) -> "
            f"{verification.verdict.value}")
        self.trace.log("S6", "commit", call_id=call.call_id, kind=call.kind,
                       name=call.name, verdict=verification.verdict.value,
                       evidence_refs=evidence.evidence_refs,
                       budget=self.budget.remaining())
        return committed

    # --------------------------------------------------------- global goal
    _TRANSITION_OPS = {"displacement_min", "disp_projection_min"}
    _R1_CODES = {ReasonCode.NOT_HOLDING, ReasonCode.NO_CONTACT,
                 ReasonCode.GRASP_SLIPPED, ReasonCode.LIFT_NOT_ENOUGH}

    def _r1_local_fix(self, call: ToolCall, verification, last_event: dict):
        """R1: re-observe, ask the planner for an adjusted retry of the same
        skill; None -> escalate to replan (R2)."""
        self.world = self._observe()
        # accounting rule (planner.py): every planner call is one llm_calls
        # slot regardless of backend — S7 must pay like S2 does. No slot left
        # -> no planner call; escalate to replan which exits on its own.
        recovery_call = None
        if self.budget.can_charge("llm_calls"):
            self.budget.charge("llm_calls")
            try:
                recovery_call = self.planner.recover_local(PlannerContext(
                    world=self.world, memory_text=self.memory.render(),
                    progress_summaries=self.progress.verified_effect_summaries(),
                    budget_remaining=self.budget.remaining(),
                    last_event={"failed_call": call,
                                "reason_code": verification.reason_code},
                    recovery_hint="R1",
                    recent_attempts=self._recent_attempts_view()))
            except PlannerProtocolError as exc:
                self.trace.log("S7", "planner_protocol_error",
                               call_id=call.call_id, name="recover_local",
                               reason_code="PLANNER_PROTOCOL_ERROR",
                               detail=str(exc)[:200],
                               budget=self.budget.remaining())
                recovery_call = None      # R1 -> escalate to replan (R2)
        if recovery_call is None:
            return self._replan_branch(call, verification.reason_code, last_event)
        self._pending_call = recovery_call
        self._recover_handled_refresh = True   # already re-observed here
        return None

    def _progress_has_effect(self, leaf: dict) -> bool:
        import harness.spec.dsl as _dsl
        want = _dsl.leaf_nodes(leaf)[0]
        op, oid = want["op"], want.get("args", {}).get("object_id")
        for entry in self.progress.entries():
            for fx in entry.verified_effects:
                if fx.get("op") == op and \
                        fx.get("args", {}).get("object_id") == oid:
                    return True
        return False

    def _goal_accepted(self, goal: dict) -> bool:
        op = goal["op"]
        if op in dsl.COMPOSITES:
            subs = [self._goal_accepted(c) for c in goal["children"]]
            return all(subs) if op == "all_of" else any(subs)
        if op in self._TRANSITION_OPS and self._progress_has_effect(goal):
            return True     # verified history counts (displacement is monotonic)
        _, overall = dsl.evaluate(goal, self.world, self.world)
        return overall is True

    def _watchdog_timed_out(self, committed_now: int) -> bool:
        """Legacy per-commit stall check (kept for unit tests); the loop-level
        no-progress watchdog is cycle-based."""
        if committed_now > 0:
            self._stall_cycles = 0
            return False
        self._stall_cycles = getattr(self, "_stall_cycles", 0) + 1
        return self._stall_cycles >= self.max_no_progress_cycles

    def _recover(self, call: ToolCall, evidence: Evidence,
                 verification, last_event: dict):
        """S7: map verdict+reason to R0..R3. Returns EXIT tuple or None."""
        self.budget.check_available("recoveries")
        self.budget.charge("recoveries")
        code = verification.reason_code

        # R0 for unknown outcomes: re-verify, NEVER re-dispatch (F2 invariant)
        if evidence.timeout_exceeded:
            status = self.adapter.query_action_status(call.call_id)
            if status == "applied":
                fresh = self.adapter.observe()
                new_evidence = Evidence(
                    call_id=evidence.call_id, action=evidence.action,
                    args=evidence.args,
                    expected_effects=evidence.expected_effects,
                    before=evidence.before, after=fresh,
                    metrics={**evidence.metrics, "reverified": True},
                    claimed=evidence.claimed)
                verification = self.verifier.verify(
                    new_evidence, self.registry.expand_effects(call.name, call.args))
                self.trace.log("S5", "reverify_after_timeout",
                               call_id=call.call_id, name=call.name,
                               verdict=verification.verdict.value,
                               reason_code=verification.reason_code)
                if verification.verdict is Verdict.SUCCESS:
                    self._commit(call, new_evidence, verification)
                    last_event["refresh"] = True
                    return None
                if verification.verdict is Verdict.FAILED and \
                        verification.reason_code in self._R1_CODES:
                    # abort was mid-skill; local fix (re-grasp) still applies
                    self.world = self._observe()
                    return self._r1_local_fix(call, verification, last_event)
                if verification.verdict is Verdict.FAILED:
                    last_event["refresh"] = True
                    return None
                # still inconclusive -> fall through to replan
                return self._replan_branch(call, code, last_event)
            # status unknown: refresh observation and re-verify evidence frames
            self.world = self._observe()
            verification = self.verifier.verify(
                evidence, self.registry.expand_effects(call.name, call.args))
            if verification.verdict is Verdict.SUCCESS:
                self._commit(call, evidence, verification)
                last_event["refresh"] = True
                return None
            if verification.verdict is Verdict.FAILED and \
                    verification.reason_code in self._R1_CODES:
                return self._r1_local_fix(call, verification, last_event)
            return self._replan_branch(call, code, last_event)

        if code in self._R1_CODES:
            # R1: local fix — planner proposes adjusted args for the same skill
            # (LIFT_NOT_ENOUGH: grasp caught but insufficient lift -> re-grasp
            # with a shifted point; the prior close left fingers shut and the
            # skill opens first)
            self.trace.log("S7", "recover", call_id=call.call_id, name="R1",
                           reason_code=code)
            return self._r1_local_fix(call, verification, last_event)

        if code in (ReasonCode.OBJECT_OCCLUDED, ReasonCode.OBJECT_NOT_VISIBLE):
            # R2: change viewpoint / re-plan
            self.trace.log("S7", "recover", call_id=call.call_id, name="R2",
                           reason_code=code)
            return self._replan_branch(call, code, last_event)

        if code in (ReasonCode.HARDWARE_FAULT,):
            self.trace.log("S7", "recover", call_id=call.call_id, name="R3",
                           reason_code=code)
            return self._exit(ExitReason.FAILED)

        # generic: R2 replan
        return self._replan_branch(call, code, last_event)

    def _replan_branch(self, call: ToolCall, code: str | None,
                       last_event: dict):
        if not self.replan_enabled:
            # arm C1: R2 escalation has nowhere to go -> planner gave up
            self.trace.log("EXIT", "replan_disabled", name="planner_gave_up",
                           reason_code=code, budget=self.budget.remaining())
            return self._exit(ExitReason.FAILED)
        self.budget.check_available("replans")
        self.budget.charge("replans")
        self.memory.invalidate_all_pose_facts()
        self.world = self._observe()
        # accounting rule (planner.py): S8 planner calls pay llm_calls too;
        # no slot left -> planner cannot be asked -> give up with facts kept
        replan_call = None
        if self.budget.can_charge("llm_calls"):
            self.budget.charge("llm_calls")
            try:
                replan_call = self.planner.replan(PlannerContext(
                    world=self.world, memory_text=self.memory.render(),
                    progress_summaries=self.progress.verified_effect_summaries(),
                    budget_remaining=self.budget.remaining(),
                    last_event={"failed_call": call, "reason_code": code},
                    recovery_hint="R2",
                    recent_attempts=self._recent_attempts_view()))
            except PlannerProtocolError as exc:
                self.trace.log("S8", "planner_protocol_error",
                               call_id=call.call_id, name="replan",
                               reason_code="PLANNER_PROTOCOL_ERROR",
                               detail=str(exc)[:200],
                               budget=self.budget.remaining())
                replan_call = None
        self.trace.log("S8", "replan", call_id=call.call_id, name="replan",
                       reason_code=code)
        if replan_call is None:
            self.trace.log("EXIT", "replan_exhausted", name="planner_gave_up")
            return self._exit(ExitReason.FAILED)
        last_event["refresh"] = True
        return None

    # ------------------------------------------------------------ summary
    def _check_no_oracle(self):     # pragma: no cover - CI static check instead
        raise NotImplementedError("oracle access is forbidden in harness/")


@dataclass
class RunSummary:
    exit_reason: ExitReason
    online_claim_success: bool
    attempts: AttemptLedger
    progress: ProgressLedger
    budget: Budget
    trace: Trace
    wall_time_s: float

    def grasp_attempts(self) -> int:
        return self.attempts.count_name("grasp")

    def physical_actions(self) -> int:
        return self.budget.used["physical_actions"]

    def to_episode_result(self, run_id: str, episode_id: str,
                          config: dict) -> dict:
        from harness.spec.types import EpisodeResult
        er = EpisodeResult(
            exit_reason=self.exit_reason,
            online_claim_success=self.online_claim_success,
            physical_actions=self.budget.used["physical_actions"],
            grasp_attempts=self.attempts.count_name("grasp"),
            queries=self.budget.used["queries"],
            llm_calls=self.budget.used["llm_calls"],
            replans=self.budget.used["replans"],
            recovery_attempts=self.budget.used["recoveries"],
            progress_entries=len(self.progress.entries()),
            trace_events=len(self.trace.events),
            wall_time_s=self.wall_time_s)
        return er.to_result_dict(run_id, episode_id, config)


def _verdict_view(verdict):
    if verdict is None:
        return None
    return {"verdict": verdict.verdict.value if hasattr(verdict, "verdict") else None,
            "reason_code": getattr(verdict, "reason_code", None)}


def _validation_view(validation, call=None):
    if validation is None:
        return None
    return {"allowed": validation.allowed,
            "reason_code": validation.reason_code,
            "suggested_next": validation.suggested_next,
            "rejected_name": getattr(call, "name", None),
            "details": validation.details}
