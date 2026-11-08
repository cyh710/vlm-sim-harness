"""S3 Validator: fail-closed checks in the specified order (design doc section
12 + patch P3). Query vs action differentiated: queries skip freshness and
preconditions (their very purpose is refreshing stale facts — a freshness gate
on where_is would self-lock).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import harness.spec.dsl as dsl
from harness.spec.budget import Budget
from harness.spec.memory import Memory
from harness.spec.registry import Registry
from harness.spec.safety import SafetySupervisor
from harness.spec.types import CallKind, ToolCall, WorldState

QUERY_NAMES = {"observe", "where_is", "list_objects", "wait"}
CONTROL_NAMES = {"done", "abort", "replan"}
MAX_WAIT_S = 5.0


@dataclass
class ValidationResult:
    allowed: bool = False
    reason_code: str | None = None
    recoverable: bool = False
    suggested_next: str | None = None     # "S1" | "S2" | None
    details: dict = field(default_factory=dict)
    checked_world_version: int = 0


class Validator:
    def __init__(self, registry: Registry, safety: SafetySupervisor,
                 ttl_versions: int = 1, max_wait_s: float = MAX_WAIT_S):
        self.registry = registry
        self.safety = safety
        self.ttl_versions = ttl_versions
        self.max_wait_s = max_wait_s

    # ------------------------------------------------------------------
    def validate(self, call: ToolCall, world: WorldState, budget: Budget,
                 memory: Memory) -> ValidationResult:
        base = ValidationResult(checked_world_version=world.version)

        # 1. structure & identity (queries, actions, control all pass through)
        if call.kind not in (CallKind.QUERY.value, CallKind.ACTION.value,
                             CallKind.CONTROL.value):
            return self._deny(base, "SCHEMA_INVALID", recoverable=False,
                              details={"problem": f"unknown kind {call.kind!r}"})
        if call.kind == CallKind.QUERY.value and call.name not in QUERY_NAMES:
            return self._deny(base, "SCHEMA_INVALID", recoverable=True,
                              suggested_next="S2",
                              details={"problem": f"unknown query {call.name!r}"})
        if call.kind == CallKind.CONTROL.value and call.name not in CONTROL_NAMES:
            return self._deny(base, "SCHEMA_INVALID", recoverable=True,
                              suggested_next="S2",
                              details={"problem": f"unknown control {call.name!r}"})
        if call.kind == CallKind.ACTION.value:
            if not self.registry.has(call.name):
                return self._deny(base, "SCHEMA_INVALID", recoverable=True,
                                  suggested_next="S2",
                                  details={"problem": f"unknown action {call.name!r}"})
            spec = self.registry.get(call.name)
            missing = [a for a in spec.required_args if a not in (call.args or {})]
            if missing:
                return self._deny(base, "SCHEMA_INVALID", recoverable=True,
                                  suggested_next="S2",
                                  details={"problem": f"missing args {missing}"})
            if call.timeout_ms > spec.timeout_ms_max:
                return self._deny(base, "SCHEMA_INVALID", recoverable=True,
                                  suggested_next="S2",
                                  details={"problem": "timeout above registry max"})
        for oid in self._referenced_object_ids(call):
            if world.object_by_id(oid) is None:
                return self._deny(base, "SCHEMA_INVALID", recoverable=True,
                                  suggested_next="S1",
                                  details={"problem": f"object {oid!r} not in world"})
        if call.name == "wait":
            if not isinstance(call.args.get("seconds"), (int, float)) or \
                    call.args["seconds"] > self.max_wait_s:
                return self._deny(base, "SCHEMA_INVALID", recoverable=True,
                                  suggested_next="S2",
                                  details={"problem": "wait seconds invalid/over cap"})

        # 2. budget: charge-before-consume; wall clock checked for everything
        item = {CallKind.ACTION.value: "physical_actions",
                CallKind.QUERY.value: "queries",
                CallKind.CONTROL.value: "llm_calls"}[call.kind]
        if call.name == "replan":
            item = "replans"
        if item in ("physical_actions", "queries", "replans") and \
                budget.remaining_count(item) <= 0:
            return self._deny(base, "BUDGET_EXHAUSTED", recoverable=False,
                              details={"budget_item": item})
        if not budget.can_charge("wall_time_s", 1.0):
            return self._deny(base, "BUDGET_EXHAUSTED", recoverable=False,
                              details={"budget_item": "wall_time_s"})

        # 3. freshness — actions only (patch P3)
        if call.kind == CallKind.ACTION.value:
            stale = self._stale_dependencies(call, world, memory)
            if stale:
                return self._deny(base, "PERCEPTION_STALE", recoverable=True,
                                  suggested_next="S1",
                                  details={"stale_dependencies": stale})

        # 4. preconditions — actions only
        if call.kind == CallKind.ACTION.value:
            pre = self.registry.precondition(call.name)
            if pre is not None:
                ast = self._bind_args(pre, call.args)
                try:
                    _, overall = dsl.evaluate(ast, world, world)
                except dsl.DSLError as exc:
                    return self._deny(base, "SCHEMA_INVALID", recoverable=True,
                                      suggested_next="S2",
                                      details={"problem": f"precondition DSL: {exc}"})
                if overall is not True:
                    return self._deny(base, "PRECONDITION_UNMET", recoverable=True,
                                      suggested_next="S1",
                                      details={"precondition": dsl.summarize(ast)})

        # 5. safety (kinematic veto; not an e-stop)
        ok, code = self.safety.check_call(call, world)
        if not ok:
            return self._deny(base, code or "COLLISION_RISK", recoverable=False,
                              details={"layer": "safety"})

        base.allowed = True
        return base

    # ------------------------------------------------------------------
    def _deny(self, base: ValidationResult, code: str, recoverable: bool,
              suggested_next: str | None = None,
              details: dict | None = None) -> ValidationResult:
        base.allowed = False
        base.reason_code = code
        base.recoverable = recoverable
        base.suggested_next = suggested_next
        base.details = details or {}
        return base

    def _referenced_object_ids(self, call: ToolCall) -> list[str]:
        keys = ("object_id", "container_id", "socket_id", "a", "b", "target")
        out = []
        args = call.args or {}
        for k in keys:
            v = args.get(k)
            if isinstance(v, str) and v not in ("ee",):
                out.append(v)
        return out

    def _stale_dependencies(self, call: ToolCall, world: WorldState,
                            memory: Memory) -> list[str]:
        deps = []
        for oid in self._referenced_object_ids(call):
            deps.append(f"pose:{oid}")
        deps.append("gripper")
        # the world version the call was based on must not lag the current one
        lagged = call.world_version < world.version
        stale = []
        for key in deps:
            if lagged or not memory.fact_valid(key, world.version, self.ttl_versions):
                stale.append(key)
        return stale

    def _bind_args(self, node: dict, args: dict) -> dict:
        import harness.spec.registry as reg
        return reg._bind(node, args)
