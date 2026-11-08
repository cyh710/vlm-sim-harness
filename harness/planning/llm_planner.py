"""LLMPlanner (M4): §9.1-constrained planner backend.

Trust boundary (design §9.1, patch P4):
- The model proposes, it never disposes. call_id/attempt_id are assigned by
  the Runtime; the model's expected_effects are DROPPED — S5 verifies via
  Registry.expand_effects, whose templates the model cannot see or alter.
- world_version is stamped from the world the planner actually observed;
  the model's echo is ignored (it can neither fake freshness nor staleness).
- Protocol failures — empty output, multiple JSON objects, invalid JSON,
  unknown skill/query/control name, non-dict args, missing required args —
  get ONE repair round inside the same llm_calls slot; a second failure
  raises PlannerProtocolError and the Runtime routes it to S7/R3.
- Accounting: one decide()/replan()/recover_local() == one llm_calls slot
  (charged by the Runtime around each call; the repair round is inside it).

stats records the M4 metrics: json 合法率 / 协议错误率 / 调用数 / 错误归因.
(被拒动作数 is Runtime-side: count S3 rejects in the trace.)
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

from harness.spec import planner_prompt as pp
from harness.spec.runtime import PlannerContext
from harness.spec.types import PlannerProtocolError, ToolCall
from harness.spec.validator import QUERY_NAMES

# LLMPlanner protocol failure taxonomy (M4 record: 错误归因分布)
E_EMPTY = "EMPTY_OUTPUT"
E_JSON = "INVALID_JSON"
E_MULTI = "MULTIPLE_OBJECTS"
E_NOT_DICT = "NOT_AN_OBJECT"
E_KIND = "BAD_KIND"
E_NAME = "UNKNOWN_NAME"
E_ARGS = "BAD_ARGS"

RepairFn = Callable[[str, str], str]      # (system_prompt, user_text) -> text


def _extract_json_object(text: str):
    """Return (obj, None) or (None, error_code). Tolerates one fenced code
    block; refuses multiple top-level objects (E_MULTI)."""
    if text is None or not text.strip():
        return None, E_EMPTY
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    # strip leading narration before the first '{'
    start = stripped.find("{")
    if start < 0:
        return None, E_JSON
    depth = 0
    end = -1
    in_str = False
    esc = False
    for i, ch in enumerate(stripped[start:], start):
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
    if end < 0:
        return None, E_JSON
    tail = stripped[end + 1:].strip().lstrip(",").strip()
    if tail:
        return None, E_MULTI
    try:
        obj = json.loads(stripped[start:end + 1])
    except json.JSONDecodeError:
        return None, E_JSON
    if not isinstance(obj, dict):
        return None, E_NOT_DICT
    return obj, None


class LLMPlanner:
    """drop-in Planner backend; `client` is injectable for tests."""

    def __init__(self, client: RepairFn, registry, task_meta: dict,
                 stats: Optional[dict] = None):
        self._client = client
        self._registry = registry
        self._task_meta = dict(task_meta)
        self.stats = stats if stats is not None else {}
        self.stats.setdefault("llm_calls", 0)
        self.stats.setdefault("json_valid_first_try", 0)
        self.stats.setdefault("repairs_used", 0)
        self.stats.setdefault("protocol_errors", 0)
        self.stats.setdefault("error_attribution", {})
        self.stats.setdefault("actions_proposed", 0)
        self.stats.setdefault("queries_proposed", 0)
        self.stats.setdefault("controls_proposed", 0)

    # -- accounting ------------------------------------------------------------
    def _note_error(self, code: str) -> None:
        self.stats["error_attribution"][code] = \
            self.stats["error_attribution"].get(code, 0) + 1

    def _ask(self, user: str) -> str:
        self.stats["llm_calls"] += 1
        return self._client(pp.SYSTEM_PROMPT, user)

    def _ask_valid_call(self, ctx: PlannerContext, user: str) -> ToolCall:
        """One ask + at most one repair round. BOTH parse errors AND semantic
        validation errors (design §9.1 lists unknown skill alongside invalid
        JSON) are repairable; a second failure raises PlannerProtocolError."""
        obj, err = _extract_json_object(self._ask(user))
        if err is None:
            call, verr = self._validate(obj, ctx)
            if verr is None:
                self.stats["json_valid_first_try"] += 1
                return call
            err = verr
        # repair round (same llm_calls slot)
        self.stats["repairs_used"] += 1
        out2 = self._ask(user + "\n\n" + pp.REPAIR_HINT.replace("{err}", err))
        obj2, err2 = _extract_json_object(out2)
        if err2 is None:
            call2, verr2 = self._validate(obj2, ctx)
            if verr2 is None:
                return call2
            err2 = verr2
        self.stats["protocol_errors"] += 1
        self._note_error(str(err2).split(":", 1)[0])
        raise PlannerProtocolError(str(err2)[:200])

    # -- shared validation (never raises; returns (ToolCall, error|None)) --------
    def _validate(self, obj: dict, ctx: PlannerContext):
        kind = obj.get("kind")
        if kind not in ("query", "action", "control"):
            return None, f"{E_KIND}: {kind!r}"
        name = obj.get("name")
        args = obj.get("args", {})
        spec_timeout = _max_timeout(self._registry, name)
        timeout = obj.get("timeout_ms", spec_timeout)
        if not isinstance(args, dict):
            return None, f"{E_ARGS}: args must be an object"
        if kind == "action":
            if not self._registry.has(name):
                return None, f"{E_NAME}: skill {name!r}"
            spec = self._registry.get(name)
            missing = [a for a in spec.required_args if a not in args]
            if missing:
                return None, (f"{E_ARGS}: {name} missing required "
                              f"args {missing}")
            # the trusted effect template must BIND with these args — if not,
            # this is a repairable BAD_ARGS here, not an S5 crash later
            try:
                self._registry.expand_effects(name, args)
            except KeyError as exc:
                return None, (f"{E_ARGS}: {name} effect template "
                              f"unbound ({exc})")
            timeout = min(int(timeout or spec.timeout_ms_max),
                          spec.timeout_ms_max)
            self.stats["actions_proposed"] += 1
        elif kind == "query":
            if name not in QUERY_NAMES:
                return None, f"{E_NAME}: query {name!r}"
            self.stats["queries_proposed"] += 1
        else:
            if name not in pp.CONTROL_NAMES:
                return None, f"{E_NAME}: control {name!r}"
            self.stats["controls_proposed"] += 1
        # trust boundary: runtime assigns call_id; effects come from the
        # Registry, not the model; version = what this planner actually saw
        return ToolCall(call_id="", kind=kind, name=name, args=dict(args),
                        world_version=ctx.world.version,
                        expected_effects=(),
                        timeout_ms=int(timeout)), None

    # -- Planner API ---------------------------------------------------------------
    def decide(self, ctx: PlannerContext) -> ToolCall:
        user = pp.payload_json(ctx, self._registry, self._task_meta)
        return self._ask_valid_call(ctx, user)

    def replan(self, ctx: PlannerContext) -> Optional[ToolCall]:
        """§9.2: task + world + verified_progress + last_reason + attempts +
        budgets in, ONE next call out (M4 pragmatic reading: the ReplanDecision
        fields ride along in the payload; the call itself still goes S2→S3)."""
        user = pp.payload_json(ctx, self._registry, self._task_meta)
        user += ("\n\nREPLAN: 上一次尝试已失败。基于失败原因与已证实进度,"
                 "给出下一个调用;若无可行恢复,输出 {\"kind\":\"control\","
                 "\"name\":\"abort\"}。")
        return self._ask_valid_call(ctx, user)

    def recover_local(self, ctx: PlannerContext) -> Optional[ToolCall]:
        """R1: adjusted retry of the failed skill; None -> escalate to R2."""
        failed = ctx.last_event.get("failed_call")
        user = pp.payload_json(ctx, self._registry, self._task_meta)
        if failed is not None:
            user += (f"\n\nRECOVERY(R1): 刚才 {failed.name} 失败"
                     f"(reason={ctx.last_event.get('reason_code')})。"
                     "给出同一技能的修正重试(可微调参数);若认为应放弃本地修复,"
                     "输出 {\"kind\":\"control\",\"name\":\"replan\"}。")
        return self._ask_valid_call(ctx, user)


def _max_timeout(registry, name) -> int:
    try:
        return registry.get(name).timeout_ms_max
    except Exception:
        return 5000
