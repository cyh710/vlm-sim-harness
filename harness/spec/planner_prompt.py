"""Planner prompt + §9.1 input payload — SINGLE SOURCE (AGENTS §5).

SYSTEM_PROMPT is the frozen text from design doc §9.1 ("Planner Prompt 结构").
Any wording change must bump PROMPT_VERSION and re-freeze planner_prompt_hash
in configs/preregistration.json — never mid-experiment (AGENTS rule 9).

build_payload() assembles the MINIMAL §9.1 input JSON. What it must never
contain (oracle leakage, AGENTS rule 1/10): simulator success signals,
reward-shaped values, hidden object ground truth, evaluation answers. The
WorldState view here is exactly what a sensor_eval perception could produce
— object id/label/pose/quality from the runtime's world only.
"""
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:      # import cycle guard: runtime imports types only
    from harness.spec.registry import Registry
    from harness.spec.runtime import PlannerContext

PROMPT_VERSION = "m4-2"

SYSTEM_PROMPT = (
    "你是上层机器人任务规划器。只能从 ALLOWED_SKILLS 选一个调用。\n"
    "你不能声称工具已成功、不能编造不存在的 object_id、不能给低层关节/电机控制命令。\n"
    "仅输出符合 schema 的单个 JSON 对象。若 task.goal_status_now 为 true，"
    "立即输出 control:done（会独立验收）；目标达成后不得重复物理动作。\n"
    "遇到结果未知或遮挡优先获取证据；安全拒绝不可通过变更措辞规避。\n"
    "INPUT: TASK, WORLD, PROGRESS, RECENT_ATTEMPTS, BUDGET, ALLOWED_SKILLS。\n"
    "OUTPUT: 单个 {kind,name,args,world_version,expected_effects,timeout_ms} JSON。"
)

REPAIR_HINT = (
    "你的上一个输出不是合法的单个调用 JSON（错误: {err}）。"
    "重新输出恰好一个符合 OUTPUT schema 的 JSON 对象，不要任何解释文字。"
)

# output schema keys, kept in one place for parser + docs
OUTPUT_KEYS = ("kind", "name", "args", "world_version", "expected_effects",
               "timeout_ms")
CONTROL_NAMES = ("done", "abort", "replan")


def prompt_hash() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def build_payload(ctx: "PlannerContext", registry: "Registry",
                  task_meta: dict) -> dict:
    """§9.1 minimal input. task_meta = {"task_id", "instruction",
    "usage_hints"} — task-level metadata only; the goal DSL comes from the
    trusted Registry. usage_hints carry CALIBRATED skill-usage conventions
    (values sourced from configs/skill_hints.json <- calibration/
    success_semantics.csv); they say HOW to invoke a skill, never whether
    the task succeeded."""
    world = ctx.world
    objects = [{"id": o.id, "label": o.label,
                "pose_xyz": [round(o.pose.x, 4), round(o.pose.y, 4),
                             round(o.pose.z, 4)],
                "quality": o.quality, "last_seen_t": round(o.last_seen_t, 3)}
               for o in world.objects]
    skills = [{"name": n,
               "kind": registry.get(n).kind,
               "capability_card": registry.get(n).capability_card,
               "required_args": list(registry.get(n).required_args),
               "optional_args": list(registry.get(n).optional_args),
               "timeout_ms_max": registry.get(n).timeout_ms_max}
              for n in registry.action_names()]
    # goal_status_now: derived ONLY from planner-visible data (the goal DSL,
    # the verified-progress summaries, and the world view) — state predicates
    # evaluate on the current world, transition predicates check the progress
    # ledger. Same rule as the runtime's global acceptance, reformatted for
    # the planner; never reads judge/oracle signals (AGENTS rule 1/10).
    goal_status = None
    if registry.task_goal:
        goal_status = _goal_status(registry.task_goal, world,
                                   set(ctx.progress_summaries))
    return {
        "task": {"task_id": task_meta.get("task_id", ""),
                 "instruction": task_meta.get("instruction", ""),
                 "goal": registry.task_goal,
                 "goal_status_now": goal_status,
                 "usage_hints": task_meta.get("usage_hints", {})},
        "world": {"version": world.version,
                  "gripper_open_fraction": world.gripper_open,
                  "objects": objects},
        "verified_progress": list(ctx.progress_summaries),
        "recent_attempts": list(ctx.recent_attempts),
        "available_skills": skills,
        "budgets_left": {k: v for k, v in ctx.budget_remaining.items()
                         if v is not None},
        "constraints": {"single_tool_call": True,
                        "no_direct_actuation": True,
                        "expected_effects_are_untrusted": True,
                        "world_version_you_saw": world.version},
    }


def _goal_status(goal: dict, world, progress: set) -> bool:
    """Mirror of the runtime's global-acceptance rule over planner-visible
    data: composite ops combine; transition predicates are verified-history
    (progress summaries, matched on op + object_id like Runtime.
    _progress_has_effect — thresholds deliberately ignored); state predicates
    evaluate on the current world."""
    import harness.spec.dsl as dsl
    op = goal["op"]
    if op in dsl.COMPOSITES:
        vals = [_goal_status(c, world, progress) for c in goal.get("children", [])]
        return all(vals) if op == "all_of" else any(vals)
    if op in ("displacement_min", "disp_projection_min"):
        oid = goal.get("args", {}).get("object_id")
        return any(s.startswith(f"{op}(") and f"object_id={oid}" in s
                   for s in progress)
    _, overall = dsl.evaluate(goal, world, world)
    return bool(overall)


def payload_json(ctx: "PlannerContext", registry: "Registry",
                 task_meta: dict) -> str:
    return json.dumps(build_payload(ctx, registry, task_meta),
                      ensure_ascii=False, sort_keys=True)
