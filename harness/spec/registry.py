"""Trusted skill registry: capability cards + effect templates + manifest hash.

The Registry is the ONLY source of trusted expected effects (patch P4): the
planner's claimed expected_effects are audit-only. Templates are parameterized
DSL ASTs; expand_effects() binds planner args into concrete ASTs, which the
verifier then evaluates. Planner cannot register or alter templates at runtime
(manifest hash would break; preregistration checks enforce this).
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import harness.spec.dsl as dsl


@dataclass(frozen=True)
class SkillSpec:
    name: str
    kind: str                                   # "action"
    capability_card: str
    required_args: tuple[str, ...]
    optional_args: tuple[str, ...] = ()
    precondition_ast: dict | None = None        # DSL AST over current world
    effect_template: dict | None = None         # DSL AST with "$param" placeholders
    timeout_ms_max: int = 10000
    arg_defaults: dict = field(default_factory=dict)   # template fallbacks
    # NOTE: numeric defaults above are mock-platform placeholders until M2
    # delivers calibration/success_semantics.csv (AGENTS rule 2).


def _bind(node: dict, args: dict) -> dict:
    """Substitute $param placeholders in a template AST with concrete values."""
    out = {}
    for k, v in node.items():
        if isinstance(v, str) and v.startswith("$"):
            key = v[1:]
            if key not in args:
                raise KeyError(f"template placeholder ${key} missing from args")
            out[k] = args[key]
        elif isinstance(v, dict):
            out[k] = _bind(v, args)
        elif isinstance(v, list):
            out[k] = [_bind(c, args) for c in v]
        else:
            out[k] = v
    return out


def _grasp_visible_ratio_default() -> float:
    """grasp 可见率下限的默认值 — 唯一真源 configs/registry.json。"""
    p = Path(__file__).resolve().parents[2] / "configs" / "registry.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return float(data.get("grasp_visible_ratio_min", 0.2))
    except (OSError, ValueError, TypeError):
        return 0.2


class Registry:
    def __init__(self, skills: list[SkillSpec], task_goal: dict | None = None):
        self._skills = {s.name: s for s in skills}
        self.task_goal = task_goal          # global goal DSL AST (for done acceptance)
        dsl.parse(self.task_goal) if self.task_goal else None

    # -- lookups ---------------------------------------------------------------
    def get(self, name: str) -> SkillSpec:
        if name not in self._skills:
            raise KeyError(f"skill {name!r} not in registry")
        return self._skills[name]

    def has(self, name: str) -> bool:
        return name in self._skills

    def action_names(self) -> list[str]:
        return sorted(n for n, s in self._skills.items() if s.kind == "action")

    def capability_cards(self) -> list[str]:
        return [s.capability_card for s in self._skills.values()]

    # -- trusted effects (patch P4) ----------------------------------------------
    def expand_effects(self, name: str, args: dict) -> tuple[dict, ...]:
        spec = self.get(name)
        if spec.effect_template is None:
            return ()
        bound = {**spec.arg_defaults, **(args or {})}
        ast = _bind(copy.deepcopy(spec.effect_template), bound)
        dsl.parse(ast)
        return (ast,)

    def precondition(self, name: str) -> dict | None:
        return self.get(name).precondition_ast

    # -- manifest -----------------------------------------------------------------
    def manifest(self) -> dict:
        return {
            "skills": [
                {
                    "name": s.name, "kind": s.kind,
                    "capability_card": s.capability_card,
                    "required_args": list(s.required_args),
                    "optional_args": list(s.optional_args),
                    "precondition_ast": s.precondition_ast,
                    "effect_template": s.effect_template,
                    "timeout_ms_max": s.timeout_ms_max,
                }
                for s in sorted(self._skills.values(), key=lambda x: x.name)
            ],
            "task_goal": self.task_goal,
        }

    def manifest_hash(self) -> str:
        blob = json.dumps(self.manifest(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# M1 built-in skills for mock scenarios. grasp/place mirror the patch-P1 table:
#   grasp = holding(obj) AND displacement_min(obj, d_lift)   [PickUpCup semantics]
#   place = contained(obj, container) AND gripper_open
# Thresholds here are mock-platform calibration placeholders; real values come
# from calibration/success_semantics.csv once M2 runs (AGENTS rule 2).

def default_registry(task_goal: dict | None = None,
                     grasp_visible_ratio_min: float | None = None) -> Registry:
    """构建受信任技能表。

    grasp_visible_ratio_min: grasp 的"目标可见率下限"前置条件。默认取
    configs/registry.json 的值 (0.2) —— 看不见的目标不该盲抓, 这是 S3
    前置条件存在的意义。

    后端可以覆盖它: 物理后端里纸团会被夹爪自身遮挡, 可见率天然偏低,
    2026-11-06 实测该门槛拒杀了 6 次正常抓取, 因此该后端把它放宽到 0。
    覆盖必须显式写出来 (而不是把门槛从代码里删掉), 这样"这个后端为什么
    不看可见率"在代码里是可见的, 且 manifest hash 会记录这次差异。
    """
    if grasp_visible_ratio_min is None:
        grasp_visible_ratio_min = _grasp_visible_ratio_default()
    skills = [
        # 查询技能 (2026-11-06 补全: M1 表缺失导致 planner 的重观测 query
        # 被 unknown-action 拒杀 13+ 次, 识别自愈链路断裂)
        SkillSpec(
            name="where_is", kind="query",
            capability_card="where_is(object_id): re-observe and report the object pose.",
            required_args=("object_id",),
            effect_template=None, timeout_ms_max=30000,
        ),
        SkillSpec(
            name="observe", kind="query",
            capability_card="observe(): full perception refresh.",
            required_args=(),
            effect_template=None, timeout_ms_max=30000,
        ),
        SkillSpec(
            name="list_objects", kind="query",
            capability_card="list_objects(): list registered object ids.",
            required_args=(),
            effect_template=None, timeout_ms_max=30000,
        ),
        SkillSpec(
            name="grasp", kind="action",
            capability_card=("grasp(object_id, target_xyz): close gripper on the "
                             "object and lift it clear of its support surface."),
            required_args=("object_id", "target_xyz"),
            # 前置可见性: 看不见的目标不该盲抓 (S3 前置条件的意义)。阈值
            # 来自 configs/registry.json, 后端可显式覆盖 (见
            # default_registry 的 docstring)。
            precondition_ast=({
                "op": "visible_ratio_min",
                "args": {"object_id": "$object_id",
                         "r": grasp_visible_ratio_min},
            } if grasp_visible_ratio_min and grasp_visible_ratio_min > 0
            else None),
            # NOTE: no gripper_open precondition — a failed close leaves the
            # gripper shut and the skill itself opens first (R1 retry path).
            effect_template={
                "op": "all_of", "children": [
                    {"op": "holding", "args": {"object_id": "$object_id"}},
                    {"op": "displacement_min",
                     "args": {"object_id": "$object_id", "distance_m": "$d_lift"}},
                ]},
            arg_defaults={"d_lift": 0.03},
            timeout_ms_max=10000,
        ),
        SkillSpec(
            name="place", kind="action",
            capability_card=("place(object_id, container_id, target_xyz): release "
                             "the held object into the container."),
            required_args=("object_id", "container_id", "target_xyz"),
            precondition_ast={
                "op": "holding", "args": {"object_id": "$object_id"}},
            effect_template={
                "op": "all_of", "children": [
                    {"op": "contained",
                     "args": {"object_id": "$object_id", "container_id": "$container_id"}},
                    {"op": "gripper_open", "args": {}},
                ]},
            timeout_ms_max=10000,
        ),
        SkillSpec(
            name="push", kind="action",
            capability_card=("push(object_id, target_xyz, dir): slide object along "
                             "dir; verified by projected displacement."),
            required_args=("object_id", "target_xyz", "dir"),
            optional_args=("lateral",),
            effect_template={
                "op": "disp_projection_min",
                "args": {"object_id": "$object_id", "dir": "$dir",
                         "d": "$d", "lateral": "$lateral"}},
            # M4 fix: same unbound-placeholder family as move_camera
            arg_defaults={"d": 0.05, "lateral": 0.0},   # mock placeholder (M1 note)
            timeout_ms_max=10000,
        ),
        SkillSpec(
            name="move_camera", kind="action",
            capability_card=("move_camera(target_xyz, object_id, r): reposition "
                             "the camera to look at the named object; consumes "
                             "motion budget; verified by the object becoming "
                             "visible (r = required visible ratio)."),
            # M4 fix: object_id was previously NOT required although the effect
            # template binds $object_id — any planner proposing this skill
            # crashed S5 with an unbound-placeholder KeyError (latent M1 bug,
            # first surfaced by the LLM planner on real-backend seeds 1001-1005).
            required_args=("target_xyz", "object_id"),
            optional_args=("look_at",),
            effect_template={
                "op": "visible_ratio_min",
                "args": {"object_id": "$object_id", "r": "$r"}},
            arg_defaults={"r": 0.2},
            timeout_ms_max=10000,
        ),
    ]
    return Registry(skills, task_goal=task_goal)
