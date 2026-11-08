"""Goal/effect DSL: whitelist AST, no eval/exec, fail-closed on unknown ops.

Grammar (design doc section 14 + patch P1/P2):
    node  := {"op": <composite>, "children": [node, ...]}
           | {"op": <predicate>, "args": {...}}
    composite := all_of | any_of           (nesting <= 3, total predicates <= 16)

Predicates (P0 unless noted): holding, contained, gripper_open, reachable,
displacement_min (P1 per patch P2 -> P0), distance_lt, within_pose,
visible_ratio_min, gripper_in_range, joint_delta_min (P1), inserted (P2),
disp_projection_min.

Evaluation contract: every leaf returns a CheckResult with
passed in {True, False, None}; None means evidence insufficient (INCONCLUSIVE
direction). counter_evidence=True marks explicit physical counter-evidence
(not mere absence), which drives FAILED instead of INCONCLUSIVE.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from harness.spec.types import CheckResult, ObjInstance, WorldState

MAX_NESTING = 3
MAX_PREDICATES = 16

COMPOSITES = {"all_of", "any_of"}

# 夹爪开度分带的默认值 — 唯一真源是 configs/gripper_bands.json,
# 这里只是缺配置时的兜底 (AGENTS rule 2: 阈值可追溯, 不散落各处)。
_GRIPPER_BANDS_FALLBACK = {
    "hold_min": 0.25, "hold_max": 0.95,
    "empty_high": 0.95, "empty_low": 0.05,
}


@lru_cache(maxsize=1)
def gripper_bands() -> dict:
    """夹爪开度分带 (holding 谓词的本体信号)。

    判定与 VLM prompt 必须共用同一份数值: 若 VLM 被告知 "0.3~0.9 算夹住"
    而 verifier 用 "0.25~0.95" 判, 两者会系统性分歧, 且分歧只出现在边界
    开度上 — 那正是最容易滑脱的工况。
    """
    p = Path(__file__).resolve().parents[2] / "configs" / "gripper_bands.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(_GRIPPER_BANDS_FALLBACK)
    return {k: data.get(k, v) for k, v in _GRIPPER_BANDS_FALLBACK.items()}


def _band_str(bands: dict) -> str:
    """开度分带的人类可读形式 — 写进 CheckResult.threshold, 让日志里的
    判据与 configs/gripper_bands.json 永远一致。"""
    return f"{bands['hold_min']}~{bands['hold_max']}"


# op -> (required arg keys, types summary for error messages)
PREDICATE_ARGS = {
    "holding": {"object_id"},
    "contained": {"object_id", "container_id"},
    "gripper_open": set(),
    "reachable": {"object_id"},
    "displacement_min": {"object_id", "distance_m"},
    "distance_lt": {"a", "b", "d"},          # a/b: "ee" or object_id
    "within_pose": {"object_id", "target", "tol_xy", "tol_z"},
    "visible_ratio_min": {"object_id", "r"},
    "gripper_in_range": {"lo", "hi"},
    "joint_delta_min": {"joint_id", "delta"},
    "inserted": {"object_id", "socket_id", "depth_m"},
    "disp_projection_min": {"object_id", "dir", "d", "lateral"},
}


class DSLError(ValueError):
    """Raised on any structure/whitelist violation; fail-closed."""


def _check_node(node: Any, depth: int = 1) -> int:
    """Validate AST; return total leaf predicate count."""
    if not isinstance(node, dict) or "op" not in node:
        raise DSLError(f"node must be a dict with 'op', got {node!r}")
    op = node["op"]
    if depth > MAX_NESTING:
        raise DSLError(f"nesting deeper than {MAX_NESTING}")
    if op in COMPOSITES:
        children = node.get("children")
        if not isinstance(children, list) or not children:
            raise DSLError(f"{op} requires non-empty children list")
        count = 0
        for child in children:
            count += _check_node(child, depth + 1)
        return count
    if op not in PREDICATE_ARGS:
        raise DSLError(f"unknown op {op!r}; whitelist: {sorted(PREDICATE_ARGS)}")
    args = node.get("args")
    if not isinstance(args, dict):
        raise DSLError(f"{op} requires args dict")
    missing = PREDICATE_ARGS[op] - set(args.keys())
    if missing:
        raise DSLError(f"{op} missing args: {sorted(missing)}")
    return 1


def parse(node: Any) -> dict:
    """Validate and return the AST unchanged (fail-closed on any violation)."""
    total = _check_node(node)
    if total > MAX_PREDICATES:
        raise DSLError(f"total predicates {total} > {MAX_PREDICATES}")
    return node


def count_predicates(node: dict) -> int:
    return _check_node(node)


def leaf_nodes(node: dict) -> list[dict]:
    """Flatten an AST to its leaf predicate nodes (commit granularity)."""
    parse(node)
    op = node["op"]
    if op in COMPOSITES:
        out = []
        for child in node["children"]:
            out.extend(leaf_nodes(child))
        return out
    return [node]


def summarize(node: dict) -> str:
    """Human-readable one-line summary for logs and CheckResult.effect."""
    op = node["op"]
    if op in COMPOSITES:
        joiner = " AND " if op == "all_of" else " OR "
        return "(" + joiner.join(summarize(c) for c in node["children"]) + ")"
    args = node.get("args", {})
    inner = ", ".join(f"{k}={args[k]}" for k in sorted(args))
    return f"{op}({inner})"


# ---------------------------------------------------------------- evaluation

def _lerp3(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def _obj(world: WorldState, object_id: str) -> Optional[ObjInstance]:
    return world.object_by_id(object_id)


def _visible(obj: Optional[ObjInstance]) -> bool:
    return obj is not None and obj.visible_ratio > 0.0


def _eval_leaf(node: dict, before: WorldState, after: WorldState) -> CheckResult:
    op = node["op"]
    args = node["args"]

    def unknown(effect: str, signal: str = "vision") -> CheckResult:
        return CheckResult(effect=effect, passed=None, value=None, threshold=None,
                           signal=signal)

    if op == "gripper_open":
        v = after.gripper_open
        return CheckResult(effect="gripper_open", passed=bool(v >= 0.9), value=v,
                           threshold=0.9, signal="proprio")
    if op == "gripper_in_range":
        v = after.gripper_open
        lo, hi = float(args["lo"]), float(args["hi"])
        return CheckResult(effect=f"gripper_in_range({lo},{hi})",
                           passed=bool(lo <= v <= hi), value=v, threshold=(lo, hi),
                           signal="proprio")
    if op == "joint_delta_min":
        jid, delta = str(args["joint_id"]), float(args["delta"])
        b, a = before.joints.get(jid), after.joints.get(jid)
        if b is None or a is None:
            return unknown(f"joint_delta_min({jid},{delta})", "proprio")
        value = a - b
        return CheckResult(effect=f"joint_delta_min({jid},{delta})",
                           passed=bool(abs(value) >= delta), value=value,
                           threshold=delta, signal="proprio")
    if op == "visible_ratio_min":
        obj = _obj(after, str(args["object_id"]))
        r = float(args["r"])
        if obj is None:
            return CheckResult(effect=f"visible_ratio_min({args['object_id']},{r})",
                               passed=False, value=0.0, threshold=r, signal="vision",
                               counter_evidence=True, reason_code="OBJECT_NOT_VISIBLE")
        return CheckResult(effect=f"visible_ratio_min({args['object_id']},{r})",
                           passed=bool(obj.visible_ratio >= r), value=obj.visible_ratio,
                           threshold=r, signal="vision",
                           reason_code=None if obj.visible_ratio >= r else "OBJECT_OCCLUDED")

    # --- predicates that need the object in both frames -----------------------
    if op in ("holding", "displacement_min", "distance_lt", "within_pose",
              "contained", "disp_projection_min", "inserted", "reachable"):
        target_id = str(args.get("object_id") or args.get("a") or "")
        obj_after = _obj(after, target_id) if target_id else None

        if op == "holding":
            obj_id = str(args["object_id"])
            obj_a = _obj(after, obj_id)
            # 夹爪开度信号 (2026-11-06): sim410 是真实物理闭合 — 夹住
            # 0.25~0.95 / 空爪 ~1.0 / 闭合到底 ~0.0 完全可区分 (RLBench
            # kinematic attach 的"开度不可判"标定不适用于本后端)。
            # 数值来自 configs/gripper_bands.json (AGENTS rule 2), 与 VLM
            # prompt 共用 — VLM 被告知的判据必须就是它被判定用的判据。
            g = float(after.gripper_open or 1.0)
            bands = gripper_bands()
            proprio_hold = bands["hold_min"] <= g <= bands["hold_max"]
            proprio_empty = (g >= bands["empty_high"]
                             or g <= bands["empty_low"])
            if not _visible(obj_a):
                # 物体不可见 (被夹爪遮挡 = 持有态的常态): 开度主导判定
                if proprio_hold:
                    return CheckResult(effect=f"holding({obj_id})",
                                       passed=True, value=g, threshold=_band_str(bands),
                                       signal="proprio", reason_code=None)
                if proprio_empty:
                    return CheckResult(effect=f"holding({obj_id})",
                                       passed=False, value=g, threshold=_band_str(bands),
                                       signal="proprio", counter_evidence=True,
                                       reason_code="NOT_HOLDING")
                return CheckResult(effect=f"holding({obj_id})", passed=None,
                                   value=g, threshold=None, signal="proprio",
                                   reason_code="OBJECT_NOT_VISIBLE")
            d_ee = obj_a.pose.distance_to(after.ee_pose)
            HOLD_DIST = 0.12
            # 推断物 (track/ee_anchor) 贴 EE 是锚定的必然结果, 无信息量 —
            # 开度主导 (2026-11-06 实测: 空夹 0.014 被锚定视觉误判 success,
            # VLM 用开度判 S0 才是对的)
            if getattr(obj_a, "source", "") in ("track", "ee_anchor"):
                if proprio_hold:
                    return CheckResult(effect=f"holding({obj_id})",
                                       passed=True, value=g,
                                       threshold=_band_str(bands), signal="proprio",
                                       reason_code=None)
                if proprio_empty:
                    return CheckResult(effect=f"holding({obj_id})",
                                       passed=False, value=g,
                                       threshold=_band_str(bands), signal="proprio",
                                       counter_evidence=True,
                                       reason_code="NOT_HOLDING")
                return CheckResult(effect=f"holding({obj_id})", passed=None,
                                   value=g, threshold=None, signal="proprio",
                                   reason_code=None)
            vision_pass = d_ee <= HOLD_DIST
            if vision_pass:
                return CheckResult(effect=f"holding({obj_id})", passed=True,
                                   value=d_ee, threshold=HOLD_DIST,
                                   signal="vision", reason_code=None)
            if proprio_empty:
                return CheckResult(effect=f"holding({obj_id})", passed=False,
                                   value=d_ee, threshold=HOLD_DIST, signal="vision",
                                   counter_evidence=True, reason_code="NOT_HOLDING")
            return CheckResult(effect=f"holding({obj_id})", passed=None,
                               value=d_ee, threshold=HOLD_DIST, signal="vision",
                               reason_code=None)

        if op == "displacement_min":
            obj_id, d_min = str(args["object_id"]), float(args["distance_m"])
            obj_b, obj_a = _obj(before, obj_id), _obj(after, obj_id)
            if not (_visible(obj_b) and _visible(obj_a)):
                return unknown(f"displacement_min({obj_id},{d_min})")
            disp = obj_b.pose.distance_to(obj_a.pose)
            return CheckResult(
                effect=f"displacement_min({obj_id},{d_min})",
                passed=bool(disp >= d_min), value=disp, threshold=d_min,
                signal="vision",
                counter_evidence=bool(disp < d_min * 0.5),
                reason_code=None if disp >= d_min else "LIFT_NOT_ENOUGH")

        if op == "distance_lt":
            a_id, b_id, d = str(args["a"]), str(args["b"]), float(args["d"])
            pos_a = after.ee_pose if a_id == "ee" else (_obj(after, a_id).pose if _obj(after, a_id) else None)
            pos_b = after.ee_pose if b_id == "ee" else (_obj(after, b_id).pose if _obj(after, b_id) else None)
            if pos_a is None or pos_b is None:
                return unknown(f"distance_lt({a_id},{b_id},{d})")
            value = pos_a.distance_to(pos_b)
            return CheckResult(effect=f"distance_lt({a_id},{b_id},{d})",
                               passed=bool(value < d), value=value, threshold=d,
                               signal="vision")

        if op == "within_pose":
            obj_id = str(args["object_id"])
            target = args["target"]     # {"x","y","z"}
            tol_xy, tol_z = float(args["tol_xy"]), float(args["tol_z"])
            obj_a = _obj(after, obj_id)
            if not _visible(obj_a):
                return unknown(f"within_pose({obj_id})")
            dx = abs(obj_a.pose.x - float(target["x"]))
            dy = abs(obj_a.pose.y - float(target["y"]))
            dz = abs(obj_a.pose.z - float(target["z"]))
            passed = bool(dx <= tol_xy and dy <= tol_xy and dz <= tol_z)
            return CheckResult(effect=f"within_pose({obj_id})", passed=passed,
                               value=(round(dx, 4), round(dy, 4), round(dz, 4)),
                               threshold=(tol_xy, tol_z), signal="vision",
                               counter_evidence=not passed,
                               reason_code=None if passed else "OUT_OF_TOLERANCE")

        if op == "contained":
            obj_id, container_id = str(args["object_id"]), str(args["container_id"])
            obj_a = _obj(after, obj_id)
            cont_a = _obj(after, container_id)
            if not (_visible(obj_a) and _visible(cont_a)):
                return unknown(f"contained({obj_id},{container_id})")
            # proxy for sensor-region semantics: xy within container half-extent,
            # z above container base, and gripper released
            s = cont_a.size
            inside_xy = (abs(obj_a.pose.x - cont_a.pose.x) <= s[0] / 2 and
                         abs(obj_a.pose.y - cont_a.pose.y) <= s[1] / 2)
            inside_z = obj_a.pose.z - cont_a.pose.z <= s[2] + 0.02
            released = after.gripper_open >= 0.9
            passed = bool(inside_xy and inside_z and released)
            if passed:
                return CheckResult(effect=f"contained({obj_id},{container_id})",
                                   passed=True, value=True, threshold=True,
                                   signal="derived")
            counter = inside_xy and inside_z and not released and \
                obj_a.pose.distance_to(after.ee_pose) > 0.10
            return CheckResult(effect=f"contained({obj_id},{container_id})",
                               passed=False, value=(inside_xy, inside_z, released),
                               threshold=True, signal="derived",
                               counter_evidence=bool(counter),
                               reason_code=None if counter else "OUT_OF_TOLERANCE")

        if op == "inserted":
            obj_id, depth_m = str(args["object_id"]), float(args["depth_m"])
            obj_a = _obj(after, obj_id)
            if not _visible(obj_a):
                return unknown(f"inserted({obj_id})")
            depth = float(args.get("_measured_depth", 0.0)) if "_measured_depth" in args \
                else max(0.0, -obj_a.pose.z)
            passed = bool(depth >= depth_m)
            return CheckResult(effect=f"inserted({obj_id},{depth_m})", passed=passed,
                               value=depth, threshold=depth_m, signal="vision",
                               counter_evidence=not passed,
                               reason_code=None if passed else "OUT_OF_TOLERANCE")

        if op == "disp_projection_min":
            obj_id, d_min = str(args["object_id"]), float(args["d"])
            lateral = float(args["lateral"])
            direction = tuple(float(v) for v in args["dir"])
            obj_b, obj_a = _obj(before, obj_id), _obj(after, obj_id)
            if not (_visible(obj_b) and _visible(obj_a)):
                return unknown(f"disp_projection_min({obj_id})")
            dx = (obj_a.pose.x - obj_b.pose.x, obj_a.pose.y - obj_b.pose.y)
            norm = (direction[0] ** 2 + direction[1] ** 2) ** 0.5 or 1.0
            proj = (dx[0] * direction[0] + dx[1] * direction[1]) / norm
            lat = abs(dx[0] * direction[1] - dx[1] * direction[0]) / norm
            passed = bool(proj >= d_min and lat <= lateral)
            return CheckResult(effect=f"disp_projection_min({obj_id})",
                               passed=passed, value=(round(proj, 4), round(lat, 4)),
                               threshold=(d_min, lateral), signal="vision",
                               counter_evidence=bool(proj < d_min * 0.5))

        if op == "reachable":
            obj_id = str(args["object_id"])
            obj_a = _obj(after, obj_id)
            if not _visible(obj_a):
                return unknown(f"reachable({obj_id})")
            passed = bool(obj_a.pose.distance_to(after.ee_pose) <= 0.6)
            return CheckResult(effect=f"reachable({obj_id})", passed=passed,
                               value=round(obj_a.pose.distance_to(after.ee_pose), 4),
                               threshold=0.6, signal="derived")

    raise DSLError(f"unhandled predicate {op}")   # pragma: no cover (whitelist guard)


def evaluate(node: dict, before: WorldState, after: WorldState,
             collect_all: bool = False) -> tuple[list[CheckResult], Optional[bool]]:
    """Evaluate AST against before/after frames.

    Returns (checks, overall) where overall in {True, False, None}.
    all_of: any explicit counter-evidence fail -> False; all pass -> True;
            otherwise None.
    any_of: any pass -> True; all counter-evidenced fails -> False; else None.
    """
    parse(node)
    op = node["op"]
    if op in COMPOSITES:
        results: list[CheckResult] = []
        child_overalls = []
        for child in node["children"]:
            sub, overall = evaluate(child, before, after, collect_all=True)
            results.extend(sub)
            child_overalls.append(overall)
        if op == "all_of":
            if any(o is False for o in child_overalls):
                overall = False
            elif all(o is True for o in child_overalls):
                overall = True
            else:
                overall = None
        else:  # any_of
            if any(o is True for o in child_overalls):
                overall = True
            elif all(o is False for o in child_overalls):
                overall = False
            else:
                overall = None
        return (results, overall) if collect_all else (results, overall)

    check = _eval_leaf(node, before, after)
    return [check], check.passed
