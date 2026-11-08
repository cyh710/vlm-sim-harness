#!/usr/bin/env python3
"""Manifest stability check (AGENTS rule 9).

The registry manifest hash and DSL whitelist hash recorded in
configs/preregistration.json must match the live harness. Any drift means
someone changed capability cards / effect templates / DSL vocabulary without
re-freezing the preregistration — results would not be comparable.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harness.spec import dsl as h_dsl                 # noqa: E402
from harness.spec.registry import default_registry    # noqa: E402


def dsl_whitelist_hash() -> str:
    blob = json.dumps(sorted(h_dsl.PREDICATE_ARGS.keys()), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _vlm_prompt_hash() -> str:
    from harness.planning.vlm_planner import prompt_hash
    return prompt_hash()


def _planner_prompt_hash() -> str:
    from harness.spec.planner_prompt import prompt_hash
    return prompt_hash()


def _semantics_hash() -> str:
    """判据标定表的指纹 — 判据是判定的唯一来源, 改了它结果就不可比。

    按**规范化后的文本**算, 不按原始字节: 换行符 (CRLF/LF) 不改变判据
    的语义, 不该让指纹漂移。按字节算的话, 同一份文件在不同平台的
    checkout 下会得到不同的哈希, 门禁就会在别人的机器上无故变红。
    """
    p = REPO / "results" / "success_semantics.csv"
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", newline="")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def main() -> int:
    pre_path = REPO / "configs" / "preregistration.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    frozen = pre.get("frozen_artifacts", {})

    registry = default_registry()
    manifest_hash = registry.manifest_hash()
    whitelist_hash = dsl_whitelist_hash()
    vlm_prompt_hash = _vlm_prompt_hash()
    planner_prompt_hash = _planner_prompt_hash()
    semantics_hash = _semantics_hash()

    problems = []
    recorded_manifest = frozen.get("skill_manifest_hash")
    if recorded_manifest not in (None, "TODO", "COMPUTED_AT_FREEZE"):
        if recorded_manifest != manifest_hash:
            problems.append(f"skill_manifest_hash drift: frozen={recorded_manifest[:12]} "
                            f"live={manifest_hash[:12]}")
    recorded_dsl = frozen.get("dsl_whitelist_hash")
    if recorded_dsl not in (None, "TODO", "COMPUTED_AT_FREEZE"):
        if recorded_dsl != whitelist_hash:
            problems.append(f"dsl_whitelist_hash drift: frozen={recorded_dsl[:12]} "
                            f"live={whitelist_hash[:12]}")
    for key, live in (("vlm_prompt_hash", vlm_prompt_hash),
                      ("planner_prompt_hash", planner_prompt_hash),
                      ("success_semantics_hash", semantics_hash)):
        rec = frozen.get(key)
        if rec not in (None, "TODO", "COMPUTED_AT_FREEZE") and rec != live:
            problems.append(f"{key} drift: frozen={rec[:12]} live={live[:12]}")

    # refresh mode: write current hashes back into the preregistration
    if "--freeze" in sys.argv:
        frozen["skill_manifest_hash"] = manifest_hash
        frozen["dsl_whitelist_hash"] = whitelist_hash
        frozen["vlm_prompt_hash"] = vlm_prompt_hash
        frozen["planner_prompt_hash"] = planner_prompt_hash
        frozen["success_semantics_hash"] = semantics_hash
        pre["frozen_artifacts"] = frozen
        pre_path.write_text(json.dumps(pre, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"[FREEZE] manifest={manifest_hash[:16]} dsl={whitelist_hash[:16]} "
              f"vlm_prompt={vlm_prompt_hash[:16]}")
        return 0

    if problems:
        print("[FAIL] preregistration drift:")
        for p in problems:
            print("  -", p)
        return 1
    print(f"[OK] manifest stable (live manifest={manifest_hash[:16]}, "
          f"dsl={whitelist_hash[:16]}, vlm_prompt={vlm_prompt_hash[:16]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
