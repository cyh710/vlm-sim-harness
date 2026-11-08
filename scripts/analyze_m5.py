#!/usr/bin/env python3
"""M5 analysis (plan §6.5/6.6): paired bootstrap on judge success.

Primary endpoint (preregistration 2.x): PutRubbishInBin + F2, C vs A,
paired by seed, diff CI 95% (percentile bootstrap, 10000 resamples, fixed
RNG seed for reproducibility). Secondaries: N0 C-vs-A, false_completion
diff, repeated-action rate under F2, cost per success.

Usage:
  python scripts/analyze_m5.py runs/m5_pilot_<stamp>/rows.json [more.json]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path


def load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        rows.extend(json.loads(Path(p).read_text(encoding="utf-8")))
    for r in rows:
        r["judge_success"] = bool(r.get("judge_success"))
        r["online_claim_success"] = bool(r.get("online_claim_success"))
        r["false_completion"] = bool(r.get("online_claim_success")
                                     and not r.get("judge_success"))
        r["repeated_grasp"] = int(r.get("grasp_dispatches") or 0) > 1
    return rows


def cell(rows: list[dict], arm: str, fault: str) -> dict:
    sel = sorted((r for r in rows if r["arm"] == arm and r["fault"] == fault),
                 key=lambda r: r["seed"])
    n = len(sel)
    if not n:
        return {"n": 0}
    succ = sum(r["judge_success"] for r in sel)
    claimed = sum(r["online_claim_success"] for r in sel)
    false_comp = sum(r["false_completion"] for r in sel)
    rep = sum(r["repeated_grasp"] for r in sel)
    succ_rows = [r for r in sel if r["judge_success"]]
    cost_act = (sum(r.get("physical_actions") or 0 for r in succ_rows)
                / max(1, len(succ_rows)))
    cost_llm = (sum(r.get("llm_calls") or 0 for r in succ_rows)
                / max(1, len(succ_rows)))
    return {"n": n, "seeds": [r["seed"] for r in sel],
            "judge_success": succ, "rate": succ / n,
            "claimed": claimed, "false_completion": false_comp,
            "false_completion_rate": false_comp / max(1, claimed),
            "repeated_grasp": rep,
            "repeated_grasp_rate": rep / n,
            "cost_per_success": {"physical_actions": round(cost_act, 2),
                                 "llm_calls": round(cost_llm, 2)}}


def paired_diff_ci(rows: list[dict], arm_x: str, arm_y: str, fault: str,
                   n_res: int = 10000, seed: int = 20261030):
    """P(arm_x) - P(arm_y) with paired-by-seed percentile bootstrap CI."""
    xs = {r["seed"]: int(r["judge_success"]) for r in rows
          if r["arm"] == arm_x and r["fault"] == fault}
    ys = {r["seed"]: int(r["judge_success"]) for r in rows
          if r["arm"] == arm_y and r["fault"] == fault}
    common = sorted(set(xs) & set(ys))
    if not common:
        return None
    diffs = [xs[s] - ys[s] for s in common]
    rng = random.Random(seed)
    boots = []
    for _ in range(n_res):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo = boots[int(0.025 * (n_res - 1))]
    hi = boots[int(0.975 * (n_res - 1))]
    obs = sum(diffs) / len(diffs)
    discordant = sum(1 for d in diffs if d != 0)
    return {"comparison": f"{arm_x} vs {arm_y}", "fault": fault,
            "n_pairs": len(common), "observed_diff": round(obs, 3),
            "ci95": [round(lo, 3), round(hi, 3)],
            "discordant_pairs": discordant,
            "x_successes": sum(xs[s] for s in common),
            "y_successes": sum(ys[s] for s in common)}


def power_note(paired: dict) -> str:
    """Normal-approx confirm-n estimate from pilot discordants (§6.5)."""
    n = paired["n_pairs"]
    d = paired["discordant_pairs"]
    if n == 0:
        return "no pairs"
    p_disc = d / n
    obs = abs(paired["observed_diff"])
    if obs == 0 or p_disc == 0:
        return ("no discordant pairs observed — effect not estimable from "
                "pilot; keep confirm n=100 per preregistration")
    # var(diff mean) ~ p_disc / n ; solve n for 80% power at alpha=.05 two-sided
    import math
    var = p_disc / n
    se = math.sqrt(var)
    if se == 0:
        return "degenerate"
    z_eff = obs / se
    n_needed = int(math.ceil(((1.96 + 0.84) / z_eff) ** 2 * n))
    return (f"pilot discordance {d}/{n}; effect {obs:+.2f}; "
            f"normal-approx n for 80% power ≈ {n_needed} "
            f"(confirm plan: 100 shared seeds)")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    rows = load_rows(sys.argv[1:])
    out = {"primary": None, "secondary": {}, "cells": {}, "notes": []}

    out["primary"] = paired_diff_ci(rows, "C", "A", "F2")
    if out["primary"]:
        out["notes"].append("POWER: " + power_note(out["primary"]))
    out["secondary"]["N0_C_vs_A"] = paired_diff_ci(rows, "C", "A", "N0")
    for fault in ("N0", "F1", "F2"):
        for arm in ("A", "B", "C1", "C"):
            out["cells"][f"{arm}@{fault}"] = cell(rows, arm, fault)
    # repeated-action rate under F2 belongs to C1/C by design (plan §6.3)
    fc = {arm: out["cells"][f"{arm}@F2"]["false_completion_rate"]
          for arm in ("A", "B", "C1", "C") if out["cells"][f"{arm}@F2"]["n"]}
    out["secondary"]["false_completion_F2"] = fc
    rg = {arm: out["cells"][f"{arm}@F2"]["repeated_grasp_rate"]
          for arm in ("A", "B", "C1", "C") if out["cells"][f"{arm}@F2"]["n"]}
    out["secondary"]["repeated_grasp_rate_F2"] = rg
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if len(sys.argv) > 1:
        dst = Path(sys.argv[1]).parent / "m5_analysis.json"
        dst.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\nanalysis -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
