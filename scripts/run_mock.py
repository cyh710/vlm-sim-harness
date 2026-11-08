#!/usr/bin/env python3
"""Run the 10-scenario mock matrix (M1 exit criterion).

    python scripts/run_mock.py [--seed 1]

Exit code 0 iff every scenario's observed outcome matches its expectation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harness.adapters.scenarios import SCENARIOS, run_scenario  # noqa: E402

EXPECTATIONS = {
    "normal": lambda r, rt: (
        r["exit_reason"] == "SUCCESS" and r["online_claim_success"] is True
        and r["grasp_attempts"] == 1),
    "timeout_after_effect": lambda r, rt: (
        r["exit_reason"] == "SUCCESS" and r["online_claim_success"] is True
        and r["grasp_attempts"] == 1),          # F2 invariant: never re-sent
    "grasp_empty_once": lambda r, rt: (
        r["exit_reason"] == "SUCCESS" and r["grasp_attempts"] == 2
        and rt.trace.count(event="recover", name="R1") >= 1),
    "early_done_once": lambda r, rt: (
        r["exit_reason"] == "SUCCESS"
        and rt.trace.count(event="global_accept", verdict="failed") == 1
        and rt.trace.count(event="global_accept", verdict="success") == 1),
    "stale_pose": lambda r, rt: (
        r["exit_reason"] == "SUCCESS"
        and rt.trace.count(event="validate", reason_code="PERCEPTION_STALE") >= 1),
    "occluded": lambda r, rt: (
        r["exit_reason"] == "SUCCESS"
        and rt.trace.count(event="validate", reason_code="PRECONDITION_UNMET") >= 1
        and rt.attempts.count_name("move_camera") == 1),
    "budget_exhausted": lambda r, rt: (
        r["exit_reason"] == "BUDGET_EXHAUSTED"
        and r["online_claim_success"] is False
        and r["progress_entries"] >= 1),        # facts preserved on exit
    "unsafe": lambda r, rt: (
        r["exit_reason"] == "SAFETY_STOP"
        and r["online_claim_success"] is False
        and rt.trace.count(event="recover") == 0),   # never routed through S7
    "partial": lambda r, rt: (
        rt.trace.count(event="verify", verdict="partial") >= 1
        and r["exit_reason"] == "FAILED"
        and all("contained" not in s for s in rt.progress.verified_effect_summaries())
        and any("gripper_open" in s for s in rt.progress.verified_effect_summaries())),
    "no_progress": lambda r, rt: (
        r["exit_reason"] == "FAILED"
        and r["online_claim_success"] is False
        and r["physical_actions"] == 0),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rows = []
    all_ok = True
    for name in SCENARIOS:
        result, rt = run_scenario(name, seed=args.seed)
        check = EXPECTATIONS[name]
        ok = bool(check(result, rt))
        all_ok &= ok
        rows.append((name, result, ok))
        print(f"[{'OK  ' if ok else 'FAIL'}] {name:22s} exit={result['exit_reason']:16s} "
              f"claim={result['online_claim_success']!s:5s} grasp={result['grasp_attempts']} "
              f"phys={result['physical_actions']} progress={result['progress_entries']} "
              f"llm={result['llm_calls']}")

    print("\n===== mock matrix =====")
    for name, result, ok in rows:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{'ALL GREEN' if all_ok else 'FAILURES PRESENT'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
