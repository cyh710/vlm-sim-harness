#!/usr/bin/env python3
"""Oracle isolation check (AGENTS rule 1/3/10).

Rules enforced:
  - judge signals (task.success() / reward) may only be touched inside
    harness/adapters/judge.py — the single judge-bridge module.
  - the executor self-report (evidence.claimed) may only be *written* by
    adapters; never read by spec/verifier/runtime.
  - `import rlbench` is allowed ONLY under harness/adapters/ (the backend
    boundary); spec/ skills/ planning/ must stay backend-free.
Static grep; run in CI before any commit.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "harness"
JUDGE_MODULE = "adapters/judge.py"
ADAPTERS_DIR = "adapters/"

JUDGE_PATTERNS = [
    (r"task\.success\s*\(", "simulator judge read (task.success)"),
    (r"\.success\s*\(\s*\)", "bare success() call"),
    (r"get_reward|shaped_reward|\.reward\b", "reward access"),
]

CLAIMED_READ_PATTERNS = [
    (r"\.claimed\s*\[", "self-report read via index"),
    (r"\.claimed\.get|getattr\([^)]*claimed", "self-report read"),
]

RLBENCH_IMPORT = r"^(from\s+rlbench|import\s+rlbench)"


def violations() -> list[str]:
    found = []
    for py in sorted(HARNESS.rglob("*.py")):
        rel = str(py.relative_to(REPO)).replace("\\", "/").replace("harness/", "", 1)
        in_judge = rel == JUDGE_MODULE
        in_adapters = rel.startswith(ADAPTERS_DIR)
        text = py.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or \
                    stripped.startswith("'''"):
                continue
            if not in_judge:
                for pattern, why in JUDGE_PATTERNS:
                    if re.search(pattern, line):
                        found.append(f"harness/{rel}:{i}: [{why}] {stripped[:90]}")
            if not in_adapters:
                if re.search(RLBENCH_IMPORT, stripped):
                    found.append(f"harness/{rel}:{i}: [sim import outside adapters/] "
                                 f"{stripped[:90]}")
            for pattern, why in CLAIMED_READ_PATTERNS:
                if re.search(pattern, line):
                    found.append(f"harness/{rel}:{i}: [{why}] {stripped[:90]}")
    return found


def main() -> int:
    found = violations()
    if found:
        print("[FAIL] oracle isolation violated:")
        for v in found:
            print("  -", v)
        return 1
    print("[OK] oracle isolation: judge/reward confined to adapters/judge.py; "
          "rlbench imports confined to adapters/; no self-report reads")
    return 0


if __name__ == "__main__":
    sys.exit(main())
