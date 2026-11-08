"""Judge bridge — the ONLY module in harness/ allowed to read the simulator
judge (AGENTS rule 1). Scripts call read_success(); spec/ runtime and verifier
never import this module (CI-enforced by check_oracle_isolation.py).
"""
from __future__ import annotations


def read_success(adapter) -> bool:
    """Offline judge: read task.success() AFTER the episode ended.

    The adapter hands over the task-environment handle; the success() READ
    lives here and nowhere else in harness/.
    """
    task = adapter.judge_bridge_task()
    ok, _ = task.success()
    return bool(ok)
