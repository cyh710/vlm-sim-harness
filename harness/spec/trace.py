"""JSONL event trace (AGENTS section 4).

Every event carries the mandatory fields; ts/latency are recorded but excluded
from the deterministic comparison view so same-seed runs are byte-identical
in decisions while wall-clock fields stay honest.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

MANDATORY = ("ts", "run_id", "episode_id", "state", "call_id", "attempt_id",
             "kind", "name", "event")

VOLATILE_KEYS = {"ts", "latency_s", "wall_time_s"}


class Trace:
    def __init__(self, run_id: str, episode_id: str, file_path: Optional[Path] = None):
        self.run_id = run_id
        self.episode_id = episode_id
        self._file_path = file_path
        self.events: list[dict] = []

    def log(self, state: str, event: str, *, call_id: str = "-", attempt_id: str = "-",
            kind: str = "-", name: str = "-", verdict: Optional[str] = None,
            reason_code: Optional[str] = None, evidence_refs: tuple = (),
            budget: Optional[dict] = None, latency_s: float = 0.0, **extra) -> None:
        rec = {
            "ts": round(time.time(), 2),
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "state": state,
            "call_id": call_id,
            "attempt_id": attempt_id,
            "kind": kind,
            "name": name,
            "event": event,
            "verdict": verdict,
            "reason_code": reason_code,
            "evidence_refs": list(evidence_refs),
            "budget": budget or {},
            "latency_s": round(latency_s, 3),
        }
        rec.update(extra)
        missing = [k for k in MANDATORY if k not in rec]
        if missing:
            raise ValueError(f"trace event missing mandatory fields: {missing}")
        self.events.append(rec)
        if self._file_path is not None:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    # -- determinism view -------------------------------------------------------
    def canonical(self) -> str:
        """JSON of all events with volatile fields removed; two runs of the same
        seed MUST produce identical canonical traces.

        budget.wall_time_s is real-clock residue and volatile by the same rule
        as ts/latency_s (AGENTS §4: wall_time feeds budgets/reports only)."""
        def strip(ev: dict) -> dict:
            out = {k: v for k, v in ev.items() if k not in VOLATILE_KEYS}
            budget = out.get("budget")
            if isinstance(budget, dict):
                out["budget"] = {k: v for k, v in budget.items()
                                 if k not in VOLATILE_KEYS}
            return out
        return json.dumps([strip(e) for e in self.events],
                          ensure_ascii=False, sort_keys=True, default=str)

    def count(self, event: Optional[str] = None, **filters) -> int:
        n = 0
        for ev in self.events:
            if event is not None and ev.get("event") != event:
                continue
            if all(ev.get(k) == v for k, v in filters.items()):
                n += 1
        return n
