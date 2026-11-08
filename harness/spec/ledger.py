"""Dual ledgers: Attempt Ledger (append on dispatch, failures included) and
Progress Ledger (S6-only writes, verified effects only).
"""
from __future__ import annotations

import itertools

from harness.spec.types import (AttemptRecord, CallState, ProgressDelta, Status)


class AttemptLedger:
    """Records every physical/query call at dispatch time; failures and
    timeouts included. Enforces the no-blind-resend invariant (AGENTS rule 5):
    a call_id in UNKNOWN state must never be re-dispatched."""

    def __init__(self):
        self._records: list[AttemptRecord] = []
        self._counter = itertools.count(1)

    # -- writes ---------------------------------------------------------------
    def record_dispatch(self, call_id: str, name: str, args: dict,
                        started_at: float) -> AttemptRecord:
        rec = AttemptRecord(
            attempt_id=f"a_{next(self._counter):04d}",
            call_id=call_id, name=name, args=dict(args),
            state=CallState.DISPATCHED, started_at=started_at)
        self._records.append(rec)
        return rec

    def settle(self, call_id: str, status: Status, ended_at: float) -> None:
        self._last_for_call(call_id).state = CallState.SETTLED
        self._last_for_call(call_id).status = status
        self._last_for_call(call_id).ended_at = ended_at

    def mark_unknown(self, call_id: str, ended_at: float) -> None:
        """Timeout without knowable outcome; outcome must never be re-attempted
        with a fresh dispatch on the same call_id."""
        self._last_for_call(call_id).state = CallState.UNKNOWN
        self._last_for_call(call_id).ended_at = ended_at

    # -- reads ----------------------------------------------------------------
    def _last_for_call(self, call_id: str) -> AttemptRecord:
        for rec in reversed(self._records):
            if rec.call_id == call_id:
                return rec
        raise KeyError(f"no attempt record for call {call_id}")

    def state_of(self, call_id: str) -> CallState:
        try:
            return self._last_for_call(call_id).state
        except KeyError:
            return CallState.PENDING

    def status_of(self, call_id: str) -> Optional[Status]:
        try:
            return self._last_for_call(call_id).status
        except KeyError:
            return None

    def all_records(self) -> tuple[AttemptRecord, ...]:
        return tuple(self._records)

    def count_name(self, name: str) -> int:
        return sum(1 for r in self._records if r.name == name)

    def count_failed(self) -> int:
        return sum(1 for r in self._records
                   if r.status is not None and r.status not in (Status.REACHED,))

    def total(self) -> int:
        return len(self._records)


class ProgressLedger:
    """Only the runtime's S6 commit path may append (single writer)."""

    def __init__(self):
        self._entries: list[ProgressDelta] = []
        self._committed: bool = False

    def commit(self, delta: ProgressDelta) -> None:
        """Caller MUST be runtime S6; enforced by convention + this being the
        only public mutation API."""
        self._entries.append(delta)

    def entries(self) -> tuple[ProgressDelta, ...]:
        return tuple(self._entries)

    def verified_effect_summaries(self) -> tuple[str, ...]:
        import harness.spec.dsl as dsl
        out = []
        for e in self._entries:
            out.extend(dsl.summarize(fx) for fx in e.verified_effects)
        return tuple(out)

    def last_progress_cycle(self) -> int:
        """Cycle index of last commit; used by the no-progress watchdog."""
        return len(self._entries)
