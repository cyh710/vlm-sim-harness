"""Six-item budget with charge-before-consume semantics.

Sanity-check requirement: a zeroed budget must reject BEFORE the first action
that would consume it, never after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExhausted(Exception):
    def __init__(self, item: str):
        self.item = item
        super().__init__(f"budget exhausted: {item}")


BUDGET_ITEMS = ("physical_actions", "queries", "llm_calls", "replans",
                "recoveries", "wall_time_s")


@dataclass
class Budget:
    max_physical_actions: int = 20
    max_queries: int = 40
    max_llm_calls: int = 30
    max_replans: int = 5
    max_recoveries: int = 8
    max_wall_time_s: float = 600.0
    used: dict = field(default_factory=lambda: {
        "physical_actions": 0, "queries": 0, "llm_calls": 0,
        "replans": 0, "recoveries": 0, "wall_time_s": 0.0})

    # -- introspection -------------------------------------------------------
    def remaining(self) -> dict:
        out = {}
        for item in BUDGET_ITEMS:
            cap = getattr(self, f"max_{item}")
            out[item] = (None if cap is None
                         else round(max(0.0, cap - self.used[item]), 3))
        return out

    def remaining_count(self, item: str) -> int:
        """Remaining integer slots; wall_time uses 1 slot per remaining second."""
        cap = getattr(self, f"max_{item}")
        left = cap - self.used[item]
        return int(left) if item == "wall_time_s" else max(0, int(left))

    def exhausted_items(self) -> list[str]:
        return [i for i in BUDGET_ITEMS if self.remaining_count(i) <= 0]

    # -- charging -------------------------------------------------------------
    def can_charge(self, item: str, amount: float = 1.0) -> bool:
        cap = getattr(self, f"max_{item}")
        return cap is None or (self.used[item] + amount) <= cap + 1e-9

    def check_available(self, item: str, amount: float = 1.0) -> None:
        """Raise BudgetExhausted BEFORE any consuming step if slots are gone."""
        if not self.can_charge(item, amount):
            raise BudgetExhausted(item)

    def charge(self, item: str, amount: float = 1.0) -> None:
        if not self.can_charge(item, amount):
            raise BudgetExhausted(item)
        self.used[item] = self.used[item] + amount

    def add_wall_time(self, seconds: float) -> None:
        self.used["wall_time_s"] += max(0.0, seconds)
