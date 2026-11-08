"""Memory: bounded summaries + facts with validity windows.

Facts carry valid_until (world version) and derived_from_call_ids so S8 can
invalidate exactly the assumptions a replan breaks, and S3 can query
freshness of the facts an action depends on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

MAX_SUMMARIES = 5
MAX_SUMMARY_CHARS = 200


@dataclass
class Fact:
    key: str                      # e.g. "pose:red_cup"
    value: dict
    world_version: int            # version at which this fact was true
    valid_until: Optional[int] = None   # exclusive bound; None = until invalidated
    derived_from_call_ids: tuple[str, ...] = ()
    source: str = "observe"       # observe | verify | planner


@dataclass
class Memory:
    summaries: list[str] = field(default_factory=list)
    facts: dict[str, Fact] = field(default_factory=dict)

    # -- summaries (written by S6/S8 only, by convention) ----------------------
    def write_summary(self, text: str) -> None:
        self.summaries.append(text[:MAX_SUMMARY_CHARS])
        if len(self.summaries) > MAX_SUMMARIES:
            self.summaries = self.summaries[-MAX_SUMMARIES:]

    # -- facts -------------------------------------------------------------------
    def upsert_fact(self, fact: Fact) -> None:
        self.facts[fact.key] = fact

    def invalidate_keys(self, keys: list[str]) -> None:
        for k in keys:
            if k in self.facts:
                self.facts[k].valid_until = -1   # expired; S1 refresh required

    def invalidate_all_pose_facts(self) -> None:
        self.invalidate_keys([k for k in self.facts if k.startswith("pose:")])

    def fact_valid(self, key: str, world_version: int, ttl_versions: int) -> bool:
        f = self.facts.get(key)
        if f is None:
            return False
        if f.valid_until is not None and world_version >= f.valid_until:
            return False
        return (world_version - f.world_version) <= ttl_versions

    def stale_dependency_keys(self, deps: list[str], world_version: int,
                              ttl_versions: int) -> list[str]:
        return [k for k in deps if not self.fact_valid(k, world_version, ttl_versions)]

    # -- planner-facing rendering -------------------------------------------------
    def render(self, limit: int = 5) -> str:
        lines = [f"- {s}" for s in self.summaries[-limit:]]
        return "\n".join(lines) if lines else "(no summaries)"
