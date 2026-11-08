"""S5 Verifier: five-value verdict from trusted effects + before/after evidence.

Judgement order (design doc section 6.2):
  1. safety hit              -> UNSAFE
  2. evidence insufficiency  -> INCONCLUSIVE
  3. all effects verified    -> SUCCESS
  4. some verified, no counter-evidence -> PARTIAL
  5. explicit counter-evidence          -> FAILED(+reason_code)

Hard rules:
  - never reads evidence.claimed (executor self-report) — the dataclass field
    exists on Evidence but this module does not touch it; CI check enforces
    no other read path in harness/ either.
  - cross-signal mismatch (vision vs proprio) -> INCONCLUSIVE, never SUCCESS.
"""
from __future__ import annotations

import harness.spec.dsl as dsl
from harness.spec.types import (CheckResult, Evidence, ReasonCode, Verdict,
                                VerificationResult)

_VERDICT_ORDER = ("unsafe", "inconclusive", "failed", "partial", "success")


def _dominant(verdicts: list[Verdict]) -> Verdict:
    """Worst-case merge across effects: unsafe > inconclusive > failed >
    partial > success."""
    for v in _VERDICT_ORDER:
        if v in verdicts:
            return v
    return Verdict.INCONCLUSIVE


class Verifier:
    """Geometric layer only. Semantic layer (section 6.4) hooks in later as a
    fallback called on INCONCLUSIVE before commit; not part of M1 mock contract
    except as an injectable hook."""

    def __init__(self, semantic_layer=None):
        # semantic_layer: optional callable(evidence, unresolved_checks) ->
        # list[CheckResult]; counts against llm budget by the caller.
        self.semantic_layer = semantic_layer

    def verify(self, evidence: Evidence,
               trusted_effects: tuple[dict, ...]) -> VerificationResult:
        reason_code: str | None = None

        # 1. safety
        if evidence.after.unsafe_flag or evidence.before.unsafe_flag:
            return VerificationResult(
                verdict=Verdict.UNSAFE, reason_code=ReasonCode.COLLISION_RISK,
                observed_effects=(), checks=(), layer="geometric")

        if not trusted_effects:
            # nothing to verify (e.g., pure queries) — treat as inconclusive
            return VerificationResult(
                verdict=Verdict.INCONCLUSIVE, reason_code=None,
                observed_effects=(), checks=(), layer="geometric")

        # 2+3+4+5. evaluate each trusted effect, merge at LEAF-check level
        checks: list[CheckResult] = []
        for ast in trusted_effects:
            try:
                sub, _overall = dsl.evaluate(ast, evidence.before, evidence.after,
                                             collect_all=True)
            except dsl.DSLError:
                continue
            checks.extend(sub)

        # section 6.2 order applied to leaf checks:
        #   3. all checks pass           -> SUCCESS
        #   4. some pass, no counter     -> PARTIAL
        #   5. explicit counter-evidence -> FAILED
        #   nothing pass, no counter     -> INCONCLUSIVE
        passed = [c for c in checks if c.passed is True]
        counter_failed = [c for c in checks
                          if c.passed is False and c.counter_evidence]
        plain_failed = [c for c in checks
                        if c.passed is False and not c.counter_evidence]
        if checks and len(passed) == len(checks):
            verdict = Verdict.SUCCESS
        elif counter_failed:
            verdict = Verdict.FAILED
            if reason_code is None:
                reason_code = counter_failed[0].reason_code or \
                    ReasonCode.OUT_OF_TOLERANCE
        elif passed:
            verdict = Verdict.PARTIAL
        elif plain_failed:
            # fails without explicit counter (e.g. visibility-limited) stay
            # conservative: INCONCLUSIVE unless clearly out of tolerance
            verdict = Verdict.INCONCLUSIVE
        else:
            verdict = Verdict.INCONCLUSIVE
        layer = "geometric"

        # semantic fallback hook: only on geometric INCONCLUSIVE
        if verdict is Verdict.INCONCLUSIVE and self.semantic_layer is not None:
            unresolved = [c for c in checks if c.passed is None]
            extra = self.semantic_layer(evidence, tuple(unresolved))
            if extra:
                checks = list(checks) + list(extra)
                new_overalls = []
                for ast in trusted_effects:
                    _, overall = dsl.evaluate(ast, evidence.before, evidence.after,
                                              collect_all=True)
                    new_overalls.append(overall)
                if any(o is True for o in new_overalls) and \
                        not any(o is False for o in new_overalls):
                    verdict = Verdict.SUCCESS
                layer = "semantic"

        observed = [{"effect": c.effect, "passed": c.passed, "value": c.value}
                    for c in checks]
        return VerificationResult(verdict=verdict, reason_code=reason_code,
                                  observed_effects=tuple(observed),
                                  checks=tuple(checks), layer=layer)
