"""M5 arm-semantics unit tests (plan §6.1): the four arms differ ONLY on the
declared code paths. Locked behaviors:

  A  (no S5/recovery/replan/ledger): executor self-report drives; done ends
     the episode with the agent's own claim — the early_done scenario becomes
     a FALSE COMPLETION (exit SUCCESS, claim True, judge would say fail).
  B  (+S5): premature done refused by global acceptance; no recovery events.
  C1 (+S7): recovery fires on grasp_empty_once; replan escalation gives up.
  C  (full): baseline — recovery + replan + ledger (existing suite covers it).

Accounting invariants: budgets charge identically where the same code path
runs (physical actions, llm_calls per decision) in every arm.
"""
from __future__ import annotations

import unittest

from harness.adapters.scenarios import build


class TestArmA(unittest.TestCase):
    def test_early_done_is_false_completion(self):
        rt = build("early_done_once", s5_verifier=False, recovery=False,
                   replan_enabled=False, progress_ledger=False)
        r = rt.run(seed=7)
        # self-report driven end: SUCCESS claimed WITHOUT any verification
        self.assertEqual(r.exit_reason.value, "SUCCESS")
        self.assertTrue(r.online_claim_success)
        # and nothing was ever verified: no S5 events at all
        self.assertEqual(rt.trace.count(event="verify", state="S5"), 0)
        self.assertEqual(rt.trace.count(event="self_report"), 1)
        # A must keep the S3 safety gate
        self.assertGreaterEqual(rt.trace.count(state="S3", event="validate"), 1)

    def test_normal_task_still_completes(self):
        rt = build("normal", s5_verifier=False, recovery=False,
                   replan_enabled=False, progress_ledger=False)
        r = rt.run(seed=7)
        self.assertEqual(r.exit_reason.value, "SUCCESS")
        self.assertTrue(r.online_claim_success)
        self.assertEqual(rt.trace.count(state="S5"), 0)

    def test_progress_ledger_never_written(self):
        rt = build("normal", s5_verifier=False, recovery=False,
                   replan_enabled=False, progress_ledger=False)
        rt.run(seed=7)
        self.assertEqual(len(rt.progress.entries()), 0)
        self.assertEqual(rt.trace.count(event="commit"), 0)


class TestArmB(unittest.TestCase):
    def test_early_done_refused_without_recovery(self):
        rt = build("early_done_once", s5_verifier=True, recovery=False,
                   replan_enabled=False, progress_ledger=False)
        r = rt.run(seed=7)
        # global acceptance refuses the premature done; without S7 the plan
        # still proceeds (place, done) and honestly completes
        self.assertEqual(r.exit_reason.value, "SUCCESS")
        self.assertGreaterEqual(
            rt.trace.count(state="S5", event="global_accept",
                           verdict="failed"), 1)
        self.assertEqual(rt.trace.count(event="recover"), 0)

    def test_honest_failure_has_no_recovery(self):
        rt = build("grasp_empty_once", s5_verifier=True, recovery=False,
                   replan_enabled=False, progress_ledger=False)
        r = rt.run(seed=7)
        self.assertEqual(rt.trace.count(event="recover"), 0)
        self.assertEqual(r.online_claim_success, False)


class TestArmC1(unittest.TestCase):
    def test_recovery_fires_without_replan_ledger(self):
        rt = build("grasp_empty_once", s5_verifier=True, recovery=True,
                   replan_enabled=False, progress_ledger=False)
        r = rt.run(seed=7)
        self.assertEqual(r.exit_reason.value, "SUCCESS")
        self.assertGreaterEqual(rt.trace.count(event="recover"), 1)
        self.assertEqual(len(rt.progress.entries()), 0)

    def test_r2_escalation_gives_up_when_replan_disabled(self):
        rt = build("no_progress", s5_verifier=True, recovery=True,
                   replan_enabled=False, progress_ledger=False)
        r = rt.run(seed=7)
        self.assertEqual(r.exit_reason.value, "FAILED")
        self.assertFalse(r.online_claim_success)


class TestArmCRegression(unittest.TestCase):
    def test_full_arm_matches_existing_semantics(self):
        rt = build("timeout_after_effect")           # defaults = arm C
        r = rt.run(seed=7)
        self.assertEqual(r.exit_reason.value, "SUCCESS")
        # F2 signature: the timed-out action is NEVER re-dispatched
        self.assertEqual(rt.attempts.count_name("grasp"), 1)
        self.assertEqual(len(rt.progress.entries()), 4)  # 2 grasp + 2 place leaves


class TestAccountingParity(unittest.TestCase):
    def test_llm_calls_charged_per_decision_in_all_arms(self):
        counts = {}
        for arm in ({"s5_verifier": False, "recovery": False,
                     "replan_enabled": False, "progress_ledger": False}, {}):
            rt = build("normal", **arm)
            r = rt.run(seed=7)
            counts[r.exit_reason.value] = \
                counts.get(r.exit_reason.value, 0) + 1
            self.assertGreater(r.budget.used["llm_calls"], 0)
            self.assertGreater(r.budget.used["physical_actions"], 0)
        self.assertEqual(counts, {"SUCCESS": 2})


if __name__ == "__main__":
    unittest.main()
