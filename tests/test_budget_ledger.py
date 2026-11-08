"""Budget charge-before-consume + ledger invariants."""
import unittest

from harness.spec.budget import Budget, BudgetExhausted
from harness.spec.ledger import AttemptLedger, ProgressLedger
from harness.spec.types import CallState, ProgressDelta, Status


class TestBudget(unittest.TestCase):
    def test_zero_budget_rejects_before_use(self):
        b = Budget(max_physical_actions=0)
        self.assertFalse(b.can_charge("physical_actions"))
        with self.assertRaises(BudgetExhausted):
            b.check_available("physical_actions")   # BEFORE any consumption
        self.assertEqual(b.used["physical_actions"], 0)

    def test_charge_then_exhaust(self):
        b = Budget(max_physical_actions=1)
        b.charge("physical_actions")
        with self.assertRaises(BudgetExhausted):
            b.charge("physical_actions")
        self.assertEqual(b.remaining()["physical_actions"], 0)

    def test_six_items_present(self):
        b = Budget()
        self.assertEqual(len(b.remaining()), 6)


class TestAttemptLedger(unittest.TestCase):
    def test_dispatch_then_settle(self):
        led = AttemptLedger()
        rec = led.record_dispatch("c1", "grasp", {"object_id": "x"}, 0.0)
        self.assertEqual(led.state_of("c1"), CallState.DISPATCHED)
        led.settle("c1", Status.REACHED, 1.0)
        self.assertEqual(led.state_of("c1"), CallState.SETTLED)
        self.assertEqual(rec.attempt_id, "a_0001")

    def test_unknown_never_reported_settled(self):
        led = AttemptLedger()
        led.record_dispatch("c1", "grasp", {}, 0.0)
        led.mark_unknown("c1", 1.0)
        self.assertEqual(led.state_of("c1"), CallState.UNKNOWN)
        self.assertIsNone(led.status_of("c1"))

    def test_failures_counted(self):
        led = AttemptLedger()
        led.record_dispatch("c1", "grasp", {}, 0.0)
        led.settle("c1", Status.BLOCKED, 1.0)
        led.record_dispatch("c2", "grasp", {}, 2.0)
        led.settle("c2", Status.REACHED, 3.0)
        self.assertEqual(led.count_name("grasp"), 2)
        self.assertEqual(led.count_failed(), 1)


class TestProgressLedger(unittest.TestCase):
    def test_commit_only_api(self):
        led = ProgressLedger()
        led.commit(ProgressDelta(verified_effects=({"op": "holding",
                                                    "args": {"object_id": "x"}},),
                                 source_call_id="c1", evidence_refs=("e",)))
        self.assertEqual(len(led.entries()), 1)
        self.assertIn("holding", led.verified_effect_summaries()[0])

    def test_empty(self):
        led = ProgressLedger()
        self.assertEqual(led.entries(), ())


if __name__ == "__main__":
    unittest.main()
