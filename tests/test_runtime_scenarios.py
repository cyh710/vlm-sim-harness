"""M1 exit criteria: 10-scenario matrix + 3 sanity checks + determinism.

Sanity check 2 (budget): zeroed budget refuses BEFORE the first consuming step.
Sanity check 3 (determinism): same seed twice -> identical canonical traces.
"""
import unittest

from harness.adapters.scenarios import SCENARIOS, run_scenario
from harness.spec.budget import Budget
from harness.spec.registry import default_registry
from harness.spec.types import ExitReason


class TestScenarioMatrix(unittest.TestCase):
    def _run(self, name, seed=1):
        return run_scenario(name, seed)

    # 1 ------------------------------------------------------------------
    def test_normal_success(self):
        r, rt = self._run("normal")
        self.assertEqual(r["exit_reason"], "SUCCESS")
        self.assertTrue(r["online_claim_success"])
        self.assertEqual(r["grasp_attempts"], 1)

    # 2: F2 signature invariant -------------------------------------------
    def test_timeout_after_effect_never_resend(self):
        r, rt = self._run("timeout_after_effect")
        self.assertEqual(r["exit_reason"], "SUCCESS")
        self.assertTrue(r["online_claim_success"])
        # THE invariant: receipt lost, effect re-verified from evidence —
        # exactly one physical grasp attempt, never re-sent
        self.assertEqual(r["grasp_attempts"], 1)
        self.assertGreaterEqual(rt.trace.count(event="reverify_after_timeout"), 1)

    # 3 ------------------------------------------------------------------
    def test_grasp_empty_once_recovers(self):
        r, rt = self._run("grasp_empty_once")
        self.assertEqual(r["exit_reason"], "SUCCESS")
        self.assertEqual(r["grasp_attempts"], 2)

    # 4: invariant 5 — done != success -------------------------------------
    def test_early_done_refused_by_global_acceptance(self):
        r, rt = self._run("early_done_once")
        self.assertEqual(r["exit_reason"], "SUCCESS")
        self.assertEqual(rt.trace.count(event="global_accept", verdict="failed"), 1)
        self.assertEqual(rt.trace.count(event="global_accept", verdict="success"), 1)

    # 5 ------------------------------------------------------------------
    def test_stale_pose_rejected_then_recovered(self):
        r, rt = self._run("stale_pose")
        self.assertEqual(r["exit_reason"], "SUCCESS")
        self.assertGreaterEqual(
            rt.trace.count(event="validate", reason_code="PERCEPTION_STALE"), 1)

    # 6 ------------------------------------------------------------------
    def test_occluded_no_blind_grasp(self):
        r, rt = self._run("occluded")
        self.assertEqual(r["exit_reason"], "SUCCESS")
        self.assertGreaterEqual(
            rt.trace.count(event="validate", reason_code="PRECONDITION_UNMET"), 1)
        self.assertEqual(rt.attempts.count_name("move_camera"), 1)
        # the first grasp attempt must have been REFUSED, never dispatched
        self.assertEqual(rt.attempts.count_name("grasp"), 1)

    # 7: sanity 2 + facts preserved ---------------------------------------
    def test_budget_exhausted_preserves_facts(self):
        r, rt = self._run("budget_exhausted")
        self.assertEqual(r["exit_reason"], "BUDGET_EXHAUSTED")
        self.assertFalse(r["online_claim_success"])
        self.assertGreaterEqual(r["progress_entries"], 1)   # grasp commit kept

    def test_zero_budget_rejects_before_first_action(self):
        r, rt = run_scenario_budget_zero()
        self.assertEqual(r["exit_reason"], "BUDGET_EXHAUSTED")
        self.assertEqual(r["physical_actions"], 0)
        self.assertEqual(rt.attempts.count_name("grasp"), 0)

    # 8 ------------------------------------------------------------------
    def test_unsafe_exits_through_safety_not_s7(self):
        r, rt = self._run("unsafe")
        self.assertEqual(r["exit_reason"], "SAFETY_STOP")
        self.assertEqual(rt.trace.count(event="recover"), 0)
        self.assertEqual(r["physical_actions"], 0)

    # 9: PARTIAL commits verified subset only -------------------------------
    def test_partial_commits_verified_subset(self):
        r, rt = self._run("partial")
        self.assertGreaterEqual(rt.trace.count(event="verify", verdict="partial"), 1)
        summaries = rt.progress.verified_effect_summaries()
        self.assertTrue(any("gripper_open" in s for s in summaries))
        self.assertFalse(any("contained" in s for s in summaries))
        self.assertEqual(r["exit_reason"], "FAILED")   # goal never fully met

    # 10 -----------------------------------------------------------------
    def test_no_progress_watchdog(self):
        r, rt = self._run("no_progress")
        self.assertEqual(r["exit_reason"], "FAILED")
        self.assertEqual(r["physical_actions"], 0)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_identical_traces(self):
        # SANITY CHECK 3: same seed twice -> identical canonical traces
        from harness.adapters.scenarios import build
        runs = []
        for _ in range(2):
            rt = build("normal")
            rt.run(seed=7)
            runs.append(rt.trace.canonical())
        self.assertEqual(runs[0], runs[1])

    def test_different_seeds_differ(self):
        from harness.adapters.scenarios import build
        t1 = build("normal"); t1.run(seed=1)
        t2 = build("normal"); t2.run(seed=2)
        self.assertNotEqual(t1.trace.canonical(), t2.trace.canonical())


def run_scenario_budget_zero():
    from harness.adapters.mock_adapter import FakeAdapter
    from harness.planning.planner import ScriptedPlanner
    from harness.adapters.scenarios import _standard_plan, GOAL_PICK_PLACE
    from harness.spec.runtime import Runtime
    from harness.spec.safety import SafetyConfig, SafetySupervisor
    from harness.spec.trace import Trace
    from harness.spec.validator import Validator
    from harness.spec.verifier import Verifier
    adapter = FakeAdapter()
    registry = default_registry(task_goal=GOAL_PICK_PLACE)
    safety = SafetySupervisor(SafetyConfig())
    planner = ScriptedPlanner(steps=_standard_plan())
    rt = Runtime(adapter=adapter, planner=planner, registry=registry,
                 budget=Budget(max_physical_actions=0), trace=Trace("mock", "zero"),
                 validator=Validator(registry, safety, ttl_versions=1),
                 verifier=Verifier())
    summary = rt.run(seed=1)
    r = summary.to_episode_result("mock", "zero", {})
    return r, rt


if __name__ == "__main__":
    unittest.main()
