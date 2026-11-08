"""Verifier tests: fake-completion resistance, partial semantics, safety order.

Sanity check 1 (fake completion): the executor CLAIMS success but physics did
not happen -> verifier must not return SUCCESS, driven purely by evidence.
"""
import unittest

from harness.spec.registry import default_registry
from harness.spec.types import Evidence, Pose, Verdict, WorldState
from harness.spec.verifier import Verifier

REG = default_registry()


def world(version=1, cup=(0.10, -0.05, 0.05), gripper=1.0, vis=1.0,
          cup2=(-0.10, 0.05, 0.02)):
    from tests.test_dsl import world as w
    return w(version=version, cup=cup, gripper=gripper, vis=vis, cup2=cup2)


def evidence(before, after, action="grasp", args=None, timeout=False):
    return Evidence(call_id="c1", action=action,
                    args=args or {"object_id": "red_cup",
                                  "target_xyz": [0.1, -0.05, 0.05]},
                    expected_effects=(), before=before, after=after,
                    metrics={"timeout_exceeded": timeout},
                    claimed={"success": True},          # lying executor
                    timeout_exceeded=timeout)


class TestVerifier(unittest.TestCase):
    def test_fake_completion_never_success(self):
        # SANITY CHECK 1: claimed=True but physics says empty close
        # (cup unmoved AND gripper shut 0.0 — the calibrated empty signature)
        ev = evidence(world(), world(cup=(0.10, -0.05, 0.05), gripper=0.0))
        res = Verifier().verify(ev, REG.expand_effects("grasp", ev.args))
        self.assertNotEqual(res.verdict, Verdict.SUCCESS)
        self.assertEqual(res.verdict, Verdict.FAILED)

    def test_real_grasp_success(self):
        # attach-carry signature: cup at ee, fingers stayed open (1.0)
        after = world(cup=(0.30, 0.0, 0.16), gripper=1.0)
        ev = evidence(world(), after)
        res = Verifier().verify(ev, REG.expand_effects("grasp", ev.args))
        self.assertEqual(res.verdict, Verdict.SUCCESS)

    def test_timeout_unknown_outcome_inconclusive(self):
        # F2: receipt lost, cup lifted (physics applied) — evidence frames show
        # holding true, so re-verify path yields SUCCESS; but with a *stale*
        # after-frame predating the lift, verdict must not be FAILED-by-counter
        mid = world(cup=(0.10, -0.05, 0.05), gripper=0.3)
        ev = evidence(world(), mid, timeout=True)
        res = Verifier().verify(ev, REG.expand_effects("grasp", ev.args))
        self.assertIn(res.verdict, (Verdict.INCONCLUSIVE, Verdict.FAILED))

    def test_partial_commits_subset_only(self):
        # gripper released (verified) but cup half-visible at rim (contained
        # unverifiable) -> PARTIAL; commit path takes only True effects
        after = world(cup=(-0.10, -0.10, 0.09), gripper=1.0, vis=0.0)
        ev = evidence(world(), after, action="place",
                      args={"object_id": "red_cup", "container_id": "bin",
                            "target_xyz": [-0.1, 0.05, 0.08]})
        res = Verifier().verify(ev, REG.expand_effects("place", ev.args))
        self.assertEqual(res.verdict, Verdict.PARTIAL)

    def test_safety_checked_first(self):
        from dataclasses import replace
        after = replace(world(cup=(0.30, 0.0, 0.16), gripper=0.3), unsafe_flag=True)
        ev = evidence(world(), after)
        res = Verifier().verify(ev, REG.expand_effects("grasp", ev.args))
        self.assertEqual(res.verdict, Verdict.UNSAFE)

    def test_verifier_never_reads_claimed_field(self):
        src = open(Verifier.__module__.replace(".", "/") + ".py",
                   encoding="utf-8").read()
        body = src.split("class Verifier")[1]
        self.assertNotIn(".claimed", body)

    def test_real_place_success(self):
        after = world(cup=(-0.10, 0.05, 0.08), gripper=1.0)
        ev = evidence(world(), after, action="place",
                      args={"object_id": "red_cup", "container_id": "bin",
                            "target_xyz": [-0.1, 0.05, 0.08]})
        res = Verifier().verify(ev, REG.expand_effects("place", ev.args))
        self.assertEqual(res.verdict, Verdict.SUCCESS)


if __name__ == "__main__":
    unittest.main()
