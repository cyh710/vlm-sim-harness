"""Validator S3 checks: order, query/action asymmetry (patch P3), safety veto."""
import unittest

from harness.spec.budget import Budget
from harness.spec.memory import Memory
from harness.spec.registry import default_registry
from harness.spec.safety import SafetyConfig, SafetySupervisor
from harness.spec.validator import Validator
from harness.spec.types import ReasonCode, ToolCall, WorldState


def make_world(version=5, gripper=1.0, vis=1.0):
    from tests.test_dsl import world as w
    return w(version=version, gripper=gripper, vis=vis, cup2=(-0.10, 0.05, 0.02))


def grasp_call(version=None):
    return ToolCall(call_id="c1", kind="action", name="grasp",
                    args={"object_id": "red_cup", "target_xyz": [0.1, -0.05, 0.05]},
                    world_version=version if version is not None else 5)


class TestValidator(unittest.TestCase):
    def setUp(self):
        self.registry = default_registry()
        self.safety = SafetySupervisor(SafetyConfig())
        self.v = Validator(self.registry, self.safety, ttl_versions=1)
        self.memory = Memory()
        # prime facts at current version so freshness passes by default
        for key, val in (("pose:red_cup", {}), ("pose:bin", {}), ("gripper", {})):
            from harness.spec.memory import Fact
            self.memory.upsert_fact(Fact(key=key, value=val, world_version=5))

    def test_valid_grasp_allowed(self):
        res = self.v.validate(grasp_call(), make_world(), Budget(), self.memory)
        self.assertTrue(res.allowed)

    def test_unknown_action_rejected(self):
        call = ToolCall(call_id="c2", kind="action", name="fly",
                        args={}, world_version=5)
        res = self.v.validate(call, make_world(), Budget(), self.memory)
        self.assertFalse(res.allowed)
        self.assertEqual(res.reason_code, "SCHEMA_INVALID")

    def test_missing_object_rejected(self):
        call = ToolCall(call_id="c3", kind="action", name="grasp",
                        args={"object_id": "ghost", "target_xyz": [0, 0, 0]},
                        world_version=5)
        res = self.v.validate(call, make_world(), Budget(), self.memory)
        self.assertFalse(res.allowed)

    def test_stale_version_rejected(self):
        # call built 3 versions ago -> PERCEPTION_STALE with S1 suggestion
        res = self.v.validate(grasp_call(version=2), make_world(), Budget(),
                              self.memory)
        self.assertFalse(res.allowed)
        self.assertEqual(res.reason_code, ReasonCode.PERCEPTION_STALE)
        self.assertEqual(res.suggested_next, "S1")

    def test_query_exempt_from_freshness(self):
        # patch P3: where_is built on an ancient version must NOT self-lock
        call = ToolCall(call_id="c4", kind="query", name="where_is",
                        args={"object_id": "red_cup"}, world_version=1)
        res = self.v.validate(call, make_world(), Budget(), self.memory)
        self.assertTrue(res.allowed)

    def test_precondition_requires_visible_target(self):
        # grasp precondition = visible_ratio_min (gripper band removed after
        # M2 calibration: the skill itself opens first on retry)
        res = self.v.validate(grasp_call(), make_world(vis=0.0), Budget(),
                              self.memory)
        self.assertFalse(res.allowed)
        self.assertEqual(res.reason_code, "PRECONDITION_UNMET")

    def test_grasp_allowed_with_closed_gripper(self):
        # R1 retry signature: previous empty close left the gripper shut; the
        # skill opens first, so validation must not block on gripper state
        res = self.v.validate(grasp_call(), make_world(gripper=0.0), Budget(),
                              self.memory)
        self.assertTrue(res.allowed)

    def test_place_requires_holding_precondition(self):
        call = ToolCall(call_id="c5", kind="action", name="place",
                        args={"object_id": "red_cup", "container_id": "bin",
                              "target_xyz": [-0.1, 0.05, 0.08]},
                        world_version=5)
        # world gripper wide open -> not holding -> precondition unmet
        res = self.v.validate(call, make_world(gripper=1.0), Budget(),
                              self.memory)
        self.assertFalse(res.allowed)
        self.assertEqual(res.reason_code, "PRECONDITION_UNMET")

    def test_safety_veto_not_recoverable(self):
        call = ToolCall(call_id="c6", kind="action", name="grasp",
                        args={"object_id": "red_cup",
                              "target_xyz": [2.0, 0.0, 0.05]},   # outside workspace
                        world_version=5)
        res = self.v.validate(call, make_world(), Budget(), self.memory)
        self.assertFalse(res.allowed)
        self.assertEqual(res.reason_code, ReasonCode.COLLISION_RISK)
        self.assertFalse(res.recoverable)

    def test_zero_budget_rejected_before_dispatch(self):
        res = self.v.validate(grasp_call(), make_world(),
                              Budget(max_physical_actions=0), self.memory)
        self.assertFalse(res.allowed)
        self.assertEqual(res.reason_code, "BUDGET_EXHAUSTED")

    def test_wait_over_cap_rejected(self):
        call = ToolCall(call_id="c7", kind="query", name="wait",
                        args={"seconds": 999}, world_version=5)
        res = self.v.validate(call, make_world(), Budget(), self.memory)
        self.assertFalse(res.allowed)
        self.assertEqual(res.reason_code, "SCHEMA_INVALID")


if __name__ == "__main__":
    unittest.main()
