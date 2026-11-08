"""LLMPlanner unit tests (M4): protocol layer only — the client is a canned
callable, no network. Covers: schema-valid passthrough, one-repair rule,
protocol error taxonomy, trust boundary (call_id/effects/version stamped by
harness, not model), payload oracle isolation, §9.1 payload shape.
"""
from __future__ import annotations

import unittest

from harness.planning import llm_planner as lp
from harness.spec import planner_prompt as pp
from harness.spec.registry import default_registry
from harness.spec.runtime import PlannerContext
from harness.spec.types import (ObjInstance, PlannerProtocolError, Pose,
                                ToolCall, WorldState)

GOAL = {"op": "all_of", "children": [
    {"op": "contained", "args": {"object_id": "rubbish", "container_id": "bin"}},
    {"op": "gripper_open", "args": {}},
]}
TASK_META = {"task_id": "t001", "instruction": "put rubbish in the bin"}


def _world(version: int = 7) -> WorldState:
    obj = ObjInstance(id="rubbish", label="rubbish", view="front",
                      pose=Pose(0.2, -0.1, 0.03), sigma_m=0.005,
                      size=(0.05, 0.05, 0.05), visible_ratio=1.0,
                      quality="ok", source="cad", last_seen_t=12.3)
    return WorldState(version=version, timestamp=0.0, ee_pose=Pose(),
                      gripper_open=1.0, joints={}, objects=(obj,))


def _ctx(**kw) -> PlannerContext:
    return PlannerContext(
        world=kw.get("world", _world()),
        memory_text="", progress_summaries=("holding(rubbish)",),
        budget_remaining={"physical_actions": 5, "queries": 9,
                          "llm_calls": 10, "replans": 2},
        last_event=kw.get("last_event", {}),
        recent_attempts=({"call_id": "c_0001", "name": "grasp",
                          "result": "reached"},))


def scripted_client(responses: list[str]):
    calls = {"n": 0}

    def client(system: str, user: str) -> str:
        calls["n"] += 1
        return responses.pop(0)
    client.calls = calls
    return client


VALID_GRASP = ('{"kind":"action","name":"grasp","args":{"object_id":"rubbish",'
               '"target_xyz":[0.2,-0.1,0.03]},"world_version":999,'
               '"expected_effects":[{"op":"holding","args":{"object_id":'
               '"rubbish"}}],"timeout_ms":12000}')


class TestProtocol(unittest.TestCase):
    def setUp(self):
        self.reg = default_registry(task_goal=GOAL)

    def test_valid_action_passthrough(self):
        client = scripted_client([VALID_GRASP])
        p = lp.LLMPlanner(client, self.reg, TASK_META)
        call = p.decide(_ctx())
        self.assertIsInstance(call, ToolCall)
        self.assertEqual(call.kind, "action")
        self.assertEqual(call.name, "grasp")
        self.assertEqual(call.args["object_id"], "rubbish")
        # trust boundary: runtime assigns call_id; version = observed world
        self.assertEqual(call.call_id, "")
        self.assertEqual(call.world_version, 7)          # NOT the echoed 999
        # patch P4: model-declared effects are dropped
        self.assertEqual(call.expected_effects, ())
        # timeout clamped to registry max
        self.assertEqual(call.timeout_ms, 10000)
        self.assertEqual(p.stats["actions_proposed"], 1)
        self.assertEqual(p.stats["llm_calls"], 1)

    def test_fenced_json_tolerated(self):
        client = scripted_client([f"```json\n{VALID_GRASP}\n```"])
        p = lp.LLMPlanner(client, self.reg, TASK_META)
        call = p.decide(_ctx())
        self.assertEqual(call.name, "grasp")

    def test_one_repair_round(self):
        client = scripted_client(["I cannot answer that.", VALID_GRASP])
        p = lp.LLMPlanner(client, self.reg, TASK_META)
        call = p.decide(_ctx())
        self.assertEqual(call.name, "grasp")
        self.assertEqual(p.stats["repairs_used"], 1)
        self.assertEqual(p.stats["llm_calls"], 2)
        self.assertEqual(p.stats["json_valid_first_try"], 0)

    def test_two_failures_raise(self):
        client = scripted_client(["nope", "still nope"])
        p = lp.LLMPlanner(client, self.reg, TASK_META)
        with self.assertRaises(PlannerProtocolError):
            p.decide(_ctx())
        self.assertEqual(p.stats["protocol_errors"], 1)
        self.assertEqual(p.stats["error_attribution"].get(lp.E_JSON), 1)

    def test_empty_output_taxed(self):
        client = scripted_client(["", "  "])
        p = lp.LLMPlanner(client, self.reg, TASK_META)
        with self.assertRaises(PlannerProtocolError):
            p.decide(_ctx())
        self.assertEqual(p.stats["error_attribution"].get(lp.E_EMPTY), 1)

    def test_multiple_objects_taxed(self):
        two = VALID_GRASP + "\n" + '{"kind":"query","name":"observe"}'
        client = scripted_client([two, two])
        p = lp.LLMPlanner(client, self.reg, TASK_META)
        with self.assertRaises(PlannerProtocolError):
            p.decide(_ctx())
        self.assertEqual(p.stats["error_attribution"].get(lp.E_MULTI), 1)

    def test_unknown_skill_taxed(self):
        bad = ('{"kind":"action","name":"fly","args":{"object_id":"rubbish"}}')
        client = scripted_client([bad, bad])
        p = lp.LLMPlanner(client, self.reg, TASK_META)
        with self.assertRaises(PlannerProtocolError):
            p.decide(_ctx())
        self.assertEqual(p.stats["error_attribution"].get(lp.E_NAME), 1)

    def test_missing_required_arg_taxed(self):
        bad = '{"kind":"action","name":"grasp","args":{"object_id":"rubbish"}}'
        client = scripted_client([bad, bad])
        p = lp.LLMPlanner(client, self.reg, TASK_META)
        with self.assertRaises(PlannerProtocolError):
            p.decide(_ctx())
        self.assertEqual(p.stats["error_attribution"].get(lp.E_ARGS), 1)

    def test_bad_kind_taxed(self):
        bad = '{"kind":"teleport","name":"grasp","args":{}}'
        client = scripted_client([bad, bad])
        p = lp.LLMPlanner(client, self.reg, TASK_META)
        with self.assertRaises(PlannerProtocolError):
            p.decide(_ctx())
        self.assertEqual(p.stats["error_attribution"].get(lp.E_KIND), 1)

    def test_move_camera_requires_object_id(self):
        # M4 regression: template binds $object_id; a proposal without it must
        # be a repairable BAD_ARGS, never an S5 KeyError crash
        bad = ('{"kind":"action","name":"move_camera",'
               '"args":{"target_xyz":[0.3,0.0,0.25]}}')
        good = ('{"kind":"action","name":"move_camera",'
                '"args":{"target_xyz":[0.3,0.0,0.25],'
                '"object_id":"rubbish"}}')
        p = lp.LLMPlanner(scripted_client([bad, good]), self.reg, TASK_META)
        call = p.decide(_ctx())          # repair round rescues it
        self.assertEqual(call.name, "move_camera")
        # repaired on the repair round: NOT a final protocol error
        self.assertEqual(p.stats["repairs_used"], 1)
        self.assertEqual(p.stats["protocol_errors"], 0)

        p2 = lp.LLMPlanner(scripted_client([bad, bad]), self.reg, TASK_META)
        with self.assertRaises(PlannerProtocolError):
            p2.decide(_ctx())

    def test_push_binds_via_arg_defaults(self):
        ok = ('{"kind":"action","name":"push","args":{"object_id":"rubbish",'
              '"target_xyz":[0.1,0.1,0.02],"dir":[1.0,0.0,0.0]}}')
        p = lp.LLMPlanner(scripted_client([ok]), self.reg, TASK_META)
        call = p.decide(_ctx())          # d/lateral come from arg_defaults
        self.assertEqual(call.name, "push")

    def test_query_and_control_pass(self):
        client = scripted_client(['{"kind":"query","name":"where_is",'
                                  '"args":{"object_id":"rubbish"}}',
                                  '{"kind":"control","name":"done",'
                                  '"args":{}}'])
        p = lp.LLMPlanner(client, self.reg, TASK_META)
        c1 = p.decide(_ctx())
        c2 = p.decide(_ctx())
        self.assertEqual(c1.kind, "query")
        self.assertEqual(c2.kind, "control")
        self.assertEqual(c2.name, "done")

    def test_replan_and_recover_local(self):
        client = scripted_client(['{"kind":"action","name":"grasp",'
                                  '"args":{"object_id":"rubbish",'
                                  '"target_xyz":[0,0,0]}}',
                                  '{"kind":"control","name":"replan",'
                                  '"args":{}}'])
        p = lp.LLMPlanner(client, self.reg, TASK_META)
        c = p.replan(_ctx())
        self.assertEqual(c.kind, "action")
        r = p.recover_local(_ctx(last_event={
            "failed_call": ToolCall(call_id="c1", kind="action",
                                    name="grasp", args={}, world_version=7),
            "reason_code": "NOT_HOLDING"}))
        self.assertEqual(r.kind, "control")
        self.assertEqual(r.name, "replan")     # runtime escalates R1 -> R2


class TestPayload(unittest.TestCase):
    def setUp(self):
        self.reg = default_registry(task_goal=GOAL)

    def test_payload_shape_is_9_1(self):
        payload = pp.build_payload(_ctx(), self.reg, TASK_META)
        self.assertEqual(
            sorted(payload.keys()),
            ["available_skills", "budgets_left", "constraints", "recent_attempts",
             "task", "verified_progress", "world"])
        self.assertEqual(sorted(payload["world"].keys()),
                         ["gripper_open_fraction", "objects", "version"])
        self.assertEqual(payload["task"]["instruction"],
                         "put rubbish in the bin")
        self.assertEqual(payload["task"]["goal"], GOAL)

    def test_payload_no_oracle_leakage(self):
        blob = pp.payload_json(_ctx(), self.reg, TASK_META)
        for banned in ("reward", "task.success", "judge", "success_sensor",
                       "online_claim"):
            self.assertNotIn(banned, blob)

    def test_prompt_hash_stable(self):
        # freeze reference: any SYSTEM_PROMPT edit must bump PROMPT_VERSION
        self.assertEqual(pp.prompt_hash(),
                         pp.prompt_hash())   # deterministic
        self.assertEqual(len(pp.prompt_hash()), 64)
        self.assertIn("ALLOWED_SKILLS", pp.SYSTEM_PROMPT)
        self.assertIn("单个 JSON", pp.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
