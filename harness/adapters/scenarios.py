"""10-scenario mock matrix (experiment plan section 2.3).

Each scenario returns a fully wired runtime: adapter + scripted planner +
registry(with scenario task_goal) + budget + safety config. Expected outcomes
are asserted by tests/test_runtime_scenarios.py and printed by run_mock.py.
"""
from __future__ import annotations

from harness.adapters.mock_adapter import FakeAdapter
from harness.planning.planner import ScriptedPlanner
from harness.spec.budget import Budget
from harness.spec.registry import default_registry
from harness.spec.runtime import Runtime
from harness.spec.safety import SafetyConfig, SafetySupervisor
from harness.spec.trace import Trace
from harness.spec.validator import Validator
from harness.spec.verifier import Verifier

GOAL_PICK_PLACE = {
    "op": "all_of", "children": [
        {"op": "contained", "args": {"object_id": "red_cup", "container_id": "bin"}},
        {"op": "gripper_open", "args": {}},
    ]}

GOAL_HOLDING = {"op": "holding", "args": {"object_id": "red_cup"}}


def _cup_target(world):
    obj = world.object_by_id("red_cup")
    return [obj.pose.x, obj.pose.y, obj.pose.z]


def _bin_target(world):
    obj = world.object_by_id("bin")
    return [obj.pose.x, obj.pose.y, obj.pose.z + 0.06]


def _standard_plan(lag_versions: int | None = None):
    steps = [
        {"kind": "query", "name": "where_is",
         "args": {"object_id": "red_cup"}},
        {"kind": "action", "name": "grasp",
         "args": {"object_id": "red_cup", "target_xyz": _cup_target},
         **({"lag_versions": lag_versions, "apply_lag_times": 1} if lag_versions else {})},
        {"kind": "action", "name": "place",
         "args": {"object_id": "red_cup", "container_id": "bin",
                  "target_xyz": _bin_target}},
        {"kind": "control", "name": "done", "args": {}},
    ]
    return steps


def _build(name: str, **arm) -> Runtime:
    cfg = SCENARIOS[name]
    adapter = FakeAdapter(**cfg["adapter"])
    registry = default_registry(task_goal=cfg.get("goal", GOAL_PICK_PLACE))
    budget = Budget(**cfg.get("budget", {}))
    safety = SafetySupervisor(SafetyConfig(**cfg.get("safety", {})))
    planner = ScriptedPlanner(**cfg["planner"])
    trace = Trace(run_id="mock", episode_id=name)
    validator = Validator(registry, safety, ttl_versions=1)
    verifier = Verifier()
    # arm overrides (M5 §6.1): s5_verifier/recovery/replan_enabled/ledger
    arm_cfg = {**cfg.get("arm", {}), **arm}
    return Runtime(adapter=adapter, planner=planner, registry=registry,
                   budget=budget, trace=trace, validator=validator,
                   verifier=verifier,
                   max_no_progress_cycles=cfg.get("max_no_progress_cycles", 4),
                   **arm_cfg)


def build(name: str, **arm) -> Runtime:
    return _build(name, **arm)


# override for the occluded scenario: react to PRECONDITION_UNMET by moving
# the camera once, then let the plan continue
_OCCLUSION_OVERRIDE = (
    lambda ctx: (ctx.last_event.get("last_validation") or {}).get("reason_code")
    == "PRECONDITION_UNMET",
    lambda ctx: ScriptedPlanner._build_call(
        {"kind": "action", "name": "move_camera",
         "args": {"target_xyz": [0.30, 0.0, 0.25], "object_id": "red_cup",
                  "r": 0.3}}, ctx, "c_movecam"))

SCENARIOS: dict[str, dict] = {
    # 1. plain success: grasp -> place -> done, goal accepted
    "normal": {
        "adapter": {}, "planner": {"steps": _standard_plan()},
        "goal": GOAL_PICK_PLACE,
    },
    # 2. F2 signature fault: grasp applied physically, receipt TIMEOUT;
    #    MUST re-verify same call_id, never re-send (grasp_attempts == 1)
    "timeout_after_effect": {
        "adapter": {"grasp_timeout_applied": True},
        "planner": {"steps": _standard_plan()},
        "goal": GOAL_PICK_PLACE,
    },
    # 3. first grasp closes on nothing (F1); R1 retries with shifted point
    "grasp_empty_once": {
        "adapter": {"grasp_empty_first": True},
        "planner": {"steps": _standard_plan()},
        "goal": GOAL_PICK_PLACE,
    },
    # 4. planner declares done early; global acceptance must refuse
    "early_done_once": {
        "adapter": {},
        "planner": {"steps": [
            {"kind": "action", "name": "grasp",
             "args": {"object_id": "red_cup", "target_xyz": _cup_target}},
            {"kind": "control", "name": "done", "args": {}},
            {"kind": "action", "name": "place",
             "args": {"object_id": "red_cup", "container_id": "bin",
                      "target_xyz": _bin_target}},
            {"kind": "control", "name": "done", "args": {}},
        ]},
        "goal": GOAL_PICK_PLACE,
    },
    # 5. call built on a stale world version -> S3 PERCEPTION_STALE -> S1
    "stale_pose": {
        "adapter": {"external_disturbance": True},
        "planner": {"steps": _standard_plan(lag_versions=2)},
        "goal": GOAL_PICK_PLACE,
    },
    # 6. target invisible until camera moved; no blind grasping
    "occluded": {
        "adapter": {"occluded_target": True},
        "planner": {"steps": [
            {"kind": "action", "name": "grasp",
             "args": {"object_id": "red_cup", "target_xyz": _cup_target}},
            {"kind": "control", "name": "done", "args": {}},
        ], "overrides": [_OCCLUSION_OVERRIDE]},
        "goal": GOAL_HOLDING,
    },
    # 7. physical budget 1: second action refused, facts preserved on exit
    "budget_exhausted": {
        "adapter": {},
        "planner": {"steps": _standard_plan()},
        "budget": {"max_physical_actions": 1},
        "goal": GOAL_PICK_PLACE,
    },
    # 8. grasp target inside forbidden zone -> S3 safety veto -> SAFETY_STOP
    "unsafe": {
        "adapter": {},
        "planner": {"steps": _standard_plan()},
        "safety": {"forbidden_zones": [{"min": [0.0, -0.15, 0.0],
                                        "max": [0.20, 0.05, 0.30]}]},
        "goal": GOAL_PICK_PLACE,
    },
    # 9. place releases (gripper verified) but contained unverifiable:
    #    PARTIAL verdict, S6 commits only the verified subset
    "partial": {
        "adapter": {"place_bounce_rim": True},
        "planner": {"steps": _standard_plan()},
        "goal": GOAL_PICK_PLACE,
        "max_no_progress_cycles": 4,
    },
    # 10. observe/where_is loop with no actions -> watchdog exits
    "no_progress": {
        "adapter": {},
        "planner": {"steps": [
            {"kind": "query", "name": "where_is",
             "args": {"object_id": "red_cup"}, "repeat_if": lambda ctx: True},
        ]},
        "goal": GOAL_PICK_PLACE,
        "max_no_progress_cycles": 4,
    },
}


def run_scenario(name: str, seed: int = 1) -> dict:
    rt = _build(name)
    summary = rt.run(seed)
    out = summary.to_episode_result("mock", f"{name}_s{seed}", {})
    out["scenario"] = name
    out["seed"] = seed
    return out, rt
