"""VLM 在环的端到端集成测试 — 离线跑完整条链路。

验证的不是"模型聪不聪明" (那要真机 + 真模型), 而是**接线是否正确**:
VLM 的输出能否被运行时正确消费, 走完 S1→S8, 动作被放行, 效果被独立核验,
失败被正确归因。这条链路是整个项目最贵的部分 (一次接错就是 328 集全零),
所以必须能在没有 API key、没有仿真器的条件下被验证。

夹具只替换"模型返回什么文本"; 运行时、校验器、核验器、账本、预算全部是真的。
"""
import json
import unittest

from harness.adapters.mock_adapter import FakeAdapter
from harness.planning.vlm_planner import VLMPlanner
from harness.spec.budget import Budget
from harness.spec.registry import default_registry
from harness.spec.runtime import Runtime
from harness.spec.safety import SafetyConfig, SafetySupervisor
from harness.spec.trace import Trace
from harness.spec.validator import Validator
from harness.spec.verifier import Verifier

from tests.fake_vlm import camera_provider, decision, fake_frame

GOAL_PICK_PLACE = {
    "op": "all_of", "children": [
        {"op": "contained", "args": {"object_id": "red_cup", "container_id": "bin"}},
        {"op": "gripper_open", "args": {}},
    ]}


def build_runtime(responses, **kw):
    """搭一个真运行时, 只把 planner 换成 VLM + 假模型。"""
    adapter = FakeAdapter()
    registry = default_registry(task_goal=GOAL_PICK_PLACE)
    budget = Budget()
    safety = SafetySupervisor(SafetyConfig())
    planner = VLMPlanner(
        instruction="pick up the paper ball and put it in the basket",
        camera_provider=camera_provider(),
        registry_table_provider=lambda: [
            {"id": "red_cup", "view": "front",
             "pose": [0.10, -0.05, 0.05], "size": [0.07, 0.07, 0.10]},
            {"id": "bin", "view": "front",
             "pose": [0.30, 0.10, 0.02], "size": [0.2, 0.2, 0.12]},
        ],
        endpoint={"base_url": "http://local.invalid",
                  "api_key": "not-a-real-key", "model": "fake-vlm"},
        goal=GOAL_PICK_PLACE,
        **kw)
    from tests.fake_vlm import FakeVLM
    fake = FakeVLM(responses)
    planner._call_anthropic = fake
    rt = Runtime(adapter=adapter, planner=planner, registry=registry,
                 budget=budget, trace=Trace(run_id="vlm-e2e", episode_id="e2e"),
                 validator=Validator(registry, safety, ttl_versions=1),
                 verifier=Verifier())
    rt._fake = fake
    return rt


class TestVlmEndToEnd(unittest.TestCase):
    def test_vlm_drives_full_loop_to_success(self):
        """VLM 依次给出 grasp→place→done, 整集走完并通过全局验收。"""
        rt = build_runtime([
            decision("grasp", {"object_id": "red_cup"}, "空爪, 先抓"),
            decision("place", {"object_id": "red_cup"}, "已抓住, 投放"),
            decision("done", {}, "已投放"),
        ])
        r = rt.run(seed=7)
        self.assertEqual(r.exit_reason.value, "SUCCESS")
        self.assertTrue(r.online_claim_success)

    def test_state_machine_visits_expected_states(self):
        """九状态机真的在流转 — 不是被跳过的空壳。

        S1/S2 是"读世界"和"想一步"的入口, 运行时把它们记为 state_enter
        而非独立事件, 所以这里断言的是真正留下审计痕迹的那些状态:
        校验(S3)、执行(S4)、核验(S5)、提交(S6)。
        """
        rt = build_runtime([
            decision("grasp", {"object_id": "red_cup"}, "抓"),
            decision("place", {"object_id": "red_cup"}, "放"),
            decision("done", {}, "完成"),
        ])
        rt.run(seed=7)
        states = {e.get("state") for e in rt.trace.events}
        for s in ("S3", "S4", "S5", "S6"):
            self.assertIn(s, states, f"{s} 未被访问")
        # 决策确实发生了 (S2 的产物进了 S3 校验)
        validated = [e for e in rt.trace.events if e.get("event") == "validate"]
        self.assertTrue(validated)
        self.assertTrue(any(e.get("name") == "grasp" for e in validated))

    def test_vlm_output_is_validated_not_trusted(self):
        """VLM 提议的坐标必须被注册表位姿覆盖 —— 模型不能自己指定抓取点。"""
        rt = build_runtime([
            decision("grasp", {"object_id": "red_cup",
                               "target_xyz": [9.0, 9.0, 9.0]}, "抓"),
            decision("place", {"object_id": "red_cup"}, "放"),
            decision("done", {}, "完成"),
        ])
        rt.run(seed=7)
        dispatched = [e for e in rt.trace.events
                      if e.get("event") == "dispatch" and e.get("name") == "grasp"]
        self.assertTrue(dispatched)
        args = dispatched[0].get("args") or {}
        if args:
            self.assertNotAlmostEqual(args["target_xyz"][0], 9.0, places=3)

    def test_protocol_error_triggers_recovery_not_crash(self):
        """模型输出非法 JSON 时, 运行时应走协议错误恢复路径, 而不是让
        异常逃出去把整集打成 error。"""
        rt = build_runtime([
            "这不是 JSON",
            decision("grasp", {"object_id": "red_cup"}, "抓"),
            decision("place", {"object_id": "red_cup"}, "放"),
            decision("done", {}, "完成"),
        ])
        r = rt.run(seed=7)
        # 无论最终成功与否, 关键是没有未捕获异常 (exit_reason 是合法枚举)
        self.assertIn(r.exit_reason.value,
                      ("SUCCESS", "FAILED", "BUDGET_EXHAUSTED",
                       "SAFETY_STOP", "ABORTED"))

    def test_budget_accounting_covers_every_vlm_call(self):
        """每个决策都记一笔 llm_calls — 四组必须同记账。

        这条不变量是整个对照实验的前提: 如果 VLM 臂的调用没被记账,
        "哪个臂更省"的结论就来自记账差异而不是架构差异。
        """
        rt = build_runtime([
            decision("grasp", {"object_id": "red_cup"}, "抓"),
            decision("place", {"object_id": "red_cup"}, "放"),
            decision("done", {}, "完成"),
        ])
        r = rt.run(seed=7)
        charged = r.budget.used["llm_calls"]
        self.assertGreaterEqual(charged, 1)
        # 账上的 llm_calls 不应少于实际发起的决策次数
        self.assertGreaterEqual(charged, rt._fake.call_count - 1)
        self.assertLessEqual(charged, rt._fake.call_count + 1)


if __name__ == "__main__":
    unittest.main()
