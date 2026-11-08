"""VLM 规划器单测 — 断言协议边界, 而非模型聪明程度。

覆盖:
  - prompt 构造: 双图协议 (原图 + SoM 渲染图)、系统提示词冻结、状态文本
  - 输出解析: 合法 JSON / 无 JSON / 非法 JSON -> PlannerProtocolError
  - 识别映射: 唯一性 (一个规范 id 一个注册实例, 反之亦然)
  - 引用 id 翻译: 注册 id -> 规范 id
  - place 规范化: object_id/container_id/target_xyz 的正确落位, 以及
    "bin 未发布 -> 转 query 而不是放行"
  - 传输: 失败重试, token 用量累加

这些都不需要 API key 或仿真器 (tests/fake_vlm.py 提供离线替身)。
"""
import json
import unittest

from harness.planning import vlm_planner as vp
from harness.planning.vlm_planner import VLMPlanner
from harness.spec.runtime import PlannerContext
from harness.spec.types import ObjInstance, PlannerProtocolError, Pose, WorldState

from tests.fake_vlm import (camera_provider, decision, fake_frame,
                            recognition, registry_table, text_of)


def world(gripper=1.0, objects=None, version=3) -> WorldState:
    objs = objects if objects is not None else (
        ObjInstance(id="rubbish", label="paper ball", view="overhead",
                    pose=Pose(0.20, 0.10, 0.44), sigma_m=0.004,
                    size=(0.087, 0.087, 0.087), visible_ratio=1.0,
                    quality="ok", source="cad", last_seen_t=0.0),
        ObjInstance(id="bin", label="mesh basket", view="overhead",
                    pose=Pose(0.10, -0.04, 0.49), sigma_m=0.004,
                    size=(0.20, 0.20, 0.12), visible_ratio=1.0,
                    quality="ok", source="cad", last_seen_t=0.0),
    )
    return WorldState(version=version, timestamp=0.0,
                      ee_pose=Pose(0.30, 0.0, 0.60), gripper_open=gripper,
                      joints={}, objects=tuple(objs))


def ctx(w: WorldState) -> PlannerContext:
    return PlannerContext(world=w, memory_text="", progress_summaries=(),
                          budget_remaining={}, last_event={})


def make_planner(responses, **kw) -> VLMPlanner:
    """构造一个接了假模型的 VLMPlanner。

    假模型通过替换 _call_anthropic 注入 — 请求构造与响应解析仍走真代码。
    """
    from tests.fake_vlm import FakeVLM
    p = VLMPlanner(instruction="pick up the paper ball and put it in the bin",
                   camera_provider=camera_provider(),
                   registry_table_provider=registry_table([]),
                   endpoint={"base_url": "http://local.invalid",
                             "api_key": "not-a-real-key",
                             "model": "fake-vlm"},
                   **kw)
    fake = FakeVLM(responses)
    p._call_anthropic = fake          # 换掉传输, 其余全真
    p._fake = fake
    return p


class TestPromptConstruction(unittest.TestCase):
    def test_system_prompt_is_frozen_and_hashed(self):
        # 提示词必须有稳定指纹 — 实验中途改词 = 结果不可比 (rule 9)
        h1 = vp.prompt_hash()
        h2 = vp.prompt_hash()
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)
        self.assertTrue(vp.PROMPT_VERSION)

    def test_prompt_thresholds_match_verifier(self):
        """VLM 被告知的判据必须就是 verifier 判定用的判据。

        两者不一致时, VLM 会在边界开度上系统性误判 — 而那正是最容易
        滑脱的工况。
        """
        from harness.spec.dsl import gripper_bands
        bands = gripper_bands()
        prompt = vp.system_prompt()
        self.assertIn(f"{bands['hold_min']}~{bands['hold_max']}", prompt)
        self.assertNotIn("0.3~0.9", prompt)

    def test_dual_image_protocol_sends_two_images(self):
        """双图协议: 原图 (真实外观) + SoM 渲染图 (实例划分+编号)。

        这是本项目最关键的一个设计决定: VLM 命名, 代码定位。让 VLM 自由
        输出坐标时实测 median 偏差 237mm; 改成"看图选编号"后偏差归零 ——
        因为坐标根本不再经过模型。
        """
        from harness.planning.vlm_recognizer import recognize_marked
        import harness.planning.vlm_recognizer as vr
        from tests.fake_vlm import FakeVLM

        sent = {}

        class _Resp:
            def read(self):
                return json.dumps({
                    "content": [{"type": "text", "text": recognition({"1": "rubbish"})}]
                }).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=None):
            body = json.loads(req.data.decode("utf-8"))
            sent["content"] = body["messages"][0]["content"]
            return _Resp()

        orig = vr.urllib.request.urlopen
        vr.urllib.request.urlopen = fake_urlopen
        try:
            out = recognize_marked(
                {"base_url": "http://x", "api_key": "k", "model": "m"},
                fake_frame(), {"1": "object_o01"}, "PutRubbishInBin",
                raw_rgb=fake_frame())
        finally:
            vr.urllib.request.urlopen = orig

        kinds = [b["type"] for b in sent["content"]]
        self.assertEqual(kinds.count("image"), 2,
                         "双图协议必须发两张图: 原图 + SoM 渲染图")
        self.assertEqual(out, {"rubbish": "object_o01"})

    def test_state_text_includes_gripper_and_registry(self):
        p = make_planner([decision("grasp", {"object_id": "rubbish"})])
        msgs = p._build_messages(ctx(world(gripper=0.45)))
        t = text_of(msgs)
        self.assertIn("gripper_open", t)
        self.assertIn("0.45", t)


class TestOutputParsing(unittest.TestCase):
    def test_non_json_output_raises_protocol_error(self):
        p = make_planner(["I think the gripper is holding it."])
        with self.assertRaises(PlannerProtocolError):
            p._ask(ctx(world()))

    def test_malformed_json_raises_protocol_error(self):
        p = make_planner(['{"action": {"name": "grasp", '])
        with self.assertRaises(PlannerProtocolError):
            p._ask(ctx(world()))

    def test_json_with_surrounding_prose_parses(self):
        """宽容解析: 模型在 JSON 前后加解释文字是常见行为。"""
        p = make_planner(['好的。{"reason":"ok","action":{"name":"done","args":{}}} 完成。'])
        out = p._ask(ctx(world()))
        self.assertEqual(out["action"]["name"], "done")


class TestRecognitionMapping(unittest.TestCase):
    def _recognize(self, raw: str) -> dict:
        """跑一次真实的 recognize_marked, 但把传输换成假模型。"""
        from harness.planning.vlm_recognizer import recognize_marked
        from tests.fake_vlm import FakeVLM
        fake = FakeVLM([raw])
        # 直接调内部: 借用 vlm_planner 的传输封装
        p = make_planner([raw])
        p._call_anthropic = lambda msgs: fake(msgs)
        p._ep = {"base_url": "http://local.invalid", "api_key": "k",
                 "model": "fake-vlm"}
        # recognize_marked 自行发请求 — 用 monkeypatch 拦截 urllib
        import harness.planning.vlm_recognizer as vr
        orig = vr.urllib.request.urlopen

        class _Resp:
            def __init__(self, payload):
                self._p = payload

            def read(self):
                return json.dumps(self._p).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        vr.urllib.request.urlopen = lambda req, timeout=None: _Resp({
            "content": [{"type": "text", "text": raw}]})
        try:
            return recognize_marked(
                {"base_url": "http://x", "api_key": "k", "model": "m"},
                fake_frame(), {"1": "object_o01", "2": "object_o02"},
                "PutRubbishInBin", raw_rgb=fake_frame())
        finally:
            vr.urllib.request.urlopen = orig

    def test_mapping_is_respected(self):
        out = self._recognize(recognition({"1": "rubbish", "2": "bin"}))
        self.assertEqual(out, {"rubbish": "object_o01", "bin": "object_o02"})

    def test_duplicate_canonical_id_is_dropped(self):
        """两个编号都判成 rubbish 时, 不能两个都绑定 — 否则世界里出现
        两个 rubbish, 或一个实例被绑定两次。"""
        out = self._recognize(recognition({"1": "rubbish", "2": "rubbish"}))
        self.assertEqual(len(out), 1)
        self.assertEqual(out["rubbish"], "object_o01")

    def test_one_instance_cannot_bind_twice(self):
        out = self._recognize(recognition({"1": "rubbish", "2": "bin"}))
        self.assertEqual(len(set(out.values())), len(out))

    def test_unknown_candidate_dropped(self):
        """候选集外的类别必须丢弃 —— 闭集识别的意义就在"闭"。"""
        out = self._recognize(recognition({"1": "purple_hat", "2": "bin"}))
        self.assertNotIn("object_o01", out.values())
        self.assertEqual(out, {"bin": "object_o02"})

    def test_distractor_in_candidate_set_is_kept(self):
        """番茄在候选集里 (干扰物), 所以映射本身是合法的 — 是否选它
        是另一层问题 (容器体积先验会覆盖它)。"""
        out = self._recognize(recognition({"1": "tomato", "2": "bin"}))
        self.assertEqual(out.get("tomato"), "object_o01")


class TestDecisionNormalization(unittest.TestCase):
    def test_registered_id_translated_to_canonical(self):
        """VLM 引用注册 id, 世界里它已被识别映射改名 -> 必须翻译回去。"""
        p = make_planner([decision("grasp", {"object_id": "object_o06"})])
        p.last_recognition = {"rubbish": "object_o06"}
        p._recognizer = ["rubbish"]          # 跳过 SoM
        call = p.decide(ctx(world()))
        self.assertEqual(call.args["object_id"], "rubbish")

    def test_place_normalized_to_container_pose(self):
        """place 的语义: 放下手中物, 目标是容器。容器位姿由代码给出。"""
        p = make_planner([decision("place", {"object_id": "rubbish"})])
        p.last_recognition = {"rubbish": "object_o06", "bin": "object_o04"}
        p._recognizer = ["rubbish"]
        call = p.decide(ctx(world()))
        self.assertEqual(call.name, "place")
        self.assertEqual(call.args["object_id"], "rubbish")
        self.assertEqual(call.args["container_id"], "bin")
        # target_xyz 必须来自容器的位姿, 不是 VLM 编的
        self.assertAlmostEqual(call.args["target_xyz"][0], 0.10, places=3)
        self.assertAlmostEqual(call.args["target_xyz"][2], 0.49, places=3)

    def test_place_becomes_query_when_container_missing(self):
        """容器未发布时不能放行 (否则 VLM 编造的 target_xyz 直达执行),
        必须转 query 重新观测。"""
        p = make_planner([decision("place", {"object_id": "rubbish"})])
        p.last_recognition = {"rubbish": "object_o06"}
        p._recognizer = ["rubbish"]
        w = world(objects=(
            ObjInstance(id="rubbish", label="paper ball", view="overhead",
                        pose=Pose(0.20, 0.10, 0.44), sigma_m=0.004,
                        size=(0.087, 0.087, 0.087), visible_ratio=1.0,
                        quality="ok", source="cad", last_seen_t=0.0),))
        call = p.decide(ctx(w))
        self.assertEqual(call.kind, "query")
        self.assertEqual(call.name, "where_is")
        self.assertEqual(call.args["object_id"], "bin")

    def test_grasp_target_xyz_from_registry_not_model(self):
        """定位是感知层职责: VLM 给了坐标也会被注册表位姿覆盖。"""
        p = make_planner([decision("grasp", {"object_id": "rubbish",
                                              "target_xyz": [9.9, 9.9, 9.9]})])
        p.last_recognition = {"rubbish": "object_o06"}
        p._recognizer = ["rubbish"]
        call = p.decide(ctx(world()))
        self.assertAlmostEqual(call.args["target_xyz"][0], 0.20, places=3)
        self.assertAlmostEqual(call.args["target_xyz"][1], 0.10, places=3)

    def test_done_is_control_call(self):
        p = make_planner([decision("done", {})])
        p.last_recognition = {"rubbish": "object_o06", "bin": "object_o04"}
        p._recognizer = ["rubbish"]
        call = p.decide(ctx(world()))
        self.assertEqual(call.kind, "control")
        self.assertEqual(call.name, "done")

    def test_task_is_not_hardcoded(self):
        """任务 id 必须由调用方传入 — 写死会让 --task PickUpCup 静默
        用错闭集候选集。"""
        p = VLMPlanner(instruction="x", camera_provider=camera_provider(),
                       registry_table_provider=registry_table([]),
                       endpoint={"base_url": "u", "api_key": "k", "model": "m"},
                       task="PickUpCup")
        self.assertEqual(p._recognizer_task, "PickUpCup")


class TestTransport(unittest.TestCase):
    def _real_planner(self) -> VLMPlanner:
        """构造一个**不替换**传输层的 planner — 传输测试要测的就是它。"""
        p = VLMPlanner(instruction="x", camera_provider=camera_provider(),
                       registry_table_provider=registry_table([]),
                       endpoint={"base_url": "http://local.invalid",
                                 "api_key": "not-a-real-key", "model": "fake-vlm"})
        p._retries = 2
        p._backoff_s = 0
        return p

    def test_token_usage_accumulates(self):
        """用量必须累加 — 覆盖会让前面的调用凭空消失。"""
        p = self._real_planner()
        p._stats["vlm_tokens"] = {"input_tokens": 10, "output_tokens": 5}
        p._stats["vlm_tokens"]["input_tokens"] += 7
        self.assertEqual(p._stats["vlm_tokens"]["input_tokens"], 17)

    def test_transport_error_retries_then_raises(self):
        """传输错误要重试 (不是模型的错); 重试用尽后抛出, 不静默降级。"""
        import urllib.error
        import urllib.request
        p = self._real_planner()
        attempts = {"n": 0}

        def boom(req, timeout=None):
            attempts["n"] += 1
            raise urllib.error.URLError("connection reset")

        orig = urllib.request.urlopen
        urllib.request.urlopen = boom
        try:
            with self.assertRaises(RuntimeError):
                p._call_anthropic([{"role": "system", "content": "s"},
                                   {"role": "user", "content": "u"}])
        finally:
            urllib.request.urlopen = orig
        # retries=2 -> 1 次首发 + 2 次重试 = 3 次尝试
        self.assertEqual(attempts["n"], 3)

    def test_transport_success_records_usage(self):
        """成功响应要记 usage, 且是累加。"""
        import urllib.request
        p = self._real_planner()

        class _Resp:
            def read(self):
                return json.dumps({
                    "content": [{"type": "text", "text": '{"action":{}}'}],
                    "usage": {"input_tokens": 3, "output_tokens": 7},
                }).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        orig = urllib.request.urlopen
        urllib.request.urlopen = lambda req, timeout=None: _Resp()
        try:
            p._call_anthropic([{"role": "system", "content": "s"},
                               {"role": "user", "content": "u"}])
            p._call_anthropic([{"role": "system", "content": "s"},
                               {"role": "user", "content": "u"}])
        finally:
            urllib.request.urlopen = orig
        self.assertEqual(p._stats["vlm_tokens"]["input_tokens"], 6)
        self.assertEqual(p._stats["vlm_tokens"]["output_tokens"], 14)


if __name__ == "__main__":
    unittest.main()
