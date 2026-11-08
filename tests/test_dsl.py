"""DSL unit tests: whitelist, structure limits, evaluation semantics."""
import unittest

import harness.spec.dsl as dsl
from harness.spec.dsl import DSLError, evaluate, parse, summarize
from harness.spec.types import ObjInstance, Pose, WorldState


def world(version=1, cup=(0.10, -0.05, 0.05), cup2=None, gripper=1.0,
          vis=1.0, joints=None):
    objs = [ObjInstance(id="red_cup", label="red cup", view="front",
                        pose=Pose(*cup), sigma_m=0.004, size=(0.07, 0.07, 0.10),
                        visible_ratio=vis, quality="ok", source="cad",
                        last_seen_t=0.0)]
    if cup2:
        objs.append(ObjInstance(id="bin", label="bin", view="front",
                                pose=Pose(*cup2), sigma_m=0.004,
                                size=(0.2, 0.2, 0.12), visible_ratio=1.0,
                                quality="ok", source="cad", last_seen_t=0.0))
    return WorldState(version=version, timestamp=0.0, ee_pose=Pose(0.3, 0, 0.25),
                      gripper_open=gripper, joints=joints or {},
                      objects=tuple(objs))


class TestDslStructure(unittest.TestCase):
    def test_valid_parse(self):
        ast = {"op": "all_of", "children": [
            {"op": "holding", "args": {"object_id": "red_cup"}},
            {"op": "displacement_min", "args": {"object_id": "red_cup",
                                                "distance_m": 0.03}}]}
        parse(ast)
        self.assertEqual(dsl.count_predicates(ast), 2)

    def test_unknown_op_rejected(self):
        with self.assertRaises(DSLError):
            parse({"op": "drop_table", "args": {}})

    def test_missing_arg_rejected(self):
        with self.assertRaises(DSLError):
            parse({"op": "holding", "args": {}})

    def test_depth_limit(self):
        node = {"op": "holding", "args": {"object_id": "x"}}
        for _ in range(3):
            node = {"op": "all_of", "children": [node]}
        with self.assertRaises(DSLError):
            parse(node)

    def test_predicate_count_limit(self):
        leaf = lambda: {"op": "holding", "args": {"object_id": "x"}}
        node = {"op": "any_of", "children": [leaf() for _ in range(17)]}
        with self.assertRaises(DSLError):
            parse(node)

    def test_no_eval_path(self):
        # 显式 utf-8: 源码含中文注释, 用平台默认编码读会抛 UnicodeDecodeError
        # (Windows 中文默认 GBK) — 静态检查本身不该因编码而失败。
        src = open(dsl.__file__, encoding="utf-8").read()
        self.assertNotIn("eval(", src.replace("evaluate(", "").replace("_eval_leaf", ""))
        self.assertNotIn("exec(", src)


class TestDslEvaluation(unittest.TestCase):
    def test_holding_true_and_counter(self):
        ast = {"op": "holding", "args": {"object_id": "red_cup"}}
        # RLBench attach-carry signature: cup near ee AND gripper open (>=0.5)
        held = world(cup=(0.30, 0.0, 0.16), gripper=1.0)
        checks, ok = evaluate(ast, world(), held)
        self.assertTrue(ok)
        # empty-close signature: cup unmoved AND gripper shut (0.0) -> counter
        empty = world(cup=(0.10, -0.05, 0.05), gripper=0.0)
        checks, ok = evaluate(ast, world(), empty)
        self.assertIs(ok, False)
        self.assertTrue(checks[0].counter_evidence)

    def test_holding_vision_primary_counter(self):
        # M2 calibration: vision distance is the primary signal; an object far
        # from the ee is ALWAYS explicit counter-evidence (NOT_HOLDING),
        # regardless of the gripper reading (kinematic attach reads open)
        ast = {"op": "holding", "args": {"object_id": "red_cup"}}
        far_open = world(cup=(0.10, -0.05, 0.05), gripper=1.0)
        checks, ok = evaluate(ast, world(), far_open)
        self.assertIs(ok, False)
        self.assertTrue(checks[0].counter_evidence)
        far_shut = world(cup=(0.10, -0.05, 0.05), gripper=0.0)
        checks, ok = evaluate(ast, world(), far_shut)
        self.assertIs(ok, False)

    def test_displacement_min(self):
        ast = {"op": "displacement_min", "args": {"object_id": "red_cup",
                                                  "distance_m": 0.03}}
        _, ok = evaluate(ast, world(), world(cup=(0.10, -0.05, 0.05)))
        self.assertIs(ok, False)   # no displacement -> fail, counter if tiny
        _, ok = evaluate(ast, world(), world(cup=(0.30, -0.05, 0.20)))
        self.assertTrue(ok)

    def test_contained_and_gripper(self):
        ast = {"op": "all_of", "children": [
            {"op": "contained", "args": {"object_id": "red_cup",
                                         "container_id": "bin"}},
            {"op": "gripper_open", "args": {}}]}
        placed = world(cup=(-0.10, 0.05, 0.08), cup2=(-0.10, 0.05, 0.02),
                       gripper=1.0)
        _, ok = evaluate(ast, world(), placed)
        self.assertTrue(ok)
        held_in_bin = world(cup=(-0.10, 0.05, 0.08), cup2=(-0.10, 0.05, 0.02),
                            gripper=0.3)
        checks, ok = evaluate(ast, world(), held_in_bin)
        self.assertIs(ok, False)     # in region but still gripped -> counter
        self.assertTrue(checks[0].counter_evidence)

    def test_invisible_object_unknown(self):
        """被遮挡的物体: 判据由夹爪开度主导 (2026-11-06 语义)。

        夹爪全开 (1.0) = 明确没夹住 -> NOT_HOLDING 反证 (False, 不是 None);
        开度落在中间带 = 证据不足 -> INCONCLUSIVE (None);
        开度在"夹住带"内 = 判定为持有 (True)。

        这三种都要分开: 把"看不见"一律当成"不知道"会让握持成功被记成
        inconclusive, 反而把失败记成模糊。
        """
        ast = {"op": "holding", "args": {"object_id": "red_cup"}}
        # 全开夹爪 = 明确空爪 -> 反证
        checks, ok = evaluate(ast, world(), world(vis=0.0, gripper=1.0))
        self.assertIs(ok, False)
        self.assertTrue(checks[0].counter_evidence)
        self.assertEqual(checks[0].reason_code, "NOT_HOLDING")
        # 闭合到底 = 明确夹空 -> 反证
        checks, ok = evaluate(ast, world(), world(vis=0.0, gripper=0.0))
        self.assertIs(ok, False)
        self.assertTrue(checks[0].counter_evidence)
        # 中间带 = 证据不足 -> 未知 (None)
        _, ok = evaluate(ast, world(), world(vis=0.0, gripper=0.2))
        self.assertIsNone(ok)
        # 夹住带 = 判定持有
        _, ok = evaluate(ast, world(), world(vis=0.0, gripper=0.5))
        self.assertIs(ok, True)

    def test_summarize(self):
        s = summarize({"op": "holding", "args": {"object_id": "red_cup"}})
        self.assertEqual(s, "holding(object_id=red_cup)")


if __name__ == "__main__":
    unittest.main()
