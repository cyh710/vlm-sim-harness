"""RLBench 适配器的契约测试 — 不需要装 RLBench。

这个仓的适配器是**仿真后端**, 而 431 集实验就是跑在它上面的。所以它的接口
必须被锁住: 拼错了参数、漏了方法、或者悄悄退回真值回退, 都是那种"跑了
300 集才发现"的错误。

这些测试只检查**接线**, 不启动仿真 (那需要 CoppeliaSim + RLBench)。
"""
import ast
import unittest
from pathlib import Path

from harness.adapters.rlbench_adapter import RlbenchAdapter

ADAPTER = Path(__file__).resolve().parents[1] / "harness" / "adapters" / "rlbench_adapter.py"


class TestAdapterSurface(unittest.TestCase):
    def test_implements_the_adapter_protocol(self):
        """运行时用到的四个方法必须都在。"""
        for name in ("reset", "observe", "execute", "shutdown"):
            self.assertTrue(callable(getattr(RlbenchAdapter, name, None)),
                            f"missing adapter method: {name}")

    def test_exposes_the_judge_bridge_handle(self):
        """判卷桥要拿到 task handle 才能读 success() —— 这是 AGENTS 规则 1
        允许的唯一一条越界路径。"""
        self.assertTrue(callable(getattr(RlbenchAdapter, "judge_bridge_task", None)))

    def test_supports_the_fault_injection_switches(self):
        """F1/F2 故障注入的开关必须在签名里, 否则对照实验无法复现。"""
        import inspect
        params = inspect.signature(RlbenchAdapter.__init__).parameters
        for p in ("fault_f1_empty_first", "fault_f2_timeout_receipt"):
            self.assertIn(p, params)


class TestPerceptionMode(unittest.TestCase):
    def test_sensor_eval_is_refused_not_silently_downgraded(self):
        """sensor_eval 需要感知栈 (不在本仓内)。必须**报错**, 绝不能悄悄
        退回真值 —— 那会让"真实感知下的结论"变成"真值下的结论", 而日志
        看起来完全正常。"""
        with self.assertRaises(NotImplementedError) as cm:
            RlbenchAdapter(perception_mode="sensor_eval")
        self.assertIn("perception", str(cm.exception).lower())

    def test_gt_debug_is_the_default(self):
        self.assertEqual(
            RlbenchAdapter.__init__.__defaults__[-1], "gt_debug")


class TestNoForeignCode(unittest.TestCase):
    def test_does_not_import_a_perception_stack(self):
        """适配器不能依赖不在本仓内的感知实现。

        依赖树一旦伸到仓外, "clone 下来就能跑"这句话就不成立了 —— 而这正是
        这个仓对读者承诺的东西。
        """
        src = ADAPTER.read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imported.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for forbidden in ("harness.perception", "segment_anything", "torch"):
            self.assertNotIn(forbidden, imported,
                             f"adapter must not depend on {forbidden}")


if __name__ == "__main__":
    unittest.main()
