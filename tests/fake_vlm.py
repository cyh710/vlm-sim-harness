"""离线的 VLM 夹具 — 让 VLM 通路可在无 API key / 无仿真器下验证。

用途:
  - 单测: 断言 prompt 构造、输出解析、识别映射、place 规范化、id 翻译、
    协议错误处理。
  - 集成: 把录制好的 VLM 响应回放进九状态运行时, 验证 "模型提议 -> 运行时
    放行 -> 执行 -> 核验" 这条链路在离线条件下是通的。

设计: 夹具只替换"模型返回什么文本"这一件事 — 请求怎么构造、响应怎么解析、
    判决怎么产生, 全部走真实代码路径。这样夹具测到的是真逻辑, 不是替身。
"""
from __future__ import annotations

import json
from typing import Callable

import numpy as np


class FakeVLM:
    """按调用序返回预设文本的假模型。

    responses: 依次返回的文本; 最后一���会被重复使用 (便于"每次都答同样
    的话"的场景)。
    calls: 每次调用收到的 (system, user_blocks) — 供断言 prompt 内容。
    """

    def __init__(self, responses: list[str] | str):
        if isinstance(responses, str):
            responses = [responses]
        self._responses = list(responses)
        self._idx = 0
        self.calls: list[tuple[str, list]] = []

    def __call__(self, messages: list) -> str:
        system = ""
        blocks: list = []
        for m in messages:
            if m.get("role") == "system":
                system = m["content"]
                continue
            c = m["content"]
            if isinstance(c, str):
                blocks.append({"type": "text", "text": c})
            else:
                blocks.extend(c)
        self.calls.append((system, blocks))
        r = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return r

    @property
    def call_count(self) -> int:
        return self._idx


def text_of(messages: list) -> str:
    """把一次请求的文本部分拼起来 — 断言 prompt 用。"""
    out = []
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            out.append(c)
        else:
            out.extend(p.get("text", "") for p in c if p.get("type") == "text")
    return "\n".join(out)


def image_blocks(messages: list) -> list:
    return [p for m in messages if not isinstance(m["content"], str)
            for p in m["content"] if p.get("type") == "image_url"]


def fake_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """一张假的相机帧 — 只需要形状正确 (H,W,3) uint8。"""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def decision(action_name: str, args: dict | None = None,
             reason: str = "test") -> str:
    """构造一个合法的决策响应 (VLM 的输出协议)。"""
    return json.dumps({"reason": reason,
                       "action": {"name": action_name,
                                  "args": args or {}}},
                      ensure_ascii=False)


def recognition(mapping: dict[str, str]) -> str:
    """构造一个 SoM 识别响应 — 协议是 {编号: 候选id}。"""
    return json.dumps(mapping, ensure_ascii=False)


def registry_table(objects: list[dict]) -> Callable[[], list[dict]]:
    """注册表快照提供者。"""
    return lambda: list(objects)


def camera_provider(frame: np.ndarray | None = None) -> Callable[[], np.ndarray]:
    f = frame if frame is not None else fake_frame()
    return lambda: f
