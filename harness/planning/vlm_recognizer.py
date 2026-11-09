"""VLM 闭集识别器 — 把实例注册表映射到任务资产描述 ("识别"环节).

感知层只出注册实例 (id/位姿, 无语义); 本模块用 Set-of-Mark (SoM) 双图协议
让 VLM 在**任务闭集** (configs/task_assets.json 的候选描述) 中做选择, 产出
{canonical_id: registered_id} 映射。

双图协议 (recognize_marked 是主入口):
  图1 = 场景原图 (真实外观/纹理)   图2 = SoM 渲染图 (实例色块 + 编号)
VLM 在渲染图上确认"编号 k 是哪一类", 参照原图的外观; **坐标永远不由 VLM
给出** — 代码按编号→注册 id 查注册表位姿。这条边界是本项目最关键的设计
决定之一: 让 VLM 自由输出坐标时实测 median 偏差 237mm, 15/17 超过 100mm。
"""
from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# numpy / PIL 只在真正编码图像时才需要, 所以惰性导入 — 框架层保持零第三方
# 依赖, clone 下来就能跑全部单测 (包括 VLM 层的), 不需要先装 pillow/numpy。


def load_task_assets(task: str, path: str | None = None) -> list[dict]:
    p = Path(path) if path else (Path(__file__).resolve().parents[2]
                                  / "configs" / "task_assets.json")
    return json.loads(p.read_text(encoding="utf-8")).get(task, {}).get("candidates", [])




def recognize_marked(endpoint: dict, annotated_rgb: "np.ndarray",
                     marker_map: dict[str, str], task: str,
                     timeout_s: int = 240,
                     raw_rgb: "np.ndarray | None" = None) -> dict[str, str]:
    """Set-of-Mark 闭集识别 (双图协议, 用户定稿 2026-11-05):
    第一张图 = 场景原图 (真实外观/纹理), 第二张图 = SAM 渲染图 (实例划分 +
    描边 id)。VLM 结合原图外观在渲染图上确认每个编号的类别; 坐标不由 VLM
    给出, 代码按编号→注册id 查注册表位姿。

    marker_map: {编号: registered_id} (adapter.annotated_overhead 生成).
    返回 {canonical_id: registered_id}。"""
    cands = load_task_assets(task)
    cand_text = "\n".join(f"- {c['id']}: {c['desc']}" for c in cands)
    marks = "\n".join(f"- 编号 {k} = 注册实例 {v}" for k, v in marker_map.items())
    if raw_rgb is not None:
        q = (f"第一张图是场景原图, 第二张图中每个物体实例用纯色高亮渲染, "
             f"实例上有白字黑描边的编号。\n"
             f"编号对照:\n{marks}\n\n"
             f"任务候选类别:\n{cand_text}\n\n"
             f"参考原图中物体的真实外观 (颜色/形状/材质), 对第二张图中每个编号, "
             f"从候选中选出最匹配的类别 (不确定用 unknown)。\n"
             f'输出严格 JSON (无其他文本): {{"<编号>": "<候选id或unknown>"}}')
        content = [{"type": "text", "text": q},
                   {"type": "image", "source": {"type": "base64",
                                                "media_type": "image/png",
                                                "data": _b64_png(raw_rgb)}},
                   {"type": "image", "source": {"type": "base64",
                                                "media_type": "image/png",
                                                "data": _b64_png(annotated_rgb)}}]
    else:
        q = (f"图中每个物体实例都用纯色高亮渲染, 实例上有白字黑描边的编号。\n"
             f"编号对照:\n{marks}\n\n"
             f"任务候选类别:\n{cand_text}\n\n"
             f"对每个编号, 从候选中选出最匹配的类别 (不确定用 unknown)。\n"
             f'输出严格 JSON (无其他文本): {{"<编号>": "<候选id或unknown>"}}')
        content = [{"type": "text", "text": q},
                   {"type": "image", "source": {"type": "base64",
                                                "media_type": "image/png",
                                                "data": _b64_png(annotated_rgb)}}]
    body = json.dumps({
        # max_tokens 65536: 多标记密集场景下思考 token 巨大 (32768 仍被吃光, 2026-11-05)
        "model": endpoint["model"], "max_tokens": 65536, "temperature": 0,
        "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(f"{endpoint['base_url']}/v1/messages", data=body,
                                 method="POST",
                                 headers={"Content-Type": "application/json",
                                          "x-api-key": endpoint["api_key"],
                                          "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        d = json.loads(r.read().decode())
    text = "".join(b.get("text", "") for b in d.get("content", [])
                   if b.get("type") == "text")
    import os as _os, sys as _sys
    if _os.environ.get("EBHARNESS_VLM_DEBUG"):
        print(f"[SoM] raw={text[:300]} stop={d.get('stop_reason')}", file=_sys.stderr)
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return {}
    try:
        raw = json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return {}
    out: dict[str, str] = {}
    cand_ids = {c["id"] for c in cands}
    # 唯一性: 一个规范 id 只能绑定一个注册实例, 一个注册实例只能被绑定一次。
    # 不检查会出结构性错误: VLM 把两个编号都判成 rubbish 时, 世界里的
    # tomato 实例会被改名成 rubbish — 目标指错, 而日志看起来完全正常。
    # 实测出现过 {'tomato': 'object_o05', 'bin': 'object_o04',
    # 'rubbish': 'object_o06'} 与 {'tomato': 'rubbish'} 的污染映射。
    taken: set[str] = set()
    for marker, cand in sorted(raw.items(), key=lambda kv: str(kv[0])):
        if cand not in cand_ids or str(marker) not in marker_map:
            continue
        rid = marker_map[str(marker)]
        if cand in out or rid in taken:
            continue          # 冲突项丢弃 (先到先得, 编号序稳定)
        out[cand] = rid
        taken.add(rid)
    return out


def _b64_png(rgb) -> str:
    """把 RGB 数组编码成 base64 PNG。

    numpy / PIL 在这里惰性导入: 只有真的要向模型发图时才需要它们,
    框架层的其余部分保持零第三方依赖。
    """
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
