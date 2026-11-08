"""VLMPlanner — 识别与决策合一的规划器 (2026-11-04 架构定稿).

分工 (用户定稿): SAM 只做分割/定位/ID 注册 (注册表, 不做规范绑定);
VLM 看相机图像 + 注册表 → 输出 ①识别映射 (rubbish/bin = 哪个注册 id)
②抓取/放置决策 (引用注册 id 的位姿)。执行/judge 链路零改动。

输入: overhead 相机帧 (base64) + 注册表 JSON + 任务指令 + 九状态上下文摘要。
输出协议 (JSON): {"recognition": {"rubbish": id, "bin": id},
                   "action": {"name": "grasp|place|done", "args": {...}}}
识别映射经 runner 写入 adapter.set_recognition_map(); 世界随之以规范 id 发布,
goal DSL / validator / verifier 不变。

传输: OpenAI 兼容 chat/completions (configs/planner_model.json), 多模态 content 数组,
密钥一律走环境变量 (AGENTS/旧仓库泄漏教训)。温度 0; 传输重试沿用 llm_client。
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from harness.planning.planner import Planner  # noqa: E402
from harness.planning.vlm_recognizer import load_task_assets  # noqa: E402
from harness.spec.runtime import PlannerContext  # noqa: E402
from harness.spec.types import PlannerProtocolError, ToolCall  # noqa: E402

PROMPT_VERSION = "vlm-1"

# 提示词模板 — 阈值位用 {hold_min}/{hold_max}/{empty_high}/{empty_low}
# 占位, 运行时从 configs/gripper_bands.json 填充。这样 VLM 被告知的判据
# 与 verifier 判定用的判据必然同源 (AGENTS rule 2 + rule 9)。
PROMPT_TEMPLATE = """你是桌面抓取任务机器人 (UR5 + 平行夹爪) 的决策模块。

## 任务与唯一正道 (ProgPrompt 式状态机)
把纸团 (rubbish) 抓起放进蓝色网状垃圾桶 (bin):
  [S0 空爪] -> grasp rubbish -> [S1 持住] -> place -> [S2 已放入] -> done
每一步只能处于一个状态; 判错状态是唯一不可接受的错误。

## 状态判定规则 (每步必须先做)
1. 读"最近执行": grasp 成功 (REACHED 且未报 NOT_HOLDING) -> 你处于 S1。
2. 交叉验证 gripper_open: {hold_min}~{hold_max} = 夹着物体 (S1); ~1.0 = 空爪 (S0);
   ~0.0 = 闭合到底没夹到 (S0, 夹空)。
3. 两信号冲突时以最近执行结果为准。
判定为 S1 后的唯一正确动作是 place; 不要重复 grasp 一个已经抓着的物体。

## 恢复策略 (Inner Monologue: 用执行反馈修正下一动作)
- 夹空/NOT_HOLDING -> 再 grasp, 带 z_offset_mm (上次夹空就下探: {retry_lo}~{retry_hi},
  系统限幅 ±{z_clamp})。同一目标连续失败 2 次后, 第三次换 ±10 步进微调。
- grasp 被拒/目标不存在 -> 先输出 place 之前的 query? 不需要, 直接重试
  grasp (识别已由感知层保证注册表正确)。
- place 后 -> done (judge 由系统判定, 你不判断成功与否)。

## 动作 API (固定技能集, SayCan 式: 你只做语义选择, 定位/限幅由系统负责)
- grasp: {"object_id": "rubbish", "z_offset_mm": <可选整数, 默认 0>}
  (定位自动取 SAM 注册位姿; z_offset_mm 是相对高度修正)
- place: {"object_id": "rubbish"} (放下手中物; 系统自动投放到 bin 口上方,
  不要输出 target_xyz)
- done: {} (只在 place 已执行后输出)

## 输出 (严格 JSON, 无其他文本; ReAct 式先一句推理再动作)
{"reason": "<<=40字: 当前状态判定依据>",
 "action": {"name": "grasp|place|done", "args": {...}}}

## 示例
例A (最近执行: grasp REACHED, gripper_open=0.6):
{"reason": "grasp 已执行且夹爪半闭, 处于持住状态", "action": {"name": "place", "args": {"object_id": "rubbish"}}}
例B (最近执行: grasp REACHED, gripper_open=0.02):
{"reason": "夹爪闭合到底, 夹空", "action": {"name": "grasp", "args": {"object_id": "rubbish", "z_offset_mm": -15}}}
例C (最近执行: place REACHED):
{"reason": "已投放", "action": {"name": "done", "args": {}}}"""


def system_prompt() -> str:
    """填充后的实际提示词。数值全部来自 configs/gripper_bands.json。"""
    import json as _json
    from pathlib import Path as _Path
    p = _Path(__file__).resolve().parents[2] / "configs" / "gripper_bands.json"
    try:
        b = _json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        b = {"hold_min": 0.25, "hold_max": 0.95, "empty_high": 0.95,
             "empty_low": 0.05, "z_offset_mm_retry": [-25, -10],
             "z_offset_mm_clamp": 50}
    retry = b.get("z_offset_mm_retry") or [-25, -10]
    # 用 str.replace 而非 str.format: 提示词正文里有大量 JSON 花括号
    # ({"reason": ..., "action": {...}}), format 会把它们当占位符解析。
    out = PROMPT_TEMPLATE
    for key, val in (("{hold_min}", b["hold_min"]), ("{hold_max}", b["hold_max"]),
                     ("{empty_high}", b["empty_high"]),
                     ("{empty_low}", b["empty_low"]),
                     ("{retry_lo}", retry[0]), ("{retry_hi}", retry[1]),
                     ("{z_clamp}", b.get("z_offset_mm_clamp", 50))):
        out = out.replace(key, str(val))
    return out


def prompt_hash() -> str:
    """提示词指纹 — 写进每条结果行。实验中途改词 = 结果不可比,
    指纹让这件事可被检出 (AGENTS rule 9)。"""
    import hashlib
    return hashlib.sha256(system_prompt().encode("utf-8")).hexdigest()


def _find_contained(node) -> tuple[str, str] | None:
    """在目标 DSL 里找出 contained(object_id, container_id)。

    目标 DSL 是受信的 (Registry 构建), 不含模型输出 —— 所以从这里取
    "放什么/放到哪" 是安全的。
    """
    if not isinstance(node, dict):
        return None
    if node.get("op") == "contained":
        args = node.get("args") or {}
        oid = args.get("object_id")
        cid = args.get("container_id")
        if oid and cid:
            return (str(oid), str(cid))
        return None
    for child in node.get("children") or []:
        got = _find_contained(child)
        if got:
            return got
    return None


def load_endpoint(cfg_path: Optional[str] = None) -> dict:
    """端点凭证 — 全部走环境变量, 密钥永不入库。

    VLM_BASE_URL / VLM_API_KEY / VLM_MODEL 是本项目约定的名字;
    也接受 ANTHROPIC_* 与 DEEPSEEK_* 作为别名, 方便对接不同网关。

    传输用 Anthropic Messages 协议 (多模态 content 块: text + image)。
    任何 OpenAI 兼容网关只要接受 /v1/messages 形状也能接。
    """
    env = os.environ

    def _pick(*names, default=None):
        for n in names:
            v = env.get(n)
            if v:
                return v
        return default

    base = _pick("VLM_BASE_URL", "ANTHROPIC_BASE_URL", "DEEPSEEK_BASE_URL")
    key = _pick("VLM_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY")
    model = _pick("VLM_MODEL", "ANTHROPIC_MODEL", "DEEPSEEK_MODEL",
                  default="claude-sonnet-5")
    if not base or not key:
        raise RuntimeError(
            "missing endpoint: set VLM_BASE_URL and VLM_API_KEY "
            "(aliases: ANTHROPIC_*/DEEPSEEK_*)")
    return {"base_url": base.rstrip("/"), "api_key": key, "model": model}


class VLMPlanner(Planner):
    """识别 + 决策合一。

    识别映射暴露在 last_recognition, 由 runner 写入 adapter (planner 不持
    adapter 引用, 层间分离)。传输 = Anthropic Messages 协议 (多模态 content
    块: text + image)。
    """

    def __init__(self, instruction: str,
                 camera_provider, registry_table_provider,
                 give_up_after: int = 2, stats: Optional[dict] = None,
                 recognition_sink=None, endpoint: Optional[dict] = None,
                 recognizer_candidates: Optional[list] = None,
                 project_fn=None, recognition_frame_provider=None,
                 raw_frame_provider=None, execution_provider=None,
                 task: str = "PutRubbishInBin",
                 goal: dict | None = None):
        self._ep = endpoint or load_endpoint()
        # 受信的目标 DSL — place 的 (物体, 容器) 从这里取, 不写死字符串。
        self._goal = goal or {}
        # 执行闭环 (2026-11-05): 最近一次工具执行结果 (status/aborted_at) 回传
        # 给 VLM — 未持住时 Agent 可带 z_offset_mm 参数化重试 (adapter clamp)
        self._execution_provider = execution_provider
        self._recognition_frame_provider = recognition_frame_provider
        # 双图协议 (2026-11-05 定稿): 原图 (真实外观) + SAM 渲染图 (实例划分
        # +id)。原图须与渲染图同帧 — runner 接线 adapter.last_overhead_frame
        # (annotated_overhead 渲染的就是这一帧)。
        self._raw_frame_provider = raw_frame_provider
        self._instruction = instruction
        self._camera_provider = camera_provider          # () -> rgb uint8 (H,W,3)
        self._registry_provider = registry_table_provider  # () -> list[dict]
        self.give_up_after = give_up_after
        self._stats = stats if stats is not None else {}
        self._recognition_sink = recognition_sink    # runner 接线: adapter.set_recognition_map
        # 闭集候选来自任务资产描述 (configs/task_assets.json); 任务 id 由调用方
        # 传入 — 不写死, 否则 --task PickUpCup 会静默用错候选集。
        self._recognizer_task = task
        self._recognizer = recognizer_candidates    # 兼容旧调用 (未用则忽略)
        self._project_fn = project_fn               # 世界坐标→像素 (识别裁图用)
        # 传输参数 (同 llm_client 的口径): 思考 token 计入输出预算, 所以
        # max_tokens 必须给足 (M4 教训: 4096 会出现 finish=length + 空 content)
        self._max_tokens = int(self._ep.get("max_tokens") or 32768)
        self._timeout_s = float(self._ep.get("request_timeout_s") or 180)
        self._retries = int(self._ep.get("transport_retries", 2))
        self._backoff_s = float(self._ep.get("transport_retry_backoff_s", 3))
        self._need_refresh = False
        self.decisions: list = []                # 每次决策快照 (识别+动作)
        self.last_recognition: dict = {}
        self._replans = 0
        self._done = False

    # -- 请求构造 ----------------------------------------------------------
    def _recognize_if_needed(self, ctx: PlannerContext) -> None:
        """首次决策前: 闭集识别 (任务资产描述) → 识别映射 → sink 写入 adapter。
        之后 runtime 重新观测, 世界即以规范 id 发布。"""
        # gate: 双目标齐 (rubbish+bin) 才跳过 SoM — 首次识别漏 bin 时,
        # place 规范化会发现 world 无 bin 并转 query, 这里必须允许重跑
        # SoM 补全, 否则 query 循环到 watchdog (seed3008 实测)
        required = {"rubbish", "bin"}
        if required <= set(self.last_recognition) or self._recognizer is None:
            return
        from harness.planning.vlm_recognizer import recognize_marked
        if self._recognition_frame_provider is None:
            return
        annotated, marker_map = self._recognition_frame_provider()
        if annotated is None or not marker_map:
            return
        # Set-of-Mark 闭集识别 (双图): 原图给真实外观, 渲染图给实例划分+id;
        # 坐标由代码按 id 查注册表 (VLM 永不输出坐标)
        raw = None
        if self._raw_frame_provider is not None:
            rf = self._raw_frame_provider()
            raw = getattr(rf, "rgb", rf)
        mapping = recognize_marked(self._ep, annotated, marker_map,
                                   self._recognizer_task, raw_rgb=raw)
        if mapping:
            self.last_recognition = mapping
            if self._recognition_sink is not None:
                self._recognition_sink(mapping)
            self._need_refresh = True

    def _build_messages(self, ctx: PlannerContext) -> list:
        registry = self._registry_provider()
        world = ctx.world
        # 可引用集合 = 渲染图上有编号的集合 (overhead 视角 + 已映射规范 id)。
        # 此前给全视角注册表 (19 实例), VLM 引用了渲染图上不存在的 left 视角
        # id (object_l07) 写进识别映射 → 重新观测后失配 (L2, 2026-11-05)
        canon = {c["id"] for c in load_task_assets(self._recognizer_task)}
        registry = [r for r in registry
                    if r.get("view") == "overhead" or r["id"] in canon]
        state = {"ee_pose": [round(world.ee_pose.x, 3), round(world.ee_pose.y, 3),
                             round(world.ee_pose.z, 3)],
                 "gripper_open": round(world.gripper_open, 2),
                 "objects": [{"id": o.id, "pose": [round(o.pose.x, 3), round(o.pose.y, 3),
                                                   round(o.pose.z, 3)],
                              "size": [round(s, 3) for s in o.size]}
                             for o in world.objects
                             if o.view == "overhead" or o.id in canon]}
        rgb = self._camera_provider()
        rgb = getattr(rgb, "rgb", rgb)                # CameraFrame -> rgb 数组
        import numpy as np
        import base64 as b64
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        img_b64 = b64.b64encode(buf.getvalue()).decode("ascii")
        last_exec = self._execution_provider() if self._execution_provider else None
        user_text = (f"任务指令: {self._instruction}\n"
                     f"注册表: {json.dumps(registry, ensure_ascii=False)}\n"
                     f"当前状态: {json.dumps(state, ensure_ascii=False)}\n"
                     f"已有识别映射: {json.dumps(self.last_recognition, ensure_ascii=False)}\n"
                     + (f"最近执行: {json.dumps(last_exec, ensure_ascii=False)}\n"
                        if last_exec else "")
                     + "输出下一步 JSON。")
        return [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]}]

    def _ask(self, ctx: PlannerContext) -> dict:
        messages = self._build_messages(ctx)          # OpenAI content 数组形式
        t0 = time.time()
        text = self._call_anthropic(messages)
        self._stats.setdefault("vlm_latency_s", []).append(round(time.time() - t0, 1))
        # 宽容解析: 首个 { 到末个 }
        i, j = text.find("{"), text.rfind("}")
        if i < 0 or j <= i:
            raise PlannerProtocolError(f"VLM 输出无 JSON: {text[:120]}")
        import os as _os
        if _os.environ.get("EBHARNESS_VLM_DEBUG"):
            print(f"[vlm] raw={text[:300]}", flush=True)
        try:
            return json.loads(text[i:j + 1])
        except json.JSONDecodeError as exc:
            # 协议层错误 (不是传输错误): 交给 runtime 的恢复路径, 不让异常
            # 逃出 decide() 把整集打成 error。
            raise PlannerProtocolError(f"VLM 输出 JSON 非法: {exc}") from exc

    def _place_targets(self) -> tuple[str, str]:
        """place 的 (物体, 容器) —— 从受信的目标 DSL 里取。

        目标 DSL 形如 all_of[contained(object_id=X, container_id=Y),
        gripper_open]。X/Y 就是"要放什么"和"放到哪"。取不到时退回
        ("rubbish", "bin") (本项目的默认任务), 但那是兜底, 不是常态。

        为什么不写死: 换一个任务 (PickUpCup) 就该自动跟着目标 DSL 走;
        写死字符串会让 VLM 提议 place 时指向一个不存在的 id, 被 S3 拒杀,
        而且日志上看像"模型不会用 place"。
        """
        goal = self._goal
        found = _find_contained(goal)
        if found:
            return found
        return ("rubbish", "bin")

    # -- planner API ---------------------------------------------------------
    def _call_anthropic(self, openai_messages: list) -> str:
        """OpenAI content 数组 -> Anthropic Messages 协议 (system 独立 + 图像 base64 块).

        传输策略与 llm_client.OpenAICompatClient 一致: 连接/HTTP/超时错误
        重试 (不是模型的错), 只有响应文本到达后才交给协议层。协议层错误
        (空/非法 JSON) 不在这里重试 — 由 runtime 的恢复路径处理。
        """
        system = ""
        user_blocks = []
        for m in openai_messages:
            if m["role"] == "system":
                system = m["content"]
                continue
            c = m["content"]
            if isinstance(c, str):
                user_blocks.append({"type": "text", "text": c})
            else:
                for part in c:
                    if part["type"] == "text":
                        user_blocks.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "image_url":
                        data_url = part["image_url"]["url"]
                        head, data = data_url.split(",", 1)
                        media = "image/png" if "png" in head else "image/jpeg"
                        user_blocks.append({"type": "image",
                                            "source": {"type": "base64",
                                                       "media_type": media,
                                                       "data": data}})
        body = json.dumps({
            "model": self._ep["model"],
            "max_tokens": self._max_tokens,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user_blocks}],
        }).encode("utf-8")
        url = f"{self._ep['base_url']}/v1/messages"
        retries = int(self._retries)
        backoff = float(self._backoff_s)
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            req = urllib.request.Request(
                url, data=body, method="POST",
                headers={"Content-Type": "application/json",
                         "x-api-key": self._ep["api_key"],
                         "anthropic-version": "2023-06-01"})
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                text = "".join(b.get("text", "") for b in data.get("content", [])
                               if b.get("type") == "text")
                usage = data.get("usage") or {}
                # 累加而非覆盖 (覆盖会丢掉前面调用的用量)
                acc = self._stats.setdefault("vlm_tokens", {})
                for k, v in usage.items():
                    if isinstance(v, (int, float)):
                        acc[k] = acc.get(k, 0) + v
                self._stats["transport_retries_used"] = attempt
                return text
            except (urllib.error.URLError, urllib.error.HTTPError,
                    TimeoutError, OSError, KeyError,
                    json.JSONDecodeError) as e:
                last_err = e
                if attempt < retries:
                    time.sleep(backoff * (attempt + 1))
        raise RuntimeError(f"VLM transport failed after {retries + 1} attempts: "
                           f"{type(last_err).__name__}")

    def decide(self, ctx: PlannerContext) -> ToolCall:
        self._recognize_if_needed(ctx)
        if self._need_refresh:
            self._need_refresh = False
            # 识别映射刚写入: 让 runtime 重新观测 (世界以规范 id 发布), 本轮只出 query
            return ToolCall(call_id=f"c_vlm_q_{ctx.world.version:03d}",
                            kind="query", name="where_is",
                            args={"object_id": "rubbish"},
                            world_version=ctx.world.version)
        out = self._ask(ctx)
        # 双图 SoM 识别是主通道: 决策输出的 recognition 只补充其未覆盖的键
        # (此前的覆盖语义让决策自由引用的 id 冲掉已验证的 SoM 映射)
        dec_rec = {k: v for k, v in out.get("recognition", {}).items()
                    if isinstance(v, str) and v.startswith("object_")}
        merged = {**dec_rec, **self.last_recognition}
        if merged != self.last_recognition:
            grew = set(merged) - set(self.last_recognition)
            self.last_recognition = merged
            if self._recognition_sink is not None and grew:
                self._recognition_sink(self.last_recognition)
        act = out.get("action", {})
        name = act.get("name", "grasp")
        if name == "done":
            return ToolCall(call_id=f"c_vlm_{ctx.world.version:03d}",
                            kind="control", name="done", args={},
                            world_version=ctx.world.version)
        args = dict(act.get("args", {}))
        # 引用 id 翻译: VLM 输出的注册 id 若已被识别映射改名为规范 id, 则引用规范 id
        # (否则映射生效后世界中被引用 id 消失, validator 判 "not in world")
        inv = {rid: canon for canon, rid in self.last_recognition.items()}
        oid = args.get("object_id")
        if oid in inv:
            args["object_id"] = inv[oid]
        # place 语义规范化 (2026-11-05): VLM 曾写 object_id=bin (容器),
        # validator 拒 → place 永不派发。正确语义: 放下手中物 rubbish,
        # 目标 = 容器 canonical 位姿 (投放高度由 adapter 口上偏移承担)
        if name == "place":
            # 语义规范化: place 的正确语义是"放下手中物, 目标是容器"。
            # 物体与容器从**受信的目标 DSL** 里取 (registry.task_goal 的
            # contained 谓词), 不写死字符串 —— 写死会让换一个任务就静默
            # 指向不存在的 id。
            obj_id, cont_id = self._place_targets()
            # 容器确定性校验: 容器是场景中最大的容器资产 (200mm vs 纸团
            # 87mm / 番茄 70mm)。SoM 偶发把小实例判为容器 → 投放点=纸团
            # 原地。体积最大规则覆盖 VLM 的容器选择 — 结构化资产用先验,
            # 语义区分 (纸团 vs 番茄) 才是 VLM 的职责。
            #
            # 候选必须排除"正在被放下的物体"本身: 否则世界里只剩纸团时,
            # 纸团会被当成容器选中, 投放点又落回纸团原地。
            _cands = [o for o in ctx.world.objects
                      if o.source != "ee_anchor" and o.id != obj_id
                      and (o.view == "overhead" or o.id in (obj_id, cont_id))]
            bin_obj = ctx.world.object_by_id(cont_id)
            if _cands:
                _big = max(_cands, key=lambda o: o.size[0] * o.size[1]
                           * o.size[2])
                bin_obj = _big if bin_obj is None else max(
                    (bin_obj, _big),
                    key=lambda o: o.size[0] * o.size[1] * o.size[2])
            # 容器必须比被放物大 — 否则挑中的是干扰物而非容器。
            _ref = ctx.world.object_by_id(obj_id)
            if bin_obj is not None and _ref is not None:
                if (bin_obj.size[0] * bin_obj.size[1] * bin_obj.size[2]
                        <= _ref.size[0] * _ref.size[1] * _ref.size[2]):
                    bin_obj = None
            # 容器尚未发布 (识别映射刚写 / 观测过期): 放行会让 VLM 编造的
            # target_xyz 直达执行 (实测目标=纸团原地)。转 query 重新观测。
            # 注意判据是"世界里没有可用的容器", 不是"存在候选" —— 后者会
            # 让整个 place 分支退化成永远返回 query, place 永不派发。
            if bin_obj is None:
                self._need_refresh = True
                return ToolCall(call_id=f"c_vlm_q_{ctx.world.version:03d}",
                                kind="query", name="where_is",
                                args={"object_id": cont_id},
                                world_version=ctx.world.version)
            args["object_id"] = obj_id
            args["container_id"] = cont_id
            args["target_xyz"] = [bin_obj.pose.x, bin_obj.pose.y,
                                  bin_obj.pose.z]
        # 目标定位 = SAM 注册位姿 (用户定稿: 定位是感知层职责, VLM 不输出坐标 —
        # 2026-11-04 实测 VLM 自由坐标 median 偏差 237mm, 15/17 超 100mm)
        obj = ctx.world.object_by_id(args.get("object_id", ""))
        if obj is None and name == "grasp":
            # 目标实例已从世界消失 (识别映射指向失效 id, 如抓取滑落后
            # 继承/锚定均未接住): 清映射 → query 重观测 → SoM 重跑补识别
            # (旧路径: 放行 grasp → S3 "object not in world" 拒 → 循环拒杀,
            # 最近5批实测 28 次 SCHEMA_INVALID, 2026-11-06)
            self.last_recognition = {}
            if self._recognition_sink is not None:
                self._recognition_sink({})
            self._need_refresh = True
            return ToolCall(call_id=f"c_vlm_q_{ctx.world.version:03d}",
                            kind="query", name="observe", args={},
                            world_version=ctx.world.version)
        if obj is not None and name != "place":
            # 目标定位 = 注册表位姿 (定位是感知层职责, VLM 不输出坐标 —
            # 2026-11-04 实测 VLM 自由坐标 median 偏差 237mm, 15/17 超 100mm)
            #
            # place 例外: 它的 target_xyz 是**容器**位姿, 不是被放物体的
            # 位姿。上面的 place 分支已经把它设成 bin 的位姿; 若这里再
            # 无条件覆盖成"手中物"的位姿, 投放点就变成纸团原地 — 正是
            # 2026-11-06 观测到的"投放点=纸团锚定位"。
            args["target_xyz"] = [obj.pose.x, obj.pose.y, obj.pose.z]
        self.decisions.append({"reason": out.get("reason", ""),
                               "recognition": dict(self.last_recognition),
                               "action": {"name": name, "args": args}})
        return ToolCall(call_id=f"c_vlm_{ctx.world.version:03d}",
                        kind="action", name=name, args=args,
                        world_version=ctx.world.version)

    def replan(self, ctx: PlannerContext) -> Optional[ToolCall]:
        self._replans += 1
        if self._replans > self.give_up_after:
            return None
        return self.decide(ctx)               # 重新看图决策 (识别可能已变化)

    def recover_local(self, ctx: PlannerContext) -> Optional[ToolCall]:
        return None
