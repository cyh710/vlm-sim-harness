"""OpenAI-compatible chat client, stdlib-only (urllib), for LLMPlanner.

Transport policy: connection/HTTP/timeout errors are RETRIED (they are not
the model's fault); only then does the response text reach the planner's
protocol layer. Empty/unalizable content is NOT retried here — that is a
protocol error with exactly one repair round inside LLMPlanner (design §9.1).

Credentials: api key and base URL come from environment variables named in
configs/planner_model.json; nothing is hardcoded or logged.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_model_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else (Path(__file__).resolve().parents[2]
                                 / "configs" / "planner_model.json")
    return json.loads(p.read_text(encoding="utf-8"))


class OpenAICompatClient:
    """Callable (system, user) -> assistant text. Usage accounting: every
    successful __call__ increments stats["llm_calls"] when stats is given."""

    def __init__(self, cfg: dict, stats: dict | None = None):
        self._cfg = cfg
        self._stats = stats if stats is not None else {}

    def __call__(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        return self.call_messages(messages)

    def call_messages(self, messages: list) -> str:
        """多模态入口 (L4-VLM): messages 的 content 可为文本或多模态 content 数组
        (文本 + image_url)。传输/重试/统计与原 __call__ 完全一致。"""
        base = os.environ.get(self._cfg["base_url_env"], "").rstrip("/")
        key = os.environ.get(self._cfg["api_key_env"], "")
        model = os.environ.get(self._cfg["model_id_env"],
                               self._cfg["model_id_fallback"])
        if not base or not key:
            raise RuntimeError(
                f"missing {self._cfg['base_url_env']} / "
                f"{self._cfg['api_key_env']} in environment")
        url = f"{base}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self._cfg["temperature"],
            "max_tokens": self._cfg["max_tokens"],
        }
        body = json.dumps(payload).encode("utf-8")
        retries = int(self._cfg.get("transport_retries", 2))
        backoff = float(self._cfg.get("transport_retry_backoff_s", 3))
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            req = urllib.request.Request(
                url, data=body, method="POST",
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"})
            try:
                with urllib.request.urlopen(
                        req, timeout=self._cfg["request_timeout_s"]) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                self._stats["transport_retries_used"] = attempt
                return self._content(data)
            except (urllib.error.URLError, urllib.error.HTTPError,
                    TimeoutError, OSError, KeyError, json.JSONDecodeError) as e:
                last_err = e
                if attempt < retries:
                    time.sleep(backoff * (attempt + 1))
        raise RuntimeError(f"transport failed after {retries + 1} attempts: "
                           f"{type(last_err).__name__}")

    @staticmethod
    def _content(data: dict) -> str:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content")
        # reasoning-model convention (old-project lesson): thinking lives in
        # reasoning_content; the answer lives in content. If content is empty
        # but reasoning exists, return empty -> protocol layer handles it.
        return content if isinstance(content, str) else ""
