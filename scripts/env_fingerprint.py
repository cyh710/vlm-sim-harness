#!/usr/bin/env python3
"""环境指纹采集 — 把"这次结果是在什么环境上跑出来的"固定下来。

一次可复现的结果 = 代码版本 + 提示词版本 + 环境。三者都要被记录, 否则
"换个环境跑就不一样"这件事永远说不清。

记录内容: OS / Python / 关键依赖版本 / 本仓 git commit 与工作区是否干净 /
模型与端点配置 (只记名字与端点, 不记密钥) / 预注册里冻结的哈希。

用法:
    python scripts/env_fingerprint.py --out env_fingerprint.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger("fingerprint")

# 只记这几个依赖 — 框架层本身零第三方依赖, 这几个是可选的
TRACKED_PACKAGES = ["numpy", "pillow", "pytest"]

# 与结果强相关、需要固定住的环境变量 (只记存在性, 不记值)
TRACKED_ENV = ["CONDA_DEFAULT_ENV", "VIRTUAL_ENV", "PYTHONHASHSEED"]

# 与模型行为强相关的配置 (记值 — 这些不是密钥)
MODEL_ENV = ["VLM_BASE_URL", "VLM_MODEL", "OPENAI_BASE_URL", "OPENAI_MODEL"]


def _run(cmd: list) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return (out.stdout or out.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"<unavailable: {type(exc).__name__}>"


def _git_sha(path: Path):
    if not path.exists():
        return None
    sha = _run(["git", "-C", str(path), "rev-parse", "HEAD"])
    dirty = _run(["git", "-C", str(path), "status", "--porcelain"])
    return {"sha": sha if len(sha) == 40 else None, "dirty": bool(dirty.strip())}


def _pkg_versions() -> dict:
    out = {}
    try:
        from importlib import metadata
    except Exception:  # noqa: BLE001
        return out
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


def _frozen_artifacts(repo: Path) -> dict:
    """预注册里冻结的哈希 — 环境指纹与它们一起才构成完整的复现信息。"""
    p = repo / "configs" / "preregistration.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("frozen_artifacts", {})
    except (OSError, ValueError):
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="采集环境指纹")
    ap.add_argument("--out", default="env_fingerprint.json")
    ap.add_argument("--repo", default=".", help="仓库根目录 (取 git commit)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    repo = Path(args.repo).resolve()

    fp = {
        "collected_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "os": {
            "platform": platform.platform(),
            "release": platform.release(),
            "machine": platform.machine(),
            "libc": "-".join(platform.libc_ver()),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
        "packages": _pkg_versions(),
        "env": {k: os.environ.get(k) for k in TRACKED_ENV if k in os.environ},
        # 端点与模型名要记 — 换模型等于换实验; 密钥一律不记。
        "model": {k: os.environ.get(k) for k in MODEL_ENV if k in os.environ},
        "git": {"harness": _git_sha(repo)},
        "frozen_artifacts": _frozen_artifacts(repo),
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(fp, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    LOG.info("环境指纹已写入 %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
