"""ExecutionMode 的持久化：把一个 mode 读写成 workspace 里的 manifest。

manifest 的读写本属于 ExecutionMode 这个实体本身，与「制造」mode 的 apply、
「评测」mode 的 correctness/benchmark 都无关。此前它私藏在 apply.py 里，导致
correctness 为了写一行 manifest 反向 import apply（层次倒置）。独立成本模块后，
apply（造）和 correctness（评）都平等地依赖这里，不再互相纠缠。

每个 ExecutionMode 都是一个自包含可运行目录，目录里的 mode_manifest.json 记录它
是谁、从哪 fork、累积了哪些 ChangeRecord、正确性如何——load_mode 能仅凭目录把它
还原回一个 ExecutionMode（baseline 缓存复用、人工排查都靠它）。
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from models import ChangeRecord, ExecutionMode

# 一个 ExecutionMode workspace 里的 manifest 文件名。
MANIFEST_NAME = "mode_manifest.json"
# 统一入口文件名（ExecutionMode.entrypoint 的默认值，contract: build_model() -> (model, tokenizer)）。
DEFAULT_ENTRYPOINT = "build_model.py"


def write_manifest(mode: ExecutionMode) -> None:
    """把 mode 落成 workspace_dir/mode_manifest.json。"""
    payload = {
        "uid": mode.uid,
        "model_id": mode.model_id,
        "strategy_uid": mode.strategy_uid,
        "parent_uid": mode.parent_uid,
        "entrypoint": mode.entrypoint,
        "change_log": [asdict(r) for r in mode.change_log],
        "correctness_passed": mode.correctness_passed,
        "extra": mode.extra,
    }
    path = Path(mode.workspace_dir) / MANIFEST_NAME
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_mode(work_dir: Path) -> ExecutionMode:
    """从 work_dir/mode_manifest.json 还原一个 ExecutionMode。"""
    data = json.loads((work_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    change_log = [ChangeRecord(**r) for r in data.get("change_log", [])]
    return ExecutionMode(
        uid=data["uid"],
        model_id=data["model_id"],
        strategy_uid=data["strategy_uid"],
        workspace_dir=str(work_dir),
        parent_uid=data.get("parent_uid"),
        entrypoint=data.get("entrypoint", DEFAULT_ENTRYPOINT),
        change_log=change_log,
        correctness_passed=data.get("correctness_passed"),
        extra=data.get("extra"),
    )
