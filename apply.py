from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from agent_client import AGENT_ENABLED, call_agent_json
from models import ChangeRecord, ExecutionMode, OptimizationStrategy

_PROJECT_ROOT = Path(__file__).parent
_ADAPTATIONS = _PROJECT_ROOT / "adaptations"
_MANIFEST_NAME = "mode_manifest.json"
_ENTRYPOINT = "build_model.py"

# 权重文件较大，fork 时硬链接而非拷贝；这些后缀视为"大权重"。
_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx"}


# --------------------------------------------------------------------------- #
# baseline：把原始模型物化成一个合法的 ExecutionMode（优化链的根）
# --------------------------------------------------------------------------- #
def ensure_baseline_mode(model_id: str, model_dir: str | Path) -> ExecutionMode:
    """物化 baseline ExecutionMode：workspace 硬链接原始模型 + 标准入口。

    baseline 本身就是一个可运行的 mode，其 build_model() 即朴素加载，
    后续 apply 永远从某个 base_mode 出发，不再特判 model_id。
    """
    work_dir = _ADAPTATIONS / model_id / "baseline"
    manifest_path = work_dir / _MANIFEST_NAME
    if manifest_path.exists():
        return _load_mode(work_dir)

    model_dir = Path(model_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    # 把原始模型软镜像进 workspace：大权重硬链接，其余文件拷贝。
    _mirror_tree(model_dir, work_dir / "model")
    _write_baseline_entrypoint(work_dir)

    mode = ExecutionMode(
        uid="mode:baseline",
        model_id=model_id,
        strategy_uid="baseline",
        workspace_dir=str(work_dir),
        parent_uid=None,
        entrypoint=_ENTRYPOINT,
        change_log=[],
    )
    _write_manifest(mode)
    return mode


# --------------------------------------------------------------------------- #
# apply：fork base_mode → 调 Agent 在副本上叠加优化 → 记录 ChangeRecord
# --------------------------------------------------------------------------- #
def apply_optimization(
    strategy: OptimizationStrategy,
    base_mode: ExecutionMode,
) -> ExecutionMode:
    """在 base_mode 的快照之上叠加 strategy，返回新的 ExecutionMode。

    步骤：
      1. fork base_mode.workspace_dir → 新 work_dir（大权重硬链接，零拷贝）。
      2. 把 base_mode.change_log 注入 prompt，要求 Agent 在已有优化之上叠加。
      3. Agent 原地修改 work_dir，保证 build_model() 仍可运行。
      4. 读回 Agent 报告，追加一条 ChangeRecord，写 manifest。
    """
    new_uid = f"mode:{base_mode.model_id}:{_short_id(strategy.uid)}:{int(time.time())}"
    safe_dir = new_uid.replace(":", "_").replace("/", "_")
    work_dir = _ADAPTATIONS / base_mode.model_id / safe_dir
    _fork_workspace(Path(base_mode.workspace_dir), work_dir)

    record: ChangeRecord | None = None
    if AGENT_ENABLED:
        record = _llm_apply_optimization(strategy, base_mode, new_uid, work_dir)

    # 运行门禁：agent 说自己改完了，但融合/自定义算子的参数错误（dtype/shape/布局）
    # 只在 forward 时才炸——构造能过、前向才挂。这里在接受这条 record 之前实际跑一次
    # 前向；跑不通就丢弃 record，于是 gate_apply 据"日志没增长"判这次 apply 无效，
    # 既不进 correctness（不会留下 correctness_passed=null），也不会递归进一个坏 mode。
    if record is not None and not _workspace_forward_ok(
        new_uid, base_mode, work_dir, record
    ):
        record = None

    # 只有产出了真 ChangeRecord 才追加——None 不入 change_log，从根上消除
    # asdict(None) 崩溃；gate_apply 据"日志是否增长"判定这次 apply 是否有效。
    new_change_log = base_mode.change_log + ([record] if record is not None else [])
    mode = ExecutionMode(
        uid=new_uid,
        model_id=base_mode.model_id,
        strategy_uid=strategy.uid,
        workspace_dir=str(work_dir),
        parent_uid=base_mode.uid,
        entrypoint=base_mode.entrypoint,
        change_log=new_change_log,
    )
    _write_manifest(mode)
    return mode


# --------------------------------------------------------------------------- #
# 运行门禁：接受 agent 的 ChangeRecord 前，实际跑一次前向，挡住"构造能过、
# forward 才炸"的算子参数错误（dtype/shape/布局）。
# --------------------------------------------------------------------------- #
def _workspace_forward_ok(
    new_uid: str,
    base_mode: ExecutionMode,
    work_dir: Path,
    record: ChangeRecord,
) -> bool:
    """import fork 出的 workspace 并跑一次小前向；能跑通返回 True，否则 False。

    构造一个临时 ExecutionMode 指向 work_dir（change_log 含本次 record），用项目
    唯一的加载入口 load_build_model 取 (model, tokenizer)，喂一个极短 prompt 跑一次
    no_grad forward。NPU 自定义/融合算子的参数错误只在 forward 时暴露，构造检查抓不到。

    本函数吞掉一切异常（含 NPU 算子错误）：门禁自身绝不能炸穿 apply。失败时打印
    原因，方便人工排查。
    """
    try:
        from workspace_loader import load_build_model

        probe = ExecutionMode(
            uid=new_uid,
            model_id=base_mode.model_id,
            strategy_uid=record.strategy_uid,
            workspace_dir=str(work_dir),
            parent_uid=base_mode.uid,
            entrypoint=base_mode.entrypoint,
            change_log=base_mode.change_log + [record],
        )
        model, tokenizer = load_build_model(probe)

        import torch

        device = next(model.parameters()).device
        input_ids = tokenizer("hello world", return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            model(input_ids)
        # 探针用完即释放，别和后续真实评测在显存里压两份模型。
        del model
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    except BaseException as exc:  # noqa: BLE001 - 门禁兜住一切，绝不炸穿 apply
        print(f"[apply] forward gate failed for {new_uid}: {type(exc).__name__}: {exc}")
        return False


def _llm_apply_optimization(
    strategy: OptimizationStrategy,
    base_mode: ExecutionMode,
    new_uid: str,
    work_dir: Path,
) -> ChangeRecord | None:
    prior = _format_change_log(base_mode.change_log)
    prompt = (
        f"{strategy.prompt_instruction}\n\n"
        "## Already-applied optimizations (DO NOT undo or duplicate these)\n"
        f"{prior}\n\n"
        "## Your workspace\n"
        f"Workspace directory (absolute, already forked from the parent mode): {work_dir}\n"
        f"Entrypoint contract: `{work_dir / _ENTRYPOINT}` MUST keep exposing\n"
        "    build_model() -> (model, tokenizer)\n"
        "After your changes build_model() must still return a runnable model. Stack the\n"
        "new optimization ON TOP of the already-applied ones — edit build_model.py and/or\n"
        "add patch/config/graph files inside the workspace. Large weight files in\n"
        "model/ are hardlinked from the parent: if you must mutate weights, write NEW\n"
        "files (do not edit hardlinked ones in place) so the parent stays intact.\n\n"
        "When done, return ONLY this JSON describing what you changed:\n"
        '{"kind": "forward_patch|operator_fusion|graph_rewrite|kvcache|parallelism|quantize|config|custom",\n'
        ' "summary": "<one sentence: what you did>",\n'
        ' "details": "<modules/operators touched, why, any constraints>",\n'
        ' "files": ["<relative-to-workspace path>", "..."],\n'
        ' "revert_cmd": "<shell cmd to undo, or null>",\n'
        ' "metadata": {}}'
    )
    result = call_agent_json("apply-agent", prompt, timeout=1000)
    if not isinstance(result, dict) or "summary" not in result:
        return None
    files = result.get("files")
    return ChangeRecord(
        mode_uid=new_uid,
        strategy_uid=strategy.uid,
        kind=str(result.get("kind") or "custom"),
        summary=str(result.get("summary") or ""),
        details=str(result.get("details") or ""),
        # file->这次修改涉及到哪些文件。
        files=[str(p) for p in files] if isinstance(files, list) else [],
        revert_cmd=result.get("revert_cmd") or None,
        metadata=result.get("metadata") if isinstance(result.get("metadata"), dict) else None,
    )


# --------------------------------------------------------------------------- #
# 快照 fork（CoW 思路：大权重硬链接，代码/config 真实拷贝）
# --------------------------------------------------------------------------- #
def _fork_workspace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    _mirror_tree(src, dst)


def _mirror_tree(src: Path, dst: Path) -> None:
    """把 src 复制到 dst：大权重文件硬链接（零拷贝、断链即独立），其余真实拷贝。"""
    dst.mkdir(parents=True, exist_ok=True)
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            s = Path(root) / name
            d = target_dir / name
            if d.exists():
                d.unlink()
            if s.suffix.lower() in _WEIGHT_SUFFIXES:
                try:
                    os.link(s, d)          # 硬链接：同一 inode，改写需先 unlink
                    continue
                except OSError:
                    pass                   # 跨设备等情况退回拷贝
            shutil.copy2(s, d)


# --------------------------------------------------------------------------- #
# manifest 读写 / change_log 渲染
# --------------------------------------------------------------------------- #
def _write_manifest(mode: ExecutionMode) -> None:
    payload = {
        "uid": mode.uid,
        "model_id": mode.model_id,
        "strategy_uid": mode.strategy_uid,
        "parent_uid": mode.parent_uid,
        "entrypoint": mode.entrypoint,
        # 如果change_log里面有一个None就会崩溃
        "change_log": [asdict(r) for r in mode.change_log],
        "correctness_passed": mode.correctness_passed,
        "extra": mode.extra,
    }
    path = Path(mode.workspace_dir) / _MANIFEST_NAME
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_mode(work_dir: Path) -> ExecutionMode:
    data = json.loads((work_dir / _MANIFEST_NAME).read_text(encoding="utf-8"))
    change_log = [ChangeRecord(**r) for r in data.get("change_log", [])]
    return ExecutionMode(
        uid=data["uid"],
        model_id=data["model_id"],
        strategy_uid=data["strategy_uid"],
        workspace_dir=str(work_dir),
        parent_uid=data.get("parent_uid"),
        entrypoint=data.get("entrypoint", _ENTRYPOINT),
        change_log=change_log,
        correctness_passed=data.get("correctness_passed"),
        extra=data.get("extra"),
    )


def _format_change_log(change_log: list[ChangeRecord]) -> str:
    if not change_log:
        return "(none — this is the first optimization on the baseline model)"
    lines = []
    for i, r in enumerate(change_log, 1):
        lines.append(f"{i}. [{r.kind}] {r.summary}")
        if r.details:
            lines.append(f"   details: {r.details}")
        if r.files:
            lines.append(f"   files: {', '.join(r.files)}")
    return "\n".join(lines)


def _short_id(uid: str) -> str:
    return uid.replace(":", "_").replace("/", "_").split("_")[-1][:24] or "opt"


# --------------------------------------------------------------------------- #
# baseline 入口模板：朴素加载，作为优化链根节点的可运行入口
# --------------------------------------------------------------------------- #
def _write_baseline_entrypoint(work_dir: Path) -> None:
    path = work_dir / _ENTRYPOINT
    if path.exists():
        return
    path.write_text(_BASELINE_ENTRYPOINT, encoding="utf-8")


_BASELINE_ENTRYPOINT = '''"""Unified ExecutionMode entrypoint (baseline).

Contract: build_model() -> (model, tokenizer). correctness/profile load the
optimized model ONLY through this function, so every optimized fork keeps the
same signature regardless of what optimization it embeds.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE / "model"


def build_model(device: str | None = None, dtype=torch.float16):
    device = device or ("npu:0" if hasattr(torch, "npu") and torch.npu.is_available()
                         else "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(
        str(_MODEL_DIR), trust_remote_code=True, low_cpu_mem_usage=True, dtype=dtype,
    )
    model.eval().to(device)
    return model, tokenizer
'''
