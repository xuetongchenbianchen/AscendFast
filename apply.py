# apply (the HOW): forks a workspace (hardlinked weights), invokes apply-agent to
# stack one optimization, and runs a forward gate before accepting the ChangeRecord.
# Forks honor the build_model() entrypoint contract.
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from agent_client import AGENT_ENABLED, call_agent_json
from models import (
    CHANGE_KINDS,
    ChangeRecord,
    ExecutionMode,
    OperatorArtifact,
    OptimizationStrategy,
)
from mode_store import (
    DEFAULT_ENTRYPOINT,
    MANIFEST_NAME,
    load_mode,
    write_manifest,
)

_PROJECT_ROOT = Path(__file__).parent
_ADAPTATIONS = _PROJECT_ROOT / "adaptations"

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
    manifest_path = work_dir / MANIFEST_NAME
    if manifest_path.exists():
        return load_mode(work_dir)

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
        entrypoint=DEFAULT_ENTRYPOINT,
        change_log=[],
    )
    write_manifest(mode)
    return mode


# --------------------------------------------------------------------------- #
# apply：fork base_mode → 调 Agent 在副本上叠加优化 → 记录 ChangeRecord
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# apply：两阶段握手。apply-agent 无 Agent 工具、不能直接调
# operator-agent，所以"apply-agent 发布算子需求"由 Python 中介成两阶段：
#
#   apply_discover()  — fork base_mode，让 apply-agent 读真实 forward 代码后二选一：
#                       (A) 直接用官方/eager 算子改完 → 返回带 ChangeRecord 的完成 mode；
#                       (B) 判断需要自定义算子 → 返回 pending mode（extra 里挂一份
#                           OperatorSpec dict，change_log 不增长，因为算子还不存在、
#                           现在接线过不了 forward gate）。
#   〔optimization.py 在两者之间插入 operator stage：generate_operator(spec)→artifact〕
#   apply_wire()      — 复用 pending mode 那个已 fork 的 workspace（不再 fork），把已
#                       验证的 OperatorArtifact 接进 build_model()，返回带 ChangeRecord
#                       的完成 mode。
#
# 防循环：算子请求只发生在 discover 一次；wire 阶段的 prompt 不给 apply-agent 再请求
# 算子的选项。operator 生成失败 ≠ 策略失败：artifact 为 None 时 optimization 直接弃用
# 这条策略（pending mode 没有 ChangeRecord，本就不该递归进去）。
# --------------------------------------------------------------------------- #
def apply_discover(
    strategy: OptimizationStrategy,
    base_mode: ExecutionMode,
) -> ExecutionMode:
    """Phase 1：fork base_mode → 让 apply-agent 读真实代码后决定怎么走，返回新 ExecutionMode。

    返回的 mode 有两种形态，调用方靠 extra["phase"] 区分：
      - "operator_pending"：apply-agent 请求一个自定义算子。extra["pending_operator_spec"]
        是 OperatorSpec dict（含 torch_reference）。change_log 未增长——还没真改完。
        调用方据此跑 operator stage，拿到 artifact 后调 apply_wire() 续上。
      - 无 phase 标记（或 None）：apply-agent 已用官方/eager 算子改完。change_log 增长了
        一条 ChangeRecord（且已过 forward gate）；与老版 apply 行为一致，可直接进 correctness。
    agent 失败时返回的 mode change_log 不增长、也无 pending 标记，gate_apply 会据此拦下。
    """
    new_uid = f"mode:{base_mode.model_id}:{_short_id(strategy.uid)}:{int(time.time())}"
    safe_dir = new_uid.replace(":", "_").replace("/", "_")
    work_dir = _ADAPTATIONS / base_mode.model_id / safe_dir
    _fork_workspace(Path(base_mode.workspace_dir), work_dir)

    record: ChangeRecord | None = None
    pending_spec: dict | None = None
    if AGENT_ENABLED:
        outcome = _llm_apply_discover(strategy, base_mode, new_uid, work_dir)
        if isinstance(outcome, dict):           # operator_request：一份 OperatorSpec dict
            pending_spec = outcome
        else:                                   # ChangeRecord | None
            record = outcome

    # 完成路径才跑 forward gate：融合/自定义算子的参数错误只在 forward 时暴露。
    # pending 路径还没接任何算子，没什么可前向验证的，跳过。
    if record is not None and not _workspace_forward_ok(new_uid, base_mode, work_dir, record):
        record = None

    extra: dict = {}
    if pending_spec is not None:
        extra = {"phase": "operator_pending", "pending_operator_spec": pending_spec}
    new_change_log = base_mode.change_log + ([record] if record is not None else [])
    mode = ExecutionMode(
        uid=new_uid,
        model_id=base_mode.model_id,
        strategy_uid=strategy.uid,
        workspace_dir=str(work_dir),
        parent_uid=base_mode.uid,
        entrypoint=base_mode.entrypoint,
        change_log=new_change_log,
        extra=extra or None,
    )
    write_manifest(mode)
    return mode


def apply_wire(
    strategy: OptimizationStrategy,
    pending_mode: ExecutionMode,
    operator_artifact: OperatorArtifact,
) -> ExecutionMode:
    """Phase 2：在 pending_mode 已 fork 的 workspace 上，把已验证算子接进 build_model()。

    pending_mode 来自 apply_discover() 的 "operator_pending" 形态：它的 workspace 已经
    fork 好、状态保留，这里**不再 fork**，原地续作。operator_artifact 已过 gate_operator
    （installed + 数值过关），通过 _format_operator_artifact 注入 prompt，让 apply-agent
    像消费官方 npu_* 一样接它（并保留 fallback）。返回带新 ChangeRecord 的完成 mode；
    其 change_log 恰在 pending_mode（== 原 base）之上增长一条，gate_apply 据此判定有效。
    """
    work_dir = Path(pending_mode.workspace_dir)
    new_uid = pending_mode.uid

    record: ChangeRecord | None = None
    if AGENT_ENABLED:
        record = _llm_apply_wire(strategy, pending_mode, new_uid, work_dir, operator_artifact)

    if record is not None and not _workspace_forward_ok(new_uid, pending_mode, work_dir, record):
        record = None

    new_change_log = pending_mode.change_log + ([record] if record is not None else [])
    mode = ExecutionMode(
        uid=new_uid,
        model_id=pending_mode.model_id,
        strategy_uid=strategy.uid,
        workspace_dir=str(work_dir),
        parent_uid=pending_mode.parent_uid,    # 原 base，不是 pending 自己
        entrypoint=pending_mode.entrypoint,
        change_log=new_change_log,
        extra=None,                            # 清掉 pending 标记：这是完成态
    )
    write_manifest(mode)
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


def _llm_apply_discover(
    strategy: OptimizationStrategy,
    base_mode: ExecutionMode,
    new_uid: str,
    work_dir: Path,
) -> dict | ChangeRecord | None:
    """Phase 1 的 agent 往返。返回 OperatorSpec dict（operator_request）/ ChangeRecord / None。

    apply-agent 读真实 forward 代码后二选一（discriminated union by "type"）：
      - "operator_request"：需要自定义算子 → 这里解析并返回一份规范化的 OperatorSpec dict
        （含 torch_reference），上层挂进 mode.extra["pending_operator_spec"]。
      - "change_record"：直接用官方/eager 改完 → 返回 ChangeRecord（走老路）。
    """
    prior = _format_change_log(base_mode.change_log)
    hint = _format_strategy_custom_op_hint(strategy)
    prompt = (
        f"{strategy.prompt_instruction}\n\n"
        "## Already-applied optimizations (DO NOT undo or duplicate these)\n"
        f"{prior}\n\n"
        f"{hint}"
        "## Your workspace\n"
        f"Workspace directory (absolute, already forked from the parent mode): {work_dir}\n"
        f"Entrypoint contract: `{work_dir / DEFAULT_ENTRYPOINT}` MUST keep exposing\n"
        "    build_model() -> (model, tokenizer)\n"
        "Large weight files in model/ are hardlinked from the parent: if you must mutate\n"
        "weights, write NEW files (do not edit hardlinked ones in place).\n\n"
        "## Decide: do you need a custom AscendC operator FIRST?\n"
        "Read the REAL forward code in this workspace (build_model.py, patches/, and\n"
        "model/config.json for exact arch params) before choosing. Return ONE of:\n\n"
        "### Option A — request a custom operator (you'll wire it next round)\n"
        "Only when official torch_npu lacks the op AND a fused/specialized kernel would cut\n"
        "real launch/cast/GM-roundtrip overhead. Return:\n"
        '{"type": "operator_request",\n'
        ' "operator_spec": {\n'
        '   "op_name": "<snake_case>",\n'
        '   "semantic": "<math/pseudocode>",\n'
        '   "why_custom": "<why official torch_npu is insufficient>",\n'
        '   "fusion_targets": ["<op>", "..."],\n'
        '   "arch_params": {"hidden_size": <int from config.json>, "eps": <float>, "dtype": "<str>"},\n'
        '   "expected_signature": "<optional call signature, or null>",\n'
        '   "torch_reference": "<self-contained torch reference source — see format below>"\n'
        ' },\n'
        ' "reason": "<one sentence: why this operator, grounded in the real code you read>"}\n\n'
        "torch_reference is the operator's I/O contract AND numeric oracle: it MUST mirror the\n"
        "ACTUAL eager code you are about to replace at the build_model() wiring site — not a\n"
        "freshly-invented version. Read that real forward, then distill it into a runnable,\n"
        "self-contained string defining a `class Model(torch.nn.Module)` whose forward()\n"
        "reproduces EXACTLY the math being replaced in pure torch eager ops, plus a module-level\n"
        "`def get_inputs():` returning a tuple of input tensors at the REAL shapes/dtypes that\n"
        "flow through that point (read them off the real code + config.json; >= 1024 elements,\n"
        "NOT toy shapes). operator-agent designs the kernel's tiling/signature for THESE\n"
        "shapes/dtypes and exec()s this as the fp32 numeric oracle, so input order MUST match\n"
        "your expected_signature.\n\n"
        "The example below illustrates ONLY the string-encoding format (escaped \\n, the\n"
        "`class Model` / `get_inputs` contract). Its math, shapes, and constants are PLACEHOLDERS\n"
        "from a made-up op — do NOT reuse them. Derive every line from the real forward you just\n"
        "read; if your torch_reference resembles this example (same op, same numbers), you almost\n"
        "certainly failed to read the real code. The numbers below are deliberately NOT this\n"
        "model's config values.\n"
        '  "import torch\\n'
        "class Model(torch.nn.Module):\\n"
        "    def __init__(self, h=1280, eps=3e-5):\\n"
        "        super().__init__(); self.h=h; self.eps=eps\\n"
        "    def forward(self, x, w):\\n"
        "        # ILLUSTRATIVE ONLY — replace with the EXACT eager math at your wiring site.\\n"
        "        return torch.nn.functional.gelu(x) @ w\\n"
        "def get_inputs():\\n"
        '    return (torch.randn(48,1280,dtype=torch.float16), torch.randn(1280,1280,dtype=torch.float16))"\n\n'
        "### Option B — apply directly, no custom operator needed\n"
        "When official ops suffice (or no kernel is warranted). Make the change now and return\n"
        "the usual ChangeRecord — build_model() must still return a runnable, equivalent model:\n"
        '{"type": "change_record",\n'
        ' "kind": "forward_patch|operator_fusion|graph_rewrite|loading_time|custom",\n'
        ' "summary": "<one sentence: what you did>",\n'
        ' "details": "<modules/operators touched, why, constraints>",\n'
        ' "files": ["<relative-to-workspace path>", "..."],\n'
        ' "revert_cmd": "<shell cmd to undo, or null>",\n'
        ' "metadata": {}}\n\n'
        "You may request an operator only in THIS round. Choose after reading the real code."
    )
    result = call_agent_json("apply-agent", prompt, timeout=2000)
    if not isinstance(result, dict):
        return None
    if result.get("type") == "operator_request":
        return _parse_operator_spec(result.get("operator_spec"))
    # 没标 type 或标了 change_record，都按完成路径解析（向后兼容旧 agent 只回 ChangeRecord）。
    return _parse_change_record(result, new_uid, strategy.uid)


def _llm_apply_wire(
    strategy: OptimizationStrategy,
    pending_mode: ExecutionMode,
    new_uid: str,
    work_dir: Path,
    operator_artifact: OperatorArtifact,
) -> ChangeRecord | None:
    """Phase 2 的 agent 往返：把已验证算子接进 build_model()，返回 ChangeRecord。"""
    prior = _format_change_log(pending_mode.change_log)
    operator_block = _format_operator_artifact(operator_artifact)
    prompt = (
        f"{strategy.prompt_instruction}\n\n"
        "## Already-applied optimizations (DO NOT undo or duplicate these)\n"
        f"{prior}\n\n"
        f"{operator_block}"
        "## Your workspace\n"
        f"Workspace directory (absolute, your fork from last round): {work_dir}\n"
        f"Entrypoint contract: `{work_dir / DEFAULT_ENTRYPOINT}` MUST keep exposing\n"
        "    build_model() -> (model, tokenizer)\n\n"
        "The custom operator above is built, installed, and numerically verified. WIRE IT\n"
        "into build_model() now (keep an official/eager fallback). You CANNOT request another\n"
        "operator. When done, return ONLY this JSON:\n"
        '{"type": "change_record",\n'
        ' "kind": "forward_patch|operator_fusion|graph_rewrite|loading_time|custom",\n'
        ' "summary": "<one sentence: what you wired>",\n'
        ' "details": "<modules/operators touched, fallback path, constraints>",\n'
        ' "files": ["<relative-to-workspace path>", "..."],\n'
        ' "revert_cmd": "<shell cmd to undo, or null>",\n'
        ' "metadata": {}}'
    )
    result = call_agent_json("apply-agent", prompt, timeout=2000)
    if not isinstance(result, dict):
        return None
    return _parse_change_record(result, new_uid, strategy.uid)


def _parse_operator_spec(raw: object) -> dict | None:
    """把 apply-agent 的 operator_spec 字段规范化成一个干净 OperatorSpec dict（缺/坏则 None）。

    存 dict 而非 OperatorSpec 实例，让它能进 ExecutionMode.extra 并随 manifest 落盘；
    optimization 要用时再 OperatorSpec(**spec) 物化。op_name 与 semantic 是底线，缺任一
    就当没请求（返回 None → discover 视为"没产出"，gate_apply 拦下，走下一条策略）。
    """
    if not isinstance(raw, dict):
        return None
    op_name = str(raw.get("op_name") or "").strip()
    semantic = str(raw.get("semantic") or "").strip()
    if not op_name or not semantic:
        return None
    fusion = raw.get("fusion_targets")
    arch = raw.get("arch_params")
    return {
        "op_name": op_name,
        "semantic": semantic,
        "why_custom": str(raw.get("why_custom") or "").strip(),
        "fusion_targets": [str(t) for t in fusion] if isinstance(fusion, list) else [],
        "arch_params": arch if isinstance(arch, dict) else {},
        "expected_signature": (str(raw["expected_signature"]).strip()
                               if raw.get("expected_signature") else None),
        "torch_reference": (str(raw["torch_reference"])
                            if raw.get("torch_reference") else None),
    }


def _parse_change_record(result: dict, mode_uid: str, strategy_uid: str) -> ChangeRecord | None:
    """把 apply-agent 的 change_record 形态 JSON 解析成 ChangeRecord（缺 summary 则 None）。"""
    if "summary" not in result:
        return None
    files = result.get("files")
    # kind 规范化到 CHANGE_KINDS：agent 报了陈旧/未知值（如旧枚举里的 kvcache）就
    # 收敛成 custom，避免脏 kind 流进 manifest 和 ledger 的按-lever 归因。
    raw_kind = str(result.get("kind") or "custom")
    kind = raw_kind if raw_kind in CHANGE_KINDS else "custom"
    return ChangeRecord(
        mode_uid=mode_uid,
        strategy_uid=strategy_uid,
        kind=kind,
        summary=str(result.get("summary") or ""),
        details=str(result.get("details") or ""),
        files=[str(p) for p in files] if isinstance(files, list) else [],
        revert_cmd=result.get("revert_cmd") or None,
        metadata=result.get("metadata") if isinstance(result.get("metadata"), dict) else None,
    )


def _format_strategy_custom_op_hint(strategy: OptimizationStrategy) -> str:
    """把 strategy.extra.custom_operator 渲染成一个「提示」（非强制；None 时返回空串）。

    strategy 凭 profile 热点名给的算子建议——可能对、也可能脱离真实代码。apply-agent 读完
    真实 forward 后自己拍板：认同就走 Option A，不认同/官方算子够用就走 Option B。
    """
    op_hint = (strategy.extra or {}).get("custom_operator")
    if not isinstance(op_hint, dict):
        return ""
    return (
        "## Strategy hint — a custom operator the strategy-agent *suspected* might help\n"
        "Based on profile hotspots only (it did NOT see the real code). Treat as a suggestion;\n"
        "YOU decide after reading the actual forward. Take it (Option A) only if the real code\n"
        "agrees; otherwise ignore it (Option B).\n"
        f"- suggested op_name: {op_hint.get('op_name')}\n"
        f"- suggested semantic: {op_hint.get('semantic')}\n"
        f"- suggested why_custom: {op_hint.get('why_custom')}\n\n"
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


def _format_operator_artifact(artifact: OperatorArtifact | None) -> str:
    """把已验证的自定义算子渲染进 apply-agent 的 prompt（None 时返回空串）。

    artifact 到这之前已过 gate_operator（installed + 数值过关），所以这里告诉
    apply-agent：这个算子「已验证、可直接用」，优先接它、保留官方/eager fallback。
    """
    if artifact is None:
        return ""
    dtypes = ", ".join(artifact.supported_dtypes) or "unknown"
    err = (f"{artifact.numeric_max_rel_err:.4g}"
           if artifact.numeric_max_rel_err is not None else "n/a")
    return (
        "## Pre-built custom operator (already compiled, installed, and numerically verified)\n"
        "An operator-agent has ALREADY built a custom AscendC kernel for this strategy and\n"
        "registered it. You do NOT build kernels — just WIRE this op into build_model().\n"
        f"- call it as: {artifact.qualified_name}\n"
        f"- signature: {artifact.signature}\n"
        f"- supported dtypes: {dtypes}\n"
        f"- verified max rel err vs fp32 ref: {err}\n"
        f"- usage note: {artifact.usage_note}\n"
        "How to use: `import ascendfast_ops` INSIDE build_model()/a patch function (never at\n"
        "module top level — workspace isolation), then call the op above. ALWAYS keep a\n"
        "fallback: probe the op once on a tiny tensor; if import/probe fails, fall back to the\n"
        "official torch_npu op or eager. Prefer this verified custom op as the primary path.\n\n"
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
    path = work_dir / DEFAULT_ENTRYPOINT
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
