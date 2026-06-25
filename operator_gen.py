# operator (kernel-HOW per the operator-agent split): sits between strategy (WHAT/WHY)
# and apply (wiring-HOW). Given an OperatorSpec, drives operator-agent to design+compile+
# install an AscendC kernel and register it as torch.ops.ascendfast.<op>, returning a
# numerically self-checked OperatorArtifact. Mirrors strategy.py / apply.py structure:
# AGENT_ENABLED guard + call_agent_json, no rule-based fallback. Failure returns None so
# the caller (optimization.py) falls back to official ops — a kernel is high-risk and must
# never block the main chain.
from __future__ import annotations

import json
from pathlib import Path

from agent_client import AGENT_ENABLED, call_agent_json
from models import AnalysisResult, ExecutionMode, OperatorArtifact, OperatorSpec, OptimizationStrategy

_PROJECT_ROOT = Path(__file__).parent
_REGISTRY_PATH = _PROJECT_ROOT / "kernels" / "registry.json"

# 数值自检阈值：fp16 舍入噪声约 2e-3 相对，留一个舒适余量。超过即视为 kernel 有 bug，
# artifact 不可用(gate_operator 也用同一阈值兜一次)。
_NUMERIC_REL_ERR_MAX = 5e-2


# --------------------------------------------------------------------------- #
# registry：已生成算子清单。operator-agent 串行写入(算子编译本就不可并行)，这里只读。
# --------------------------------------------------------------------------- #
def _load_registry() -> dict:
    try:
        return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"operators": []}


def _registry_lookup(op_name: str) -> dict | None:
    """按 op_name 找一条已安装且数值过关的算子记录；用于幂等短路。"""
    for entry in _load_registry().get("operators", []):
        if entry.get("op_name") != op_name:
            continue
        if not entry.get("installed"):
            return None
        err = entry.get("numeric_max_rel_err")
        if err is not None and err > _NUMERIC_REL_ERR_MAX:
            return None
        return entry
    return None


def _artifact_from_registry(entry: dict) -> OperatorArtifact:
    return OperatorArtifact(
        op_name=entry["op_name"],
        qualified_name=entry.get("qualified_name", f"torch.ops.ascendfast.{entry['op_name']}"),
        signature=entry.get("signature", ""),
        installed=bool(entry.get("installed")),
        supported_dtypes=entry.get("supported_dtypes", []),
        numeric_max_rel_err=entry.get("numeric_max_rel_err"),
        usage_note=entry.get("usage_note", ""),
        files=entry.get("files", []),
        metadata={"source": "registry_cache"},
    )


# --------------------------------------------------------------------------- #
# generate_operator：spec → operator-agent → OperatorArtifact（或 None）
# --------------------------------------------------------------------------- #
def generate_operator(
    spec: OperatorSpec,
    strategy: OptimizationStrategy,
    analysis: AnalysisResult,
    base_mode: ExecutionMode | None = None,
) -> OperatorArtifact | None:
    """请 operator-agent 生成(或复用)一个 torch.ops.ascendfast.<op> 自定义算子。

    返回 None 表示这次没拿到可用算子(agent 不可用 / 生成失败 / 数值不过关)；调用方
    据此把 operator_artifact 置 None，apply 退回官方算子。绝不抛异常炸穿主链——与
    strategy.generate 不同(策略为空该显式停)，算子缺席是允许的降级路径。
    """
    # ① 幂等短路：同名算子已注册且数值过关，直接复用，跳过几分钟的重编。
    cached = _registry_lookup(spec.op_name)
    if cached is not None:
        print(f"[operator] reuse registered op '{spec.op_name}' (cached)")
        return _artifact_from_registry(cached)

    if not AGENT_ENABLED:
        print(f"[operator] agent disabled; no custom op for '{spec.op_name}'")
        return None

    try:
        return _llm_generate_operator(spec, strategy, analysis, base_mode)
    except Exception as exc:  # noqa: BLE001 - 算子缺席是允许的降级，绝不炸穿主链
        print(f"[operator] generate failed for '{spec.op_name}': {type(exc).__name__}: {exc}")
        return None


def _format_torch_reference(src: str | None) -> str:
    """把 apply-agent 抽出的 torch 参考渲染进 prompt——它就是算子的 I/O 契约 + 数值 oracle。

    这段 src 不是凭 semantic 现编的玩具，而是 apply-agent 从 build_model() 接线点**真实
    要被替换掉的那段 eager 代码**抽出来的：它的 forward() 是要复现的精确语义，它的
    get_inputs() 给出真实流经该点的形状/dtype。所以 operator-agent 要 (1) 按这里的
    shape/dtype 设计 kernel 的 tiling 与签名，(2) exec 它、用 get_inputs() 造输入、用
    Model.forward 在 fp32 上算 oracle 做数值自检——别再凭 semantic 字面重猜语义。
    """
    if not src:
        return ""
    return (
        "torch_reference — the REAL eager code being replaced at the build_model() wiring\n"
        "site, NOT a toy. Its forward() is the exact math to reproduce; its get_inputs()\n"
        "yields the REAL shapes/dtypes that flow through that point. Design your kernel's\n"
        "tiling/signature for THESE shapes/dtypes, and exec this as the fp32 numeric oracle\n"
        "for your self-check (do NOT re-guess the math from `semantic`):\n"
        f"```python\n{src}\n```\n"
    )


def _llm_generate_operator(
    spec: OperatorSpec,
    strategy: OptimizationStrategy,
    analysis: AnalysisResult,
    base_mode: ExecutionMode | None,
) -> OperatorArtifact | None:
    workspace_hint = (
        f"\nReference model workspace (read its model/config.json for exact arch params): "
        f"{base_mode.workspace_dir}\n" if base_mode is not None else "\n"
    )
    prompt = (
        "You are an AscendC custom-operator engineer. Design, compile, install, and "
        "register ONE custom NPU operator into the torch.ops.ascendfast.* namespace, "
        "then numerically self-check it. Follow the npu-operator skill exactly.\n\n"
        "## Operator request (WHAT/WHY — the HOW is yours)\n"
        f"op_name (register as torch.ops.ascendfast.<op_name>): {spec.op_name}\n"
        f"semantic: {spec.semantic}\n"
        f"why a custom op (official torch_npu insufficient): {spec.why_custom}\n"
        f"fusion_targets: {json.dumps(spec.fusion_targets, ensure_ascii=False)}\n"
        f"arch_params (specialize the kernel for THIS model): "
        f"{json.dumps(spec.arch_params, ensure_ascii=False)}\n"
        f"expected_signature: {spec.expected_signature or '(you decide)'}\n"
        f"{_format_torch_reference(spec.torch_reference)}"
        f"{workspace_hint}"
        "## Profile context\n"
        f"model_id: {analysis.model_id or 'unknown'}\n"
        f"device: {analysis.device_kind or ''} {analysis.device_name or ''}\n"
        f"dtype: {analysis.dtype or 'unknown'}\n"
        f"top_ops: {analysis.top_ops[:10] if analysis.top_ops else []}\n\n"
        "## Hard rules\n"
        "- Touch ONLY the kernels/ tree (ops.json, op_host/, op_kernel/, csrc/adapter_*.cpp, "
        "build scripts). NEVER touch any adaptations/ workspace — wiring the op into "
        "build_model() is the apply step's job, not yours.\n"
        "- The op must end up callable as torch.ops.ascendfast.<op_name> after `import "
        "ascendfast_ops`, and must pass a real numeric check vs an fp32 reference at a "
        "realistic shape (use arch_params; >= 1024 elems).\n"
        "- After success, append a record to kernels/registry.json so it can be reused.\n"
        "- Operator builds are NOT parallelizable (shared build_out/ and global install "
        "path); do one op end to end.\n\n"
        "## Output — return ONLY this JSON (no markdown fences)\n"
        '{"op_name": "<op_name>",\n'
        ' "qualified_name": "torch.ops.ascendfast.<op_name>",\n'
        ' "signature": "<actual call signature you registered>",\n'
        ' "installed": true,\n'
        ' "supported_dtypes": ["float16", "float32"],\n'
        ' "numeric_max_rel_err": <float, max rel err vs fp32 reference>,\n'
        ' "usage_note": "<shape constraints, reshape needs, tuple return, etc. for the apply step>",\n'
        ' "files": ["<kernels-relative path>", "..."],\n'
        ' "metadata": {}}\n'
        "Set installed=false (and explain in usage_note) if you could NOT get the op to "
        "compile, install, and pass the numeric check — do not claim success you did not verify."
    )
    result = call_agent_json("operator-agent", prompt, timeout=3000)
    if not isinstance(result, dict) or "op_name" not in result:
        return None

    op_name = str(result.get("op_name") or spec.op_name)
    artifact = OperatorArtifact(
        op_name=op_name,
        qualified_name=str(result.get("qualified_name") or f"torch.ops.ascendfast.{op_name}"),
        signature=str(result.get("signature") or ""),
        installed=bool(result.get("installed")),
        supported_dtypes=[str(d) for d in result.get("supported_dtypes", [])]
        if isinstance(result.get("supported_dtypes"), list) else [],
        numeric_max_rel_err=_to_float_or_none(result.get("numeric_max_rel_err")),
        usage_note=str(result.get("usage_note") or ""),
        files=[str(p) for p in result.get("files", [])]
        if isinstance(result.get("files"), list) else [],
        metadata=result.get("metadata") if isinstance(result.get("metadata"), dict) else None,
    )
    return artifact


def _to_float_or_none(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
