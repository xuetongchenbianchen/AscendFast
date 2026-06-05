from __future__ import annotations

import json

from agent_client import AGENT_ENABLED, call_agent_json
from models import AnalysisResult, OptimizationStrategy


def generate_optimization_strategies(
    analysis: AnalysisResult,
    max_count: int = 5,
) -> list[OptimizationStrategy]:
    """LLM-first strategy agent; falls back to rule-based on any failure."""
    if AGENT_ENABLED:
        strategies = _llm_generate_optimization_strategies(analysis, max_count)
        if strategies:
            return strategies
        
    return _rule_generate_optimization_strategies(analysis, max_count)


def _llm_generate_optimization_strategies(
    analysis: AnalysisResult,
    max_count: int,
) -> list[OptimizationStrategy] | None:
    top_ops = analysis.top_ops[:10] if analysis.top_ops else []
    prompt = (
        "You are an NPU optimization strategy expert.\n"
        "Given the AnalysisResult summary, generate up to "
        f"{max_count} ranked OptimizationStrategy candidates.\n\n"
        f"analysis_uid: {analysis.uid}\n"
        f"model_id: {analysis.model_id or 'unknown'}\n"
        f"device: {analysis.device_kind} {analysis.device_name}\n"
        f"dtype: {analysis.dtype}\n"
        f"total_latency_ms: {analysis.total_latency:.4f}\n"
        f"top_ops: {top_ops}\n"
        f"op_type_totals: {json.dumps(analysis.op_type_totals, ensure_ascii=False)}\n"
        f"roofline_summary: {json.dumps(analysis.roofline_summary, ensure_ascii=False)}\n"
        f"profile_findings: {json.dumps(analysis.profile_findings or [], ensure_ascii=False)}\n\n"
        "Return JSON:\n"
        '{"strategies": [{"rule_name": "...", "focus": "...", '
        '"measures": ["...", "..."], "local_speedup_ratio": 1.1}, ...]}'
    )
    result = call_agent_json("strategy-agent", prompt)
    if not isinstance(result, dict):
        return None
    raw_strategies = result.get("strategies")
    if not isinstance(raw_strategies, list) or not raw_strategies:
        return None

    strategies: list[OptimizationStrategy] = []
    for item in raw_strategies:
        if not isinstance(item, dict):
            continue
        _append_strategy(
            strategies, analysis, max_count,
            rule_name=str(item.get("rule_name") or "llm"),
            pct_total=_to_float(item.get("pct_total", 0.0)),
            measures=item.get("measures") if isinstance(item.get("measures"), list) else ["see focus"],
            focus=str(item.get("focus") or ""),
            local_speedup_ratio=_to_float(item.get("local_speedup_ratio")) or None,
            extra={"source": "llm_strategy_agent"},
        )
    return strategies if strategies else None


def _rule_generate_optimization_strategies(
    analysis: AnalysisResult,
    max_count: int = 5,
) -> list[OptimizationStrategy]:
    """
    Rule-based strategy agent: convert AnalysisResult into OptimizationStrategy candidates.
    """
    if max_count <= 0:
        return []

    strategies: list[OptimizationStrategy] = []

    matmul_pct = _op_pct(analysis, "matmul")
    attention_pct = _op_pct(analysis, "flash_attention")
    copy_cast_pct = _op_pct(analysis, "copy_cast")
    rmsnorm_pct = _op_pct(analysis, "rmsnorm")
    reduce_pct = _op_pct(analysis, "reduce")

    if matmul_pct >= 20.0:
        _append_strategy(
            strategies,
            analysis,
            max_count,
            rule_name="matmul",
            pct_total=matmul_pct,
            measures=[
                "prioritize GEMM shape/layout/dtype alignment",
                "avoid redundant layout conversion around matmul",
                "check whether batch, sequence, and hidden dimensions are kernel-friendly",
            ],
            focus=(
                f"matmul accounts for {matmul_pct:.1f}% of profiled top-kernel time; "
                "optimize GEMM execution and its surrounding tensor layout."
            ),
        )

    if attention_pct >= 3.0:
        _append_strategy(
            strategies,
            analysis,
            max_count,
            rule_name="flash_attention",
            pct_total=attention_pct,
            measures=[
                "keep the fused attention path enabled",
                "verify attention mask layout and sequence length handling",
                "remove conversions before and after attention kernels",
            ],
            focus=(
                f"flash_attention accounts for {attention_pct:.1f}% of profiled top-kernel time; "
                "keep attention on the fused high-performance path."
            ),
        )

    if copy_cast_pct >= 1.0:
        _append_strategy(
            strategies,
            analysis,
            max_count,
            rule_name="copy_cast",
            pct_total=copy_cast_pct,
            measures=[
                "remove redundant dtype conversions",
                "keep tensor layout stable across adjacent operators",
                "move unavoidable casts out of repeated hot paths",
            ],
            focus=(
                f"copy_cast accounts for {copy_cast_pct:.1f}% of profiled top-kernel time; "
                "reduce dtype/layout conversion overhead before kernel tuning."
            ),
        )

    norm_reduce_pct = rmsnorm_pct + reduce_pct
    if norm_reduce_pct >= 3.0:
        _append_strategy(
            strategies,
            analysis,
            max_count,
            rule_name="norm_reduce",
            pct_total=norm_reduce_pct,
            measures=[
                "consider fused RMSNorm or reduce kernels",
                "avoid materializing intermediate tensors in normalization paths",
                "check whether reduction axes and tensor layout match optimized kernels",
            ],
            focus=(
                f"normalization/reduce kernels account for {norm_reduce_pct:.1f}% "
                "of profiled top-kernel time; reduce launch and memory overhead."
            ),
        )

    if not strategies and analysis.top_ops:
        _append_strategy(
            strategies,
            analysis,
            max_count,
            rule_name="top_ops",
            pct_total=0.0,
            measures=[
                "inspect the top profiled operators",
                "remove avoidable tensor conversions around top operators",
                "prefer fused or vendor-optimized operator paths",
            ],
            focus="optimize the current top operators from profile analysis.",
            local_speedup_ratio=1.05,
        )

    if not strategies:
        _append_strategy(
            strategies,
            analysis,
            max_count,
            rule_name="generic",
            pct_total=0.0,
            measures=[
                "rerun profiling with enough iterations",
                "inspect operator layout and dtype stability",
                "prioritize measurable changes with correctness checks",
            ],
            focus="profile does not expose a clear hotspot; start with general execution cleanup.",
            local_speedup_ratio=1.03,
        )

    return strategies


def _append_strategy(
    strategies: list[OptimizationStrategy],
    analysis: AnalysisResult,
    max_count: int,
    rule_name: str,
    pct_total: float,
    measures: list[str],
    focus: str,
    local_speedup_ratio: float | None = None,
    extra: dict | None = None,
) -> None:
    if len(strategies) >= max_count:
        return

    speedup = local_speedup_ratio
    if speedup is None:
        speedup = _estimate_local_speedup_ratio(pct_total)

    strategy_extra = {
        "source": "rule_based_strategy_agent",
        "rule_name": rule_name,
        "source_analysis_uid": analysis.uid,
        "model_id": analysis.model_id,
        "device_kind": analysis.device_kind,
        "device_name": analysis.device_name,
        "dtype": analysis.dtype,
        "pct_total": round(pct_total, 6),
    }
    if extra:
        strategy_extra.update(extra)

    strategies.append(
        OptimizationStrategy(
            uid=f"strategy:{analysis.uid}:{rule_name}",
            local_speedup_ratio=round(speedup, 4),
            measures=measures,
            prompt_instruction=_build_strategy_prompt(analysis, focus, measures),
            extra=strategy_extra,
        )
    )


def _build_strategy_prompt(
    analysis: AnalysisResult,
    focus: str,
    measures: list[str],
) -> str:
    measures_text = "\n".join(f"- {measure}" for measure in measures)
    top_ops = ", ".join(analysis.top_ops[:10]) if analysis.top_ops else "none"
    return (
        "Optimize the model execution according to the profiling analysis.\n"
        "Keep numerical correctness unchanged and prefer small, measurable changes.\n\n"
        f"Focus:\n{focus}\n\n"
        f"Measures:\n{measures_text}\n\n"
        "Profile context:\n"
        f"- analysis_uid: {analysis.uid}\n"
        f"- model_id: {analysis.model_id or 'unknown'}\n"
        f"- device: {analysis.device_kind or 'unknown'} {analysis.device_name or ''}\n"
        f"- dtype: {analysis.dtype or 'unknown'}\n"
        f"- total_latency_ms: {analysis.total_latency:.6f}\n"
        f"- top_ops: {top_ops}\n"
        f"- op_type_totals: {analysis.op_type_totals}\n"
        f"- roofline_summary: {analysis.roofline_summary}\n"
    )


def _op_pct(analysis: AnalysisResult, op_type: str) -> float:
    item = (analysis.op_type_totals or {}).get(op_type)
    if not isinstance(item, dict):
        return 0.0
    return _to_float(item.get("pct_total"))


def _estimate_local_speedup_ratio(pct_total: float) -> float:
    pct = min(max(pct_total, 0.0), 100.0)
    return 1.0 + min(0.5, pct / 100.0 * 0.45)


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
