from __future__ import annotations

import json

from agent_client import AGENT_ENABLED, call_agent_json
from models import AnalysisResult, OptimizationStrategy


def generate_optimization_strategies(
    analysis: AnalysisResult,
    max_count: int = 5,
) -> list[OptimizationStrategy]:
    """LLM strategy agent (no rule-based fallback).

    Requires the agent runtime to be enabled; raises otherwise so the failure
    is visible instead of silently degrading to an empty strategy list.
    """
    try:
        if not AGENT_ENABLED:
            raise RuntimeError(
                f"strategy agent unavailable: AGENT_ENABLED={AGENT_ENABLED!r}"
            )
        return _llm_generate_optimization_strategies(analysis, max_count) or []
    except Exception as exc:
        print(f"[strategy] generate failed (AGENT_ENABLED={AGENT_ENABLED!r}): {exc}")
        raise


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
        '{"strategies": [{"focus": "<bottleneck + target mechanism, one line>", '
        '"measures": ["<mechanism step, describe what, not implementation code>"], '
        '"local_speedup_ratio": 1.1}, ...]}\n'
        "local_speedup_ratio is a conservative estimate, >= 1.0."
    )
    result = call_agent_json("strategy-agent", prompt, timeout=1000)
    if not isinstance(result, dict):
        return None
    raw_strategies = result.get("strategies")
    if not isinstance(raw_strategies, list) or not raw_strategies:
        return None

    strategies: list[OptimizationStrategy] = []
    for item in raw_strategies:
        if not isinstance(item, dict) or len(strategies) >= max_count:
            continue
        measures = item.get("measures") if isinstance(item.get("measures"), list) else ["see focus"]
        focus = str(item.get("focus") or "")
        speedup = _to_float(item.get("local_speedup_ratio")) or 1.05  
        strategies.append(
            OptimizationStrategy(
                uid=f"strategy:{analysis.uid}:{len(strategies) + 1}",
                local_speedup_ratio=round(speedup, 4),
                measures=measures,
                prompt_instruction=_build_strategy_prompt(analysis, focus, measures),
                extra={
                    "source_analysis_uid": analysis.uid,
                    "model_id": analysis.model_id,
                    "device_kind": analysis.device_kind,
                    "device_name": analysis.device_name,
                    "dtype": analysis.dtype,
                },
            )
        )
    return strategies if strategies else None


def _build_strategy_prompt(
    analysis: AnalysisResult,
    focus: str,
    measures: list[str],
) -> str:
    measures_text = "\n".join(f"- {measure}" for measure in measures)
    top_ops = ", ".join(analysis.top_ops[:10]) if analysis.top_ops else "none"
    return (
        "Implement the optimization strategy below against the model execution.\n"
        "The focus and measures are already chosen — your job is the HOW: select the "
        "concrete API, signature, and guards; decide where to apply the patch and how "
        "to wire build_model; and verify functional equivalence. Preserve correctness "
        "while delivering the targeted latency reduction.\n\n"
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


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
