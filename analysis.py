# analysis-agent role (WHERE time goes, no fixes); raises instead of silently
# degrading when the agent runtime is unavailable.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_client import AGENT_ENABLED, call_agent_json
from models import AnalysisResult, ProfileResult

def analyze_profile(
    profile: ProfileResult,
) -> AnalysisResult:
    """
    将 ProfileResult 整理汇总为 AnalysisResult，
    结果可反馈至下一轮策略生成。

    Args:
        profile:  本轮 profile 结果

    Returns:
        AnalysisResult
    """
    # report:profile_report.json(dict)
    # report_path:AscendFast/profiles/profile_py_qwen_real
    report, report_path = _load_profile_report(profile)
    extra = profile.extra if isinstance(profile.extra, dict) else {}
    # top_kernels:
    #  [{"rank": 1,
    #       "name": "aclnnMatmul_MatMulCommon_MatMulV2",
    #       "op_type": "matmul",
    #       "shape_info": "",
    #       "device_time_ms": 8.493,
    #       "device_time_us": 8493.36,
    #       "call_count": 291,
    #       "avg_time_us": 29.19,
    #       "pct_total": 18.9,
    #       "cumulative_pct": 18.9,
    #       "roofline": "compute-bound",
    #       "optimization_priority": "HIGH",}, ...]
    top_kernels = _profile_top_kernels(report)
    top_ops = [str(kernel.get("name", "")) for kernel in top_kernels if kernel.get("name")]
    # hot_groups:
    # {
    #   "matmul": [
    #       "aclnnMatmul_MatMulCommon_MatMulV2",
    #       "another_matmul_kernel",
    #   ],
    #   "copy_cast": [
    #       "aclnnCast_CastAiCore_Cast",
    #   ],
    #   "reduce": [
    #       "aclnnMean_ReduceMeanAiCore_ReduceMean",
    #   ],
    # }
    hot_groups = _hot_groups_by_op_type(top_kernels)
    # op_type_totals =
    # {
    # "other": {"device_time_ms": 21.784, "pct_total": 48.6, "call_count": 2340, "kernel_count": 17},
    # "matmul": {"device_time_ms": 10.152, "pct_total": 22.6, "call_count": 510, "kernel_count": 3},
    # "copy_cast": {"device_time_ms": 4.505, "pct_total": 10.0, "call_count": 522, "kernel_count": 3},
    # "flash_attention": {"device_time_ms": 3.714, "pct_total": 8.3, "call_count": 72, "kernel_count": 1}
    # }
    op_type_totals = _op_type_totals(top_kernels)
    # roofline_summary
    # {
    #   "likely compute-bound": 21.706,
    #   "compute-bound": 13.866,
    #   "memory-bound": 9.223,
    #   "likely memory-bound": 0.078
    # }
    roofline_summary = _roofline_summary(top_kernels)
    latency_stats = report.get("latency_stats_ms") if isinstance(report.get("latency_stats_ms"), dict) else None
    dataset = _dataset_manifest(report)
    total_latency = _infer_total_latency_ms(profile, report, latency_stats)
    profile_findings = _profile_findings(report, op_type_totals, latency_stats, roofline_summary)

    analysis_extra = {
        "source_profile_uid": profile.uid,
        "execution_mode_uid": profile.execution_mode_uid,
        "latency_before": profile.latency_before,
        "latency_after": profile.latency_after,
        "total_device_time_ms": report.get("total_device_time_ms"),
        "profile_iters": report.get("profile_iters"),
        "total_kernels": report.get("total_kernels"),
        "optimization_summary": report.get("optimization_summary"),
        "analysis_readiness": report.get("analysis_readiness"),
        "artifacts": report.get("artifacts"),
    }
    if extra:
        analysis_extra["profile_extra"] = extra

    return AnalysisResult(
        uid=f"analysis:{profile.uid}",
        total_latency=total_latency,
        top_ops=top_ops,
        hot_groups=hot_groups,
        extra=analysis_extra,
        model_id=str(report.get("pretrained") or report.get("model") or ""),
        device_kind=_optional_str(report.get("device_kind")),
        device_name=_optional_str(report.get("device_name")),
        dtype=_optional_str(report.get("dtype")),
        profile_report_path=str(report_path) if report_path is not None else None,
        latency_stats_ms=latency_stats,
        dataset=dataset,
        top_kernels=top_kernels,
        op_type_totals=op_type_totals,
        roofline_summary=roofline_summary,
        profile_findings=profile_findings,
    )

def _load_profile_report(profile: ProfileResult) -> tuple[dict[str, Any], Path | None]:
    if isinstance(profile.profile_report, dict):
        return profile.profile_report, Path(profile.profile_report_path) if profile.profile_report_path else None

    for value in (profile.profile_report_path, profile.profiler_output_dir):
        if isinstance(value, (str, Path)):
            path = Path(value)
            if path.is_dir():
                path = path / "profile_report.json"
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8")), path

    extra = profile.extra if isinstance(profile.extra, dict) else {}
    direct_report = extra.get("profile_report")
    if isinstance(direct_report, dict):
        return direct_report, _path_from_extra(extra)

    for key in ("profile_report_path", "report_path", "output_path", "profile_report"):
        value = extra.get(key)
        if isinstance(value, (str, Path)):
            path = Path(value)
            if path.is_dir():
                path = path / "profile_report.json"
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8")), path

    profiler_output_dir = extra.get("profiler_output_dir")
    if isinstance(profiler_output_dir, (str, Path)):
        path = Path(profiler_output_dir) / "profile_report.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")), path

    return {}, None

def _path_from_extra(extra: dict[str, Any]) -> Path | None:
    for key in ("profile_report_path", "report_path", "output_path"):
        value = extra.get(key)
        if isinstance(value, (str, Path)):
            return Path(value)
    return None


def _profile_top_kernels(report: dict[str, Any]) -> list[dict[str, Any]]:
    raw = report.get("top_kernels")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _hot_groups_by_op_type(top_kernels: list[dict[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for kernel in top_kernels:
        op_type = str(kernel.get("op_type") or "unknown")
        name = str(kernel.get("name") or "")
        if name:
            groups.setdefault(op_type, []).append(name)
    return groups


def _op_type_totals(top_kernels: list[dict[str, Any]]) -> dict[str, dict]:
    totals: dict[str, dict[str, float | int]] = {}
    for kernel in top_kernels:
        op_type = str(kernel.get("op_type") or "unknown")
        item = totals.setdefault(
            op_type,
            {
                "device_time_ms": 0.0,
                "pct_total": 0.0,
                "call_count": 0,
                "kernel_count": 0,
            },
        )
        duration_ms = _float_or_zero(kernel.get("device_time_ms"))
        pct_total = _float_or_zero(kernel.get("pct_total"))
        call_count = int(_float_or_zero(kernel.get("call_count")))
        item["device_time_ms"] = float(item["device_time_ms"]) + duration_ms
        item["pct_total"] = float(item["pct_total"]) + pct_total
        item["call_count"] = int(item["call_count"]) + call_count
        item["kernel_count"] = int(item["kernel_count"]) + 1
    return {
        key: {
            "device_time_ms": round(float(value["device_time_ms"]), 6),
            "pct_total": round(float(value["pct_total"]), 6),
            "call_count": int(value["call_count"]),
            "kernel_count": int(value["kernel_count"]),
        }
        for key, value in sorted(totals.items(), key=lambda item: float(item[1]["device_time_ms"]), reverse=True)
    }

def _roofline_summary(top_kernels: list[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for kernel in top_kernels:
        roofline = str(kernel.get("roofline") or "unknown")
        summary[roofline] = summary.get(roofline, 0.0) + _float_or_zero(kernel.get("device_time_ms"))
    return {
        key: round(value, 6)
        for key, value in sorted(summary.items(), key=lambda item: item[1], reverse=True)
    }

def _dataset_manifest(report: dict[str, Any]) -> dict | None:
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    dataset = artifacts.get("dataset")
    return dict(dataset) if isinstance(dataset, dict) else None


def _infer_total_latency_ms(
    profile: ProfileResult,
    report: dict[str, Any],
    latency_stats: dict | None,
) -> float:
    if latency_stats:
        mean = _float_or_zero(latency_stats.get("mean"))
        if mean > 0.0:
            return mean
    if profile.latency_after > 0.0:
        return float(profile.latency_after)
    total_device_time = _float_or_zero(report.get("total_device_time_ms"))
    profile_iters = int(_float_or_zero(report.get("profile_iters")))
    if total_device_time > 0.0 and profile_iters > 0:
        return total_device_time / profile_iters
    return total_device_time


def _llm_profile_findings(
    report: dict[str, Any],
    op_type_totals: dict[str, dict],
    latency_stats: dict | None,
    roofline_summary: dict[str, float],
) -> list[str] | None:
    noise = _float_or_zero((latency_stats or {}).get("noise_relative"))
    prompt = (
        "You are an NPU performance DIAGNOSIS expert (not an optimizer).\n"
        "Describe WHERE time is spent and WHAT the bottleneck characteristics are.\n"
        "State objective findings only: which op types dominate, compute- vs "
        "memory-bound split, fragmentation (high call_count / low avg time), and "
        "measurement reliability. Do NOT propose fixes or say how to optimize.\n\n"
        f"model: {report.get('pretrained') or report.get('model') or 'unknown'}\n"
        f"device: {report.get('device_kind')} {report.get('device_name')}\n"
        f"dtype: {report.get('dtype')}\n"
        f"op_type_totals (top by device_time_ms): {json.dumps(op_type_totals, ensure_ascii=False)}\n"
        f"roofline_summary: {json.dumps(roofline_summary, ensure_ascii=False)}\n"
        f"latency_noise_relative: {noise:.4f}\n"
        f"top5_pct: {(report.get('optimization_summary') or {}).get('top5_pct')}\n\n"
        'Return JSON: {"hints": ["<finding1>", "<finding2>", ...]}'
    )
    result = call_agent_json("analysis-agent", prompt, timeout=1000)
    # 
    if not isinstance(result, dict):
        return None
    hints = result.get("hints")
    if isinstance(hints, list) and all(isinstance(h, str) for h in hints) and hints:
        return hints
    return None


def _profile_findings(
    report: dict[str, Any],
    op_type_totals: dict[str, dict],
    latency_stats: dict | None,
    roofline_summary: dict[str, float] | None = None,
) -> list[str]:
    """LLM diagnosis findings (no rule-based fallback).

    Requires the agent runtime to be enabled; raises otherwise so the failure
    is visible instead of silently degrading to empty findings.
    """
    try:
        if not AGENT_ENABLED:
            raise RuntimeError(
                f"analysis agent unavailable: AGENT_ENABLED={AGENT_ENABLED!r}"
            )
        return _llm_profile_findings(
            report, op_type_totals, latency_stats, roofline_summary or {}
        ) or []
    except Exception as exc:
        print(f"[analysis] findings failed (AGENT_ENABLED={AGENT_ENABLED!r}): {exc}")
        raise


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
