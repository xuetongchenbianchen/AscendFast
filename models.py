from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class AppliedArtifact:
    """Agent 应用优化后留下的产物记录。"""
    kind: str                   # "patch" | "config" | "weight" | "graph" | "custom"
    paths: list[str]            # 相对于 adaptations/<model_id>/<strategy_uid>/ 的路径
    revert_cmd: str | None = None
    metadata: dict | None = None


@dataclass
class OptimizationStrategy:
    uid: str
    local_speedup_ratio: float
    measures: list[str]
    prompt_instruction: str
    extra: dict | None = None


@dataclass
class AnalysisResult:
    uid: str
    total_latency: float
    top_ops: list[str]
    hot_groups: dict[str, list[str]]
    extra: dict | None = None
    model_id: str | None = None
    device_kind: str | None = None
    device_name: str | None = None
    dtype: str | None = None
    profile_report_path: str | None = None
    latency_stats_ms: dict | None = None
    dataset: dict | None = None
    top_kernels: list[dict] = field(default_factory=list)
    op_type_totals: dict[str, dict] = field(default_factory=dict)
    roofline_summary: dict[str, float] = field(default_factory=dict)
    profile_findings: list[str] = field(default_factory=list)


@dataclass
class ExecutionMode:
    uid: str
    model_id: str
    strategy_uid: str
    artifacts: list[AppliedArtifact] = field(default_factory=list)
    correctness_passed: bool | None = None
    extra: dict | None = None


@dataclass
class ProfileResult:
    uid: str
    execution_mode_uid: str
    latency_before: float
    latency_after: float
    extra: dict | None = None
    profile_report_path: str | None = None
    profiler_output_dir: str | None = None
    profile_report: dict | None = None
