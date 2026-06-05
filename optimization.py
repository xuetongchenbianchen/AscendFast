from __future__ import annotations

from analysis import (
    _dataset_manifest,
    _hot_groups_by_op_type,
    _infer_total_latency_ms,
    _load_profile_report,
    _op_type_totals,
    _optional_str,
    _profile_findings,
    _profile_top_kernels,
    _roofline_summary,
)

from models import (
    OptimizationStrategy, AnalysisResult, ExecutionMode, ProfileResult,
)
from strategy import generate_optimization_strategies


def apply_optimization(
    strategy: OptimizationStrategy,
    model_id: str,
) -> ExecutionMode:
    """
    将 strategy.prompt_instruction 传给 Agent，Agent 修改模型后
    返回对应的 ExecutionMode（correctness_passed=None）。
    
    Args:
        strategy:  待执行的优化策略
        model_id:   目标模型标识
    
    Returns:
        ExecutionMode，correctness_passed 尚未填写
    """

def run_correctness_test(
    mode: ExecutionMode,
) -> ExecutionMode:
    """
    对 ExecutionMode 执行正确性验证，填写 correctness_passed 字段。
    测试不通过时直接丢弃（调用方应跳过后续步骤）。
    
    Args:
        mode:  待测试的执行模式
    
    Returns:
        填写了 correctness_passed 的 ExecutionMode
    """

def run_profile(
    mode: ExecutionMode,
) -> ProfileResult:
    """
    对通过正确性测试的 ExecutionMode 执行性能 profile。
    
    Args:
        mode:  correctness_passed=True 的执行模式
    
    Returns:
        ProfileResult，包含优化前后延迟数据
    
    Raises:
        ValueError: 若 mode.correctness_passed != True
    """


def optimization_pipeline(
    strategies: list[OptimizationStrategy],
    model_id: str,
    top_k: int = 5,
    baseline_latency: float | None = None,
    _depth: int = 0,
) -> ExecutionMode | None:
    if _depth >= 5 or top_k <= 0:
        return None
    best: tuple[float, ExecutionMode] | None = None  # (latency, mode)
    for strategy in strategies[:top_k]:
        mode = apply_optimization(strategy, model_id)
        mode = run_correctness_test(mode)
        if not mode.correctness_passed:
            continue
        current_latency = run_real_benchmark(model_id)
        # 记录当前最优
        if best is None or current_latency < best[0]:
            best = (current_latency, mode)
        # 达到 2x 加速比，提前返回
        if baseline_latency and current_latency <= baseline_latency / 2:
            return mode
        profile = run_profile(mode)
        analysis = analyze_profile(profile)
        next_strategies = generate_optimization_strategies(analysis, top_k)
        result = optimization_pipeline(
            next_strategies, model_id, top_k, baseline_latency, _depth + 1,
        )
        if result is not None:
            return result
    # 未达到 2x，返回本轮最优（可能为 None，即全部正确性不通过）
    return best[1] if best else None
