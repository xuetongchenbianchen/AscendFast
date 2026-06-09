from __future__ import annotations

from analysis import analyze_profile
from apply import apply_optimization, ensure_baseline_mode
from benchmark import run_real_benchmark
from correctness import run_correctness_test
from models import ExecutionMode
from profile import run_profile
from strategy import generate_optimization_strategies

# 优化链最大深度（baseline 为 depth=0）。
MAX_DEPTH = 5


def optimize(
    base_mode: ExecutionMode,
    baseline_latency: float | None = None,
    depth: int = 0,
    top_k: int = 5,
    baseline_mode: ExecutionMode | None = None,
) -> tuple[ExecutionMode, float]:
    """在 base_mode 快照之上迭代叠加优化，返回 (最优 mode, 其延迟 ms)。

    baseline 即 depth=0 的 base_mode，与深层子节点完全同构：每个节点进来都先
    benchmark（定延迟标尺）+ profile→analyze（诊断热点、生成策略），再 fork 出子
    mode 递归。胜出的子 mode 作为下一轮 base_mode，因此优化是叠加而非每轮从原始
    模型重来。

    延迟口径：判断加速比（2x）用 benchmark 的真实数据集延迟，不用 profile 的模拟
    数据延迟——profile 只负责诊断与生成策略。

    正确性口径：每个 fork 都与 baseline_mode 的金标准比（贯穿递归传递），而非与
    父 mode 比，避免误差沿优化链累积。
    """
    if baseline_mode is None:
        baseline_mode = base_mode               # depth=0：base_mode 即 baseline 根

    # ① 延迟标尺：每个 mode（含 baseline）进来先用真实数据集 benchmark 测 forward 延迟
    latency = run_real_benchmark(base_mode)
    if baseline_latency is None:
        baseline_latency = latency          # base_mode 是 baseline 时自己定标尺
    if latency <= baseline_latency / 2 or depth >= MAX_DEPTH:
        return base_mode, latency           # 达成 2x 或到底，止步

    # ② 诊断 + 策略生成（唯一一处）：profile→analyze 只为定位热点、产出策略
    analysis = analyze_profile(run_profile(base_mode))
    strategies = generate_optimization_strategies(analysis, top_k)
    best_mode, best_lat = base_mode, latency
    for strategy in strategies[:top_k]:
        child = run_correctness_test(apply_optimization(strategy, base_mode), baseline_mode)
        if not child.correctness_passed:
            continue
        cand_mode, cand_lat = optimize(child, baseline_latency, depth + 1, top_k, baseline_mode)
        if cand_lat < best_lat:
            best_mode, best_lat = cand_mode, cand_lat
            if best_lat <= baseline_latency / 2:
                break                       # 提前命中 2x，逐层冒泡返回
    return best_mode, best_lat


def run(model_id: str, model_dir: str, top_k: int = 5) -> tuple[ExecutionMode, float]:
    """顶层：物化 baseline → 从 depth=0 开始统一迭代。无任何首轮特例。"""
    baseline = ensure_baseline_mode(model_id, model_dir)
    baseline.correctness_passed = True      # 原始模型按定义正确
    return optimize(baseline, top_k=top_k)
