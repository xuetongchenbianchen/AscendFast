from __future__ import annotations

import time

from analysis import analyze_profile
from apply import apply_optimization, ensure_baseline_mode
from benchmark import run_real_benchmark
from correctness import run_correctness_test
from models import ExecutionMode, RunLedger
from profile_runner import run_profile
from strategy import generate_optimization_strategies
from verify import (
    gate_apply,
    gate_strategy,
    set_current_ledger,
    stage,
    write_ledger,
)

# 优化链最大深度（baseline 为 depth=0）。
MAX_DEPTH = 5


def optimize(
    base_mode: ExecutionMode,
    ledger: RunLedger,
    baseline_latency: float | None = None,
    depth: int = 0,
    top_k: int = 5,
    baseline_mode: ExecutionMode | None = None,
    max_depth: int = MAX_DEPTH,
) -> tuple[ExecutionMode, float]:
    """在 base_mode 快照之上迭代叠加优化，返回 (最优 mode, 其延迟 ms)。

    baseline 即 depth=0 的 base_mode，与深层子节点完全同构：每个节点进来都先
    benchmark（定延迟标尺）+ profile→analyze（诊断热点、生成策略），再 fork 出子
    mode 递归。胜出的子 mode 作为下一轮 base_mode，因此优化是叠加而非每轮从原始
    模型重来。

    每个环节都包在 stage() 里：成败记进 ledger、异常被吞掉（不再炸穿整条 run）。
    两道门禁（gate_strategy / gate_apply）把此前隐式缺失的判断显式化。

    延迟口径：判断加速比（2x）用 benchmark 的真实数据集延迟，不用 profile 的模拟
    数据延迟——profile 只负责诊断与生成策略。

    正确性口径：每个 fork 都与 baseline_mode 的金标准比（贯穿递归传递），而非与
    父 mode 比，避免误差沿优化链累积。
    """
    if baseline_mode is None:
        baseline_mode = base_mode               # depth=0：base_mode 即 baseline 根

    indent = "  " * depth
    print(f"\n{indent}[depth={depth}] ▶ {base_mode.uid}")

    # ① 延迟标尺：每个 mode（含 baseline）进来先用真实数据集 benchmark 测 forward 延迟
    print(f"{indent}  📊 benchmark ...")
    with stage(ledger, "benchmark", base_mode.uid) as st:
        st.value = run_real_benchmark(base_mode)
    if not st.ok:
        # 测不出延迟的候选记为无穷差，绝不能被选成 best（哪怕 baseline_latency 已有值）。
        ledger.stop_reason = ledger.stop_reason or "stage_failed:benchmark"
        return base_mode, float("inf")

    latency = st.value
    # baseline（首次进入）只负责定标尺；child 才判断是否达成 2x——分开写避免
    # "刚设完 baseline_latency 又立刻拿它判 2x"的误读。
    if baseline_latency is None:
        baseline_latency = latency
        ledger.baseline_latency = baseline_latency
        print(f"{indent}  ✅ baseline latency = {latency:.4f} ms")
    else:
        speedup = baseline_latency / latency
        print(f"{indent}  ✅ latency = {latency:.4f} ms  ({speedup:.2f}x vs baseline)")
        if latency <= baseline_latency / 2:
            print(f"{indent}  🎉 达成 2x 加速，提前终止！")
            ledger.stop_reason = "reached_2x"
            return base_mode, latency
    if depth >= max_depth:
        print(f"{indent}  ⛔ 达到最大深度 {max_depth}，停止递归")
        ledger.stop_reason = ledger.stop_reason or "max_depth"
        return base_mode, latency

    # ② 诊断 + 策略生成（唯一一处）：profile→analyze 只为定位热点、产出策略
    print(f"{indent}  🔬 profile ...")
    with stage(ledger, "profile", base_mode.uid) as st:
        st.value = run_profile(base_mode)
    if not st.ok:
        print(f"{indent}  ❌ profile 失败，跳过")
        ledger.stop_reason = ledger.stop_reason or "stage_failed:profile"
        return base_mode, latency
    profile_result = st.value

    print(f"{indent}  🧠 analyze ...")
    with stage(ledger, "analyze", base_mode.uid) as st:
        st.value = analyze_profile(profile_result)
    if not st.ok:
        print(f"{indent}  ❌ analyze 失败，跳过")
        ledger.stop_reason = ledger.stop_reason or "stage_failed:analyze"
        return base_mode, latency
    analysis = st.value

    print(f"{indent}  💡 generate strategies (top_k={top_k}) ...")
    with stage(ledger, "strategy", base_mode.uid) as st:
        st.value = generate_optimization_strategies(analysis, top_k)
        ok, reason = gate_strategy(st.value)
        if not ok:
            st.fail(reason)                 # 空策略列表门禁：不再静默零循环
    if not st.ok:
        print(f"{indent}  ❌ 无可用策略，停止")
        ledger.stop_reason = ledger.stop_reason or "no_strategies"
        return base_mode, latency
    strategies = st.value
    print(f"{indent}  📋 生成 {len(strategies)} 条策略")

    best_mode, best_lat = base_mode, latency
    for i, strategy in enumerate(strategies[:top_k]):
        print(f"{indent}  ⚙️  apply [{i+1}/{min(len(strategies), top_k)}]: {strategy.uid}")
        # apply：fork + agent 叠加优化。gate_apply 拦下 None-record（agent 失败），
        # 这次没产出可叠加的优化就跳过，不再递归进一个"没真改"的 mode。
        with stage(ledger, "apply", base_mode.uid) as st:
            st.value = apply_optimization(strategy, base_mode)
            ok, reason = gate_apply(st.value, base_mode)
            if not ok:
                st.fail(reason)
        if not st.ok:
            print(f"{indent}    ❌ apply 失败，跳过")
            continue
        child = st.value

        print(f"{indent}    🧪 correctness test ...")
        with stage(ledger, "correctness", child.uid) as st:
            st.value = run_correctness_test(child, baseline_mode)
        if not st.ok:
            print(f"{indent}    ❌ 正确性测试失败，跳过")
            continue
        child = st.value
        if not child.correctness_passed:
            print(f"{indent}    ❌ 输出不一致，跳过")
            continue
        print(f"{indent}    ✅ 正确性通过，递归优化 ...")

        cand_mode, cand_lat = optimize(
            child, ledger, baseline_latency, depth + 1, top_k, baseline_mode, max_depth
        )
        if cand_lat < best_lat:
            best_mode, best_lat = cand_mode, cand_lat
            if best_lat <= baseline_latency / 2:
                ledger.stop_reason = "reached_2x"
                break                       # 提前命中 2x，逐层冒泡返回
    print(f"{indent}  → depth={depth} 最优: {best_mode.uid}  {best_lat:.4f} ms")
    if ledger.stop_reason is None:
        ledger.stop_reason = "exhausted"
    return best_mode, best_lat


def run(
    model_id: str,
    model_dir: str,
    top_k: int = 5,
    max_depth: int = MAX_DEPTH,
) -> tuple[ExecutionMode, float]:
    """顶层：物化 baseline → 从 depth=0 开始统一迭代。无任何首轮特例。

    创建本次 run 的 RunLedger 并设为当前（agent_client 据此记 agent_call 事件），
    递归结束后写 best_mode/best_latency 并落盘——这一次探索了哪棵树、为什么停，
    从此有处可寻。
    """
    print("🚀 物化 baseline ...")
    baseline = ensure_baseline_mode(model_id, model_dir)
    baseline.correctness_passed = True      # 原始模型按定义正确

    ledger = RunLedger(run_uid=f"run:{model_id}:{int(time.time())}", model_id=model_id)
    set_current_ledger(ledger)
    try:
        best_mode, best_lat = optimize(baseline, ledger, top_k=top_k, max_depth=max_depth)
        ledger.best_mode_uid = best_mode.uid
        ledger.best_latency = best_lat
        return best_mode, best_lat
    finally:
        write_ledger(ledger)
        set_current_ledger(None)
