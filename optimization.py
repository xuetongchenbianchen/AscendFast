# Orchestrator. Unified recursive node (no first-round special case); benchmark
# latency judges 2x while profile only diagnoses; correctness is checked against
# the baseline golden, threaded through as baseline_mode.
from __future__ import annotations

import time

from analysis import analyze_profile
from apply import apply_discover, apply_wire, ensure_baseline_mode
from benchmark import run_real_benchmark
from correctness import run_correctness_test
from models import ExecutionMode, OperatorSpec, RunLedger
from operator_gen import generate_operator
from profile_runner import run_profile
from strategy import generate_optimization_strategies
from verify import (
    gate_apply,
    gate_operator,
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

        # 两阶段握手：apply-agent 无法直接调 operator-agent，由这里中介。
        #   ① apply_discover：fork + 让 apply-agent 读真实代码后二选一——直接改完，或
        #      请求一个自定义算子（返回 phase=operator_pending 的 mode，extra 挂 spec）。
        #   ② 若请求了算子：operator stage 生成+数值自检，gate_operator 把关。
        #   ③ apply_wire：在同一个已 fork 的 workspace 上把已验证算子接进 build_model()。
        # 算子是高风险产物：生成失败 ≠ 策略本身不可行，但 pending mode 没有 ChangeRecord，
        # 接不上就只能弃用这条策略（不递归进一个"没真改"的 mode）。串行执行，正好嵌在
        # 本就串行的策略循环里（算子编译共享 build_out/ 与全局安装路径，不可并行）。

        # ① discover
        with stage(ledger, "apply_discover", base_mode.uid) as st:
            st.value = apply_discover(strategy, base_mode)
            if st.value is None:
                st.fail("apply_discover returned no ExecutionMode")
            else:
                st.metadata = {"phase": (st.value.extra or {}).get("phase")}
        if not st.ok:
            print(f"{indent}    ❌ apply(discover) 失败，跳过")
            continue
        child = st.value

        # ②+③ 仅当 apply-agent 在 discover 阶段请求了自定义算子
        if (child.extra or {}).get("phase") == "operator_pending":
            op_spec = (child.extra or {}).get("pending_operator_spec") or {}
            print(f"{indent}    🔧 generate custom operator: {op_spec.get('op_name')}")
            with stage(ledger, "operator", base_mode.uid) as st:
                st.value = generate_operator(
                    OperatorSpec(**op_spec), strategy, analysis, base_mode
                )
                ok, reason = gate_operator(st.value)
                if not ok:
                    st.fail(reason)
            if not st.ok:
                print(f"{indent}    ⚠️  custom op unavailable ({st.reason}); 弃用此策略")
                continue
            operator_artifact = st.value
            print(f"{indent}    ✅ custom op ready: {operator_artifact.qualified_name}")

            with stage(ledger, "apply_wire", base_mode.uid) as st:
                st.value = apply_wire(strategy, child, operator_artifact)
                ok, reason = gate_apply(st.value, base_mode)
                if not ok:
                    st.fail(reason)
                else:
                    st.metadata = {
                        "strategy_kind": (strategy.extra or {}).get("kind"),
                        "applied_kind": st.value.change_log[-1].kind,
                        "custom_op": operator_artifact.qualified_name,
                    }
            if not st.ok:
                print(f"{indent}    ❌ apply(wire) 失败，跳过")
                continue
            child = st.value
        else:
            # apply-agent 直接用官方/eager 改完：discover 已含 ChangeRecord + 过了 forward
            # gate，这里只补 gate_apply（日志增长判定）与按-lever 归因，不再二次调 agent。
            with stage(ledger, "apply", base_mode.uid) as st:
                st.value = child
                ok, reason = gate_apply(child, base_mode)
                if not ok:
                    st.fail(reason)
                else:
                    st.metadata = {
                        "strategy_kind": (strategy.extra or {}).get("kind"),
                        "applied_kind": child.change_log[-1].kind,
                        "custom_op": None,
                    }
            if not st.ok:
                print(f"{indent}    ❌ apply 失败，跳过")
                continue

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

    # 一次优化 run 只会有一个 ledger
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
