"""真实领域数据集上的离线 benchmark：测一个 ExecutionMode 的 forward（prefill）延迟。

「加速比标尺」一侧：用真实 ShareGPT 数据集裸计时（不走 profiler），
其 mean 延迟是判定 2x 的唯一依据。通过 build_model() 统一入口加载。

定位（与 profile_runner.run_profile 区分）：
- run_profile        —— 用模拟数据 + torch_npu profiler 做**诊断**，产 top_kernels 等。
- run_real_benchmark —— 用真实 ShareGPT 数据集测**纯推理延迟**，作为加速比标尺。

benchmark 只关心一个指标：forward 延迟。所以这里**不走 profiler**，也不产出
records / top_kernels / op_type_totals 这类诊断物——profiler 的插桩会污染计时。
计时是最朴素的可信形态：

    warmup → (synchronize → perf_counter → forward → synchronize → perf_counter) × N → 统计

在 Ascend NPU 上 forward() 是异步下发，计时前后必须 synchronize，否则测到的是
"下发耗时"而非"执行耗时"。两端（baseline / 优化版）都通过 build_model() 统一入口
加载，唯一区别是被测的模型本身。

加速比用法：
    base = run_real_benchmark(baseline_mode)
    opt  = run_real_benchmark(optimized_mode)
    speedup = base / opt          # >1 表示变快

只测 forward（prefill），是本项目 benchmark 环节当前的范围。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import median, pstdev
from typing import Any

from dataset import load_prompt_dataset, tokenize_prompts
from models import ExecutionMode
from workspace_loader import load_build_model
from device_utils import (
    device_spec_for,
    import_torch,
    release_device_memory,
    run_forward,
    synchronize,
)

_PROJECT_ROOT = Path(__file__).parent
# 真实领域 benchmark 数据集，由 data/sharegpt_to_jsonl.py 从 ShareGPT 转出。
# 与 profile 的模拟数据 data/prompts_real.jsonl 刻意区分。
_BENCHMARK_DATASET = _PROJECT_ROOT / "data" / "prompts_sharegpt.jsonl"

# benchmark 比 profile 多测几轮，让 mean/median/p* 更稳。
_WARMUP_ITERS = 5
_BENCH_ITERS = 30


def run_real_benchmark(
    mode: ExecutionMode,
    dataset_path: str | None = None,
    *,
    max_samples: int = 64,
    max_input_tokens: int = 512,
    warmup_iters: int = _WARMUP_ITERS,
    bench_iters: int = _BENCH_ITERS,
) -> float:
    """用真实数据集测 mode 的 forward 端到端平均延迟（ms）。

    Args:
        mode:             已性通过正确测试的执行模式（correctness_passed=True）。
        dataset_path:     真实领域数据集 jsonl；None 时用默认 ShareGPT 转出文件。
        max_samples:      取数据集前 N 条 prompt 作为一个 batch。
        max_input_tokens: tokenize 的 max_length / padding 上限（prefill 序列长度）。
        warmup_iters:     预热轮数（不计入统计）。
        bench_iters:      计时轮数（取 mean 作为返回值，median/p* 一并落盘）。

    Returns:
        forward 平均延迟（ms）。
    """
    if mode.correctness_passed is not True:
        raise ValueError(
            f"run_real_benchmark requires correctness_passed=True, got {mode.correctness_passed}"
        )

    ds_path = Path(dataset_path) if dataset_path else _BENCHMARK_DATASET
    if not ds_path.exists():
        raise FileNotFoundError(
            f"benchmark 数据集不存在: {ds_path}\n"
            "先用 data/sharegpt_to_jsonl.py 从 ShareGPT 生成，或显式传 dataset_path。"
        )

    torch = import_torch()
    model, tokenizer = load_build_model(mode)
    device, _ = device_spec_for(model)               # 模型在哪就用哪，不二次搬运
    model = model.eval()

    try:
        # 真实 prompt → 按真实 token 长度排序后再 tokenize，减少 batch 内 padding 失真。
        prompts = load_prompt_dataset(ds_path, max_samples=max_samples).prompts
        prompts = _sort_by_token_length(tokenizer, prompts)
        inputs = tokenize_prompts(
            torch, tokenizer, prompts, device=device.device, max_length=max_input_tokens
        )

        samples_ms = _time_forward(
            torch, model, inputs, device.kind,
            warmup_iters=warmup_iters, bench_iters=bench_iters,
        )
        stats = _latency_stats(samples_ms)

        _write_report(
            mode, ds_path, device, stats, samples_ms,
            num_prompts=len(prompts), max_input_tokens=max_input_tokens,
        )
    finally:
        # 本节点用完即释放：不然递归深处每个父帧都还压着一个模型，显存只增不减。
        del model
        release_device_memory(torch, device.kind)

    if stats["mean"] <= 0.0:
        raise RuntimeError("benchmark produced non-positive latency.")
    return stats["mean"]


# --------------------------------------------------------------------------- #
# 裸计时：warmup → (sync → perf_counter → forward → sync → perf_counter) × N
# 不走 profiler，不产 kernel records；这是 benchmark 的可信延迟形态。
# --------------------------------------------------------------------------- #
def _time_forward(
    torch: Any,
    model: Any,
    inputs: dict[str, Any],
    device_kind: str,
    *,
    warmup_iters: int,
    bench_iters: int,
) -> list[float]:
    samples_ms: list[float] = []
    with torch.no_grad():
        for _ in range(max(warmup_iters, 0)):
            run_forward(model, inputs)
            synchronize(torch, device_kind)
        for _ in range(max(bench_iters, 1)):
            synchronize(torch, device_kind)        # 隔离上一轮残留
            started = time.perf_counter()
            run_forward(model, inputs)
            synchronize(torch, device_kind)        # 等 NPU 真正算完再停表
            samples_ms.append((time.perf_counter() - started) * 1000.0)
    return samples_ms


def _latency_stats(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        return {"mean": 0.0}
    ordered = sorted(samples_ms)
    n = len(ordered)
    mean = sum(ordered) / n
    lo, hi = ordered[0], ordered[-1]
    return {
        "mean": mean,
        "median": median(ordered),
        "std": pstdev(ordered) if n > 1 else 0.0,
        "min": lo,
        "max": hi,
        "p90": ordered[min(n - 1, int(round(0.90 * (n - 1))))],
        "p99": ordered[min(n - 1, int(round(0.99 * (n - 1))))],
        "noise_relative": (hi - lo) / mean if mean > 0 else 0.0,
        "samples": n,
    }


def _write_report(
    mode: ExecutionMode,
    ds_path: Path,
    device: Any,
    stats: dict[str, float],
    samples_ms: list[float],
    *,
    num_prompts: int,
    max_input_tokens: int,
) -> None:
    out_dir = Path(mode.workspace_dir) / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "mode_uid": mode.uid,
        "profile_mode": "forward",
        "device": {"kind": device.kind, "name": device.name},
        "dataset": {"path": str(ds_path), "num_prompts": num_prompts},
        "input_shape": [num_prompts, max_input_tokens],
        "latency_stats_ms": stats,
        "latency_samples_ms": [round(v, 6) for v in samples_ms],
    }
    (out_dir / "benchmark_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _sort_by_token_length(tokenizer: Any, prompts: tuple[str, ...]) -> tuple[str, ...]:
    """按真实 token 数升序排，让同 batch 长度接近、padding 更少。失败则原样返回。"""
    try:
        return tuple(sorted(prompts, key=lambda p: len(tokenizer(p)["input_ids"])))
    except Exception:
        return prompts
