---
name: npu-analysis
description: Analyze NPU (Ascend) profiling results. Use when interpreting profile_report.json, AnalysisResult, or investigating performance hotspots on Ascend NPU hardware.
---

# NPU Profile Analysis

Domain knowledge for interpreting Ascend NPU profiling data and AnalysisResult.

## ProfileResult & AnalysisResult Fields

**ProfileResult** (raw output from run_profile):
- `profile_report`: dict or path to `profile_report.json`
- `latency_before` / `latency_after`: end-to-end latency (ms)
- `top_kernels`: list of dicts with `rank, name, op_type, device_time_ms, pct_total, roofline`

**AnalysisResult** (from analyze_profile):
- `total_latency`: mean latency (ms)
- `top_ops`: operator names ranked by time
- `op_type_totals`: `{"matmul": {"device_time_ms": ..., "pct_total": ..., "call_count": ..., "kernel_count": ...}, ...}`
- `roofline_summary`: time breakdown by roofline category (compute-bound / memory-bound / unknown)
- `profile_findings`: natural-language suggestions from analysis

## Op Type Categories

Common `op_type` values:
- **matmul**: GEMM, dense linear layers. Often 40–60% of transformer inference. Optimize via layout (ND→NZ), dtype (FP16/BF16), and avoiding small batch/hidden dims.
- **flash_attention**: Fused attention kernel. Check mask layout, sequence length padding, and whether you're hitting the fast path.
- **copy_cast**: Data layout or dtype conversions. Target: eliminate redundant copies by stabilizing layout across ops.
- **rmsnorm** / **reduce**: Normalization and reduction ops. Fuse when possible (RMSNorm + Linear).
- **elementwise**: Add, ReLU, etc. Usually cheap unless very fragmented.

## Roofline Categories

- **compute-bound**: Arithmetic intensity high; limited by FLOPs. Matmul often lands here.
- **memory-bound**: Limited by bandwidth. Copy/cast, small reductions.
- **unknown**: Profiler couldn't classify; usually tiny ops or profiler artifacts.

## Latency Stats

`latency_stats_ms`: `{mean, std, min, max, p50, p90, p99, noise_relative}`  
`noise_relative > 0.05`: High variance; re-profile with more iters before comparing small deltas.

## Typical Hotspot Patterns

1. **Matmul dominance (>35%)**: Focus on GEMM shape, layout stability, avoiding small tiles.
2. **Flash_attention spike**: Check attention mask, seq len, fused path enabled.
3. **Copy_cast overhead (>2%)**: Audit dtype/layout conversions; remove redundant ones.
4. **Fragmented elementwise**: Fuse adjacent ops to reduce launch overhead.
