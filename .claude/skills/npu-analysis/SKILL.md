---
name: npu-analysis
description: Analyze NPU (Ascend) profiling results. Use when interpreting profile_report.json, AnalysisResult, or investigating performance hotspots on Ascend NPU hardware.
---

# NPU Profile Analysis

解读 Ascend NPU profiling 数据与 AnalysisResult 所需的领域知识。

## ProfileResult & AnalysisResult 字段

**ProfileResult**（run_profile 的原始输出）:
- `profile_report`: dict，或指向 `profile_report.json` 的路径
- `latency_before` / `latency_after`: 端到端延迟（ms）
- `top_kernels`: dict 列表，含 `rank, name, op_type, device_time_ms, pct_total, roofline`

**AnalysisResult**（analyze_profile 的输出）:
- `total_latency`: 平均延迟（ms）
- `top_ops`: 按耗时排序的算子名
- `op_type_totals`: `{"matmul": {"device_time_ms": ..., "pct_total": ..., "call_count": ..., "kernel_count": ...}, ...}`
- `roofline_summary`: 按 roofline 类别（compute-bound / memory-bound / unknown）的耗时拆分
- `profile_findings`: 分析给出的自然语言建议

## Op Type 类别

常见的 `op_type` 取值：
- **matmul**: GEMM、稠密 linear 层。常占 transformer 推理的 40–60%。通过 layout（ND→NZ）、dtype（FP16/BF16）、避免小 batch/hidden 维度来优化。
- **flash_attention**: 融合 attention kernel。检查 mask 布局、序列长度 padding，以及是否命中快路径。
- **copy_cast**: 数据 layout 或 dtype 转换。目标：通过在算子间稳定 layout 来消除冗余拷贝。
- **rmsnorm** / **reduce**: 归一化与 reduction 算子。尽量融合（RMSNorm + Linear）。
- **elementwise**: Add、ReLU 等。通常很便宜，除非过于碎片化。

## Roofline 类别

- **compute-bound**: 算术强度高，受 FLOPs 限制。matmul 常落在这里。
- **memory-bound**: 受带宽限制。copy/cast、小型 reduction。
- **unknown**: profiler 无法分类；通常是极小算子或 profiler 噪声。

## 延迟统计

`latency_stats_ms`: `{mean, std, min, max, p50, p90, p99, noise_relative}`
`noise_relative > 0.05`: 方差偏高；在比较小幅 delta 前，用更多 iters 重新 profile。

## 典型热点模式

1. **Matmul 主导 (>35%)**: 关注 GEMM shape、layout 稳定性、避免小 tile。
2. **Flash_attention 飙高**: 检查 attention mask、seq len、融合路径是否启用。
3. **Copy_cast 开销 (>2%)**: 审计 dtype/layout 转换，删除冗余的。
4. **碎片化 elementwise**: 融合相邻算子以降低 launch 开销。
