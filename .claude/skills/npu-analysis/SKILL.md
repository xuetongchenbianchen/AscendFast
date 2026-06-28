---
name: npu-analysis
description: Analyze NPU (Ascend) profiling results and diagnose performance bottlenecks. Use this skill whenever you need to interpret profile_report.json, AnalysisResult data, investigate performance hotspots, understand roofline analysis, diagnose kernel inefficiencies, or translate profiling metrics into actionable insights for Ascend NPU hardware. Also use when the user asks "why is my model slow", "what's the bottleneck", or mentions profiling data interpretation.
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
- `top_ops`: 按耗时排序的算子名列表（如 `["aten::linear", "aten::softmax", ...]`）
- `op_type_totals`: 按算子类型聚合的统计
  ```python
  {
    "matmul": {
      "device_time_ms": 18.5,
      "pct_total": 40.9,
      "call_count": 128,
      "kernel_count": 64
    },
    ...
  }
  ```
- `roofline_summary`: 按 roofline 类别拆分的耗时
  ```python
  {
    "compute_utilization": 0.42,        # 算力利用率（0-1）
    "memory_bandwidth_utilization": 0.68,
    "compute_bound_pct": 45.2,          # compute-bound 算子占比
    "memory_bound_pct": 38.1,
    "unknown_pct": 16.7
  }
  ```
- `profile_findings`: 自然语言建议列表（如 `["launch-bound: 328 small kernels", ...]`）
- `latency_stats_ms`: 延迟统计 `{mean, std, min, max, p50, p90, p99, noise_relative}`

## Op Type 类别详解

### matmul（GEMM / 稠密矩阵乘法）
**典型占比**: transformer 推理的 40–60%
**关键指标**:
- `device_time_ms`: 绝对耗时
- `call_count` vs `kernel_count`: 理想情况下应该接近（一次调用 = 一个 kernel launch）

**常见问题**:
1. **Layout 转换开销**: matmul 前后有 `aten::copy_` 或 `aten::contiguous`，说明权重不是 NZ 格式
2. **Shape 对齐**: hidden_dim / seq_len 不是 16 的倍数，导致 tile 利用率低
3. **Dtype 混用**: 输入是 fp16 但权重残留 fp32，触发隐式转换

**诊断路径**:
- 占比 >45%：正常，matmul 是计算主体
- 占比 >60%：可能其他部分已优化好，或 matmul 本身有问题（检查 layout/对齐）
- 占比 <30%：非典型，检查是否 attention/copy 占比异常高

### flash_attention（融合 attention kernel）
**典型占比**: 5–15%（启用融合时）；未融合时会拆成多个 matmul/softmax
**关键指标**:
- 是否出现在 `top_ops` 里：出现 = 走了融合路径
- 未出现但有 `aten::softmax` + 多个 `aten::matmul`：说明走了 naive 实现

**常见问题**:
1. **Mask 布局不兼容**: attention mask 导致回退到 eager 实现
2. **Seq len 不对齐**: 未 pad 到 128/256，kernel 效率低
3. **配置未开启**: `attn_implementation` 没设成融合后端

**诊断路径**:
- 看到 `flash_attention` 且占比 5-15%：健康
- 没看到但 `softmax` 占比 >3%：说明 attention 走了 naive 路径，优先优化

### copy_cast（数据搬运 / dtype/layout 转换）
**典型占比**: 理想情况 <2%；超过 5% 说明有问题
**关键指标**:
- `call_count`: 调用次数；过高说明碎片化严重
- `pct_total`: 占比；>5% 值得优化

**常见问题**:
1. **冗余转换**: 同一数据在 ND ↔ NZ 间反复转
2. **Dtype 不一致**: eager 代码里散落 `.to(torch.float32)` / `.half()`
3. **Per-step 转换**: 权重每步都转一次，而非加载时一次性转

**诊断路径**:
- 占比 2-5%：轻微问题，可优化但不紧急
- 占比 >5%：优先级高，审计 forward 里的 `.to()` / `.contiguous()` 调用
- 占比 >10%：严重，可能 layout 策略根本性错误

### rmsnorm / layernorm / reduce
**典型占比**: 3–8%
**关键指标**:
- `kernel_count`: 每层 norm 应该是 1 个 kernel；过多说明未融合
- 是否紧挨着 `aten::linear`：是 → 可融合成 RMSNormLinear

**常见问题**:
1. **未融合**: RMSNorm 和后续 Linear 分离
2. **Reduction 轴不优**: 沿非连续维度 reduce
3. **中间落 CPU**: 自定义层里有 `.cpu()` / `.numpy()`

**诊断路径**:
- 占比 3-6%：正常
- 占比 >8%：检查是否可融合，或有冗余 norm 层

### elementwise（逐元素算子：Add / Mul / ReLU / GELU）
**典型占比**: 2–5%
**关键指标**:
- `call_count` vs `kernel_count`：比值接近 1 = 好；远大于 1 = 碎片化

**常见问题**:
1. **Launch overhead**: 每个小算子独立 launch，开销累积
2. **未融合**: Add + Mul + GELU 可融合但分离执行

**诊断路径**:
- 占比 <5% 且 kernel_count 合理：正常
- 占比 >5% 或 kernel_count 过多：考虑 graph_rewrite 融合

## Roofline 类别详解

### compute-bound（算力瓶颈）
**含义**: 算术强度高，受 FLOPs 限制；NPU 算力接近饱和
**典型算子**: matmul、convolution
**优化方向**: 提升 GEMM 利用率（shape 对齐、layout、dtype）

### memory-bound（带宽瓶颈）
**含义**: 受内存带宽限制；算力未饱和，瓶颈在数据搬运
**典型算子**: copy/cast、小型 reduction、elementwise
**优化方向**: 减少数据搬运（消除冗余 copy、融合算子）

### unknown（未分类）
**含义**: profiler 无法归类；通常是极小算子或 profiler 噪声
**典型占比**: <20%
**处理**: 占比 >20% 说明 profiler 质量差或模型有大量非标准算子

### Roofline Summary 的解读

```python
roofline_summary = {
    "compute_utilization": 0.35,        # 算力利用率 35%
    "memory_bandwidth_utilization": 0.68,
    "compute_bound_pct": 28.5,
    "memory_bound_pct": 55.2,
    "unknown_pct": 16.3
}
```

**诊断**:
- `compute_utilization < 0.4` 且 `memory_bound_pct > 50%`
  → **Launch-bound**：kernel 太多太小，launch overhead 占主导
  → 推荐 `graph_rewrite`（torch.compile / 图模式）

- `compute_utilization > 0.6` 且 `compute_bound_pct > 60%`
  → **Compute-bound**：算力饱和，优化空间有限
  → 推荐 `loading_time`（权重预处理）或微调 GEMM shape

- `memory_bandwidth_utilization > 0.7` 且 `copy_cast` 占比高
  → **Memory-bound**：带宽饱和，数据搬运过多
  → 推荐消除冗余 copy、稳定 layout

## 延迟统计解读

```python
latency_stats_ms = {
    "mean": 45.2,
    "std": 2.1,
    "min": 42.8,
    "max": 51.3,
    "p50": 44.9,
    "p90": 47.8,
    "p99": 50.2,
    "noise_relative": 0.046  # std / mean
}
```

**噪声判定**:
- `noise_relative < 0.03`：低噪声，数据可信
- `0.03 ≤ noise_relative < 0.05`：中等噪声，可接受
- `noise_relative ≥ 0.05`：高噪声，在比较小幅 delta 前用更多 iters 重新 profile

**异常检测**:
- `max / min > 1.2`：有异常值（如首次 warmup、后台干扰）
- `p99 - p50 > mean * 0.1`：长尾严重，可能有偶发的慢路径

## 诊断决策树

### 入口：从 roofline_summary 开始

```
roofline_summary.compute_utilization < 0.4?
├─ Yes → Launch-bound
│   ├─ kernel 数量 >200?
│   │   ├─ Yes → 推荐 graph_rewrite（整模型 compile）
│   │   └─ No → 检查是否有大量小 elementwise，考虑局部融合
│   └─ copy_cast 占比 >5%?
│       ├─ Yes → 同时推荐 loading_time（稳定 layout）
│       └─ No → 纯 launch overhead，graph_rewrite 优先级最高
│
└─ No → Compute/Memory-bound，分析 op_type_totals
    ├─ matmul 占比 >50%?
    │   ├─ Yes → Matmul 主导
    │   │   ├─ copy_cast 占比 >3%? → 推荐 loading_time（权重预转 NZ）
    │   │   └─ 否则 → matmul 已优化好，找次要瓶颈
    │   └─ No → 继续
    ├─ flash_attention 不在 top_ops 但 softmax >2%?
    │   └─ Yes → Attention 走了 naive 路径，推荐 operator_fusion
    └─ rmsnorm >5% 且紧挨 linear?
        └─ Yes → 推荐 forward_patch（融合 RMSNormLinear）
```

### 场景 1：Launch-bound Profile

**特征**:
- `compute_utilization < 0.4`
- `profile_findings` 含 "launch-bound" 或 "many small kernels"
- kernel 数量 >200

**推荐策略**:
1. **主策略** (`graph_rewrite`): 整模型 compile / 图模式，预期 1.1-1.4× 加速
2. **辅助策略** (`loading_time`): 若 copy_cast >5%，同时预转权重 layout

**示例 profile**:
```python
{
  "compute_utilization": 0.35,
  "kernel_count": 328,
  "copy_cast_pct": 8.0,
  "findings": ["launch-bound: 328 small kernels with low compute utilization"]
}
```

### 场景 2：Attention 未融合

**特征**:
- `flash_attention` 不在 `top_ops` 里
- `aten::softmax` 占比 >2%
- `op_type_totals` 里有多个 attention 相关的 matmul

**推荐策略**:
1. **主策略** (`operator_fusion`): 设置 `attn_implementation="flash_attention"`，预期 1.2-1.5× 加速
2. **检查项**: 确认 transformers 版本支持、mask 布局兼容

**示例 profile**:
```python
{
  "top_ops": ["aten::linear", "aten::softmax", "aten::matmul", "aten::transpose"],
  "op_type_totals": {
    "matmul": {"pct_total": 42.3},
    "softmax": {"pct_total": 4.8}  # 异常高
  }
}
```

### 场景 3：Copy/Cast 开销高

**特征**:
- `copy_cast` 占比 >5%
- `profile_findings` 含 "frequent ND↔NZ conversions"

**推荐策略**:
1. **主策略** (`loading_time`): 预转权重为 NZ、稳定 dtype，预期 1.08-1.12× 加速
2. **辅助**: 审计 forward 里的 `.to()` / `.contiguous()` 调用

**示例 profile**:
```python
{
  "op_type_totals": {
    "copy_cast": {"device_time_ms": 4.2, "pct_total": 9.3, "call_count": 156}
  },
  "findings": ["frequent ND↔NZ layout conversions detected"]
}
```

### 场景 4：单算子瓶颈

**特征**:
- 某个 op 占比 >10%，其他分散
- 无明显的 launch-bound 特征

**推荐策略**:
1. **主策略** (`forward_patch`): 针对该算子优化（融合、自定义 kernel）
2. **判断**: 官方有融合版本吗？有 → 直接用；无 → 考虑自定义算子

**示例 profile**:
```python
{
  "top_ops": ["aten::native_layer_norm", "aten::linear", "aten::add"],
  "op_type_totals": {
    "rmsnorm": {"pct_total": 12.3}  # 单个算子占比异常高
  }
}
```

## 典型反模式识别

### 反模式 1：过度优化已优化的部分
**症状**: matmul 已占 45%、无明显问题，策略仍针对 matmul
**问题**: 优化空间有限，应转向次要瓶颈
**建议**: 检查 copy/cast、elementwise、attention 是否有优化空间

### 反模式 2：忽视 launch-bound
**症状**: `compute_utilization < 0.4` 但策略仍在抠单个算子
**问题**: launch overhead 是系统性问题，局部优化无法解决
**建议**: 优先 `graph_rewrite`，从根源解决 launch 碎片化

### 反模式 3：误判噪声为真实差异
**症状**: `noise_relative > 0.05`，两个策略的延迟差 <3%
**问题**: 差异可能在噪声范围内，结论不可信
**建议**: 增加 profile iters 或放宽比较阈值

## Profile Findings 的解读

`profile_findings` 是 analyze_profile 生成的自然语言建议，常见模式：

| Finding | 含义 | 推荐 Lever |
|---------|------|-----------|
| `"launch-bound: N small kernels"` | Launch overhead 占主导 | `graph_rewrite` |
| `"frequent ND↔NZ conversions"` | Layout 转换频繁 | `loading_time` |
| `"attention on naive path"` | Attention 未融合 | `operator_fusion` |
| `"high copy/cast overhead (X%)"` | 数据搬运过多 | `loading_time` 或 `forward_patch` |
| `"matmul shape misalignment"` | GEMM tile 利用率低 | `loading_time`（padding） |
| `"residual fp32 parameters"` | Dtype 混用 | `loading_time`（dtype 清理） |

## 实战案例

### 案例 1：Launch-bound + Copy/Cast 双重问题

**输入**:
```python
AnalysisResult(
    total_latency=52.3,
    op_type_totals={
        "matmul": {"pct_total": 38.2},
        "copy_cast": {"pct_total": 11.5},
        "elementwise": {"pct_total": 6.3}
    },
    roofline_summary={
        "compute_utilization": 0.31,
        "kernel_count": 412
    },
    profile_findings=[
        "launch-bound: 412 small kernels",
        "frequent ND↔NZ conversions (11.5% overhead)"
    ]
)
```

**诊断**:
1. Launch-bound：`compute_utilization` 低 + kernel 多 → **主要瓶颈**
2. Copy/Cast 高：11.5% → **次要瓶颈**

**策略优先级**:
1. `graph_rewrite`（1.2-1.4× 预期）：解决 launch overhead
2. `loading_time`（1.10-1.12× 预期）：预转 layout，叠加收益

### 案例 2：Attention 未融合

**输入**:
```python
AnalysisResult(
    total_latency=48.6,
    top_ops=["aten::linear", "aten::softmax", "aten::transpose", "aten::matmul"],
    op_type_totals={
        "matmul": {"pct_total": 42.1},
        "softmax": {"pct_total": 5.8}  # 异常高
    },
    roofline_summary={"compute_utilization": 0.54}
)
```

**诊断**:
- `flash_attention` 缺失 + `softmax` 高占比 → Attention 走了 naive 路径

**策略**:
- `operator_fusion`（1.2-1.5× 预期）：设置 `attn_implementation`
