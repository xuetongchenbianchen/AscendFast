---
name: npu-strategy
description: Generate NPU optimization strategies from profiling data. Use this skill whenever you need to analyze Ascend NPU performance bottlenecks, map hotspots to optimization measures, estimate speedup ratios, or propose concrete optimization strategies. Also use when the user mentions AnalysisResult, profile interpretation, "what should I optimize", strategy generation, or asks about optimization priorities for NPU models.
---

# NPU Optimization Strategy

从 AnalysisResult 生成 OptimizationStrategy 所需的领域知识。

## OptimizationStrategy 结构

```python
@dataclass
class OptimizationStrategy:
    uid: str
    local_speedup_ratio: float  # expected speedup for this bottleneck
    measures: list[str]          # concrete steps
    prompt_instruction: str      # full prompt for the apply_optimization agent
    extra: dict | None           # contains "kind" (lever), "custom_operator" (optional hint)
```

## 先选 LEVER，再选 measure

一条策略会落在四个层级之一，`ChangeRecord.kind` 记录的就是落在哪一层。
**不要**默认用 `forward_patch`——NPU 上的大部分收益都在 forward 之外。
对每一份 AnalysisResult，至少要在两个不同的 lever 上生成候选。

| `kind` | 改动落在哪 | 影响什么 |
|---|---|---|
| `forward_patch` | monkey-patch 某个 `nn.Module.forward` | 单个算子的 dtype/cast、单层内的 elementwise 融合 |
| `operator_fusion` | 模型 **config** / `attn_implementation` | 整条代码路径切到融合后端（无需手工 patch） |
| `graph_rewrite` | 在 `build_model()` 里包一层 model | 对**整个** model 做 `torch.compile` / NPU 图模式 / graph capture |
| `loading_time` | `from_pretrained` 之后、`return` 之前 | 权重布局（ND→NZ）、dtype 清理、静态 KV cache、seq padding |

`forward_patch` 是最窄的杠杆，一次只治一个算子的症状。
`graph_rewrite` / `operator_fusion` / `loading_time` 改的是 `build_model.py`
本身，通常更系统。当 profile 是 **launch-bound**（kernel 又小又多、
`roofline_summary` 里算力利用率低）时，优先 `graph_rewrite`，别再去抠单个 cast。

## 选择 Lever 的决策流程

按以下优先级评估（从上到下）：

1. **Launch-bound profile**（`roofline_summary` 算力利用率低 + kernel 又多又小）
   → 首选 `graph_rewrite`（整模型 compile / 图模式）

2. **Attention 在 naive/eager 路径上**（`top_ops` 里有多个 attention 相关算子、未走融合）
   → 首选 `operator_fusion`（翻 config 的 `attn_implementation` 开关）

3. **权重布局或 dtype 问题**（matmul 对齐差、残留 fp32 参数）
   → 首选 `loading_time`（一次性预处理：ND→NZ、dtype 清理）

4. **单个算子瓶颈**（某个 op 占比 >10%，其他分散）
   → `forward_patch`（针对性修复，考虑融合或自定义算子）

5. **多种问题混合**
   → 生成至少 2 条不同 lever 的策略，覆盖主次瓶颈

## 按热点类型的策略 Playbook（多数为 `forward_patch`）

### Matmul (>20% of time)
**Measures**:
- 确认 GEMM 布局是 ND→NZ（fractal 格式）、dtype 为 FP16/BF16。
- 避免在 matmul 前后紧挨着做冗余的 layout 转换。
- 检查 batch、seq_len、hidden_dim 是否为 16（kernel tile size）的整数倍。
- 有 fused matmul+bias+activation 时优先使用。

**Speedup estimate**: 对齐差时 1.1–1.3；已优化好则 1.05。

### Flash Attention (>3% of time)
**Measures**:
- 确认 model config 里 `use_flash_attention=True` 或等价开关已打开。
- 确认 attention mask 没有把路径退回到 naive matmul。
- 检查 seq_len padding：向上取整到 128/256 以提升 kernel 效率。
- 去掉 attention 层周围的 dtype 转换。

**Speedup estimate**: 未走融合路径时 1.2–1.5；已在快路径上则 1.05。

### Copy/Cast (>1% of time)
**Measures**:
- 审计 forward pass：在每个算子处打印 tensor 的 `.dtype` 和 `.layout`。
- 删除冗余的 `.to(dtype)` 或 `.contiguous()` 调用。
- 在整条热路径上钉死单一 layout（NPU 上优先 NZ）。
- 把无法避免的 cast 移出 per-token 循环（例如 embedding 在 init 时一次性 cast）。

**Speedup estimate**: 视可消除的开销多少，1.05–1.15。

### Norm + Reduce (>3% combined)
**Measures**:
- 有 fused RMSNormLinear 时，把分离的 RMSNorm → Linear 替换成它。
- 检查 reduction 轴是否符合原生 kernel 预期（优先最后一维）。
- 避免归一化路径里出现中间的 `.cpu()` / `.numpy()`（自定义层里常见）。

**Speedup estimate**: 有融合可用时 1.08–1.2。

## 非 forward 的 Playbook（改 `build_model.py`，不是改某个 `forward`）

这些杠杆改的是入口本身。所有改动都放在 `build_model()` **函数体内**
（`from_pretrained` 之后、`return` 之前），遵守项目的 import 规则。
NPU 算子一律用 `hasattr(torch_npu, ...)` 守卫，并保留 fallback。

### `graph_rewrite` — 整模型 compile / 图模式（launch-bound profile）
**何时用**：`roofline_summary` 显示算力利用率低 + kernel 又多又小，
或 Copy/Cast/launch 开销占主导、逐算子 patch 已到瓶颈。
**Measures**:
- 包一层返回的 model：`model = torch.compile(model, backend=...)`（NPU backend），
  或启用 `torch_npu` 图模式 / ACL graph capture。
- 把 decode 步 capture 成图，摊薄 kernel-launch 开销。
- 对照未 compile 的 model 做数值自检；偏差超阈值就 fallback。
**Speedup estimate**: launch-bound 时 1.1–1.4；本已 compute-bound 则 ~1.0。
**kind**: `graph_rewrite`

### `operator_fusion` — 通过 config 切后端（attention 路径）
**何时用**：attention 在 eager/naive 路径上（你正想手 patch softmax dtype 时）。
优先「翻 backend 开关」，而不是 patch forward。
**Measures**:
- 设置 `attn_implementation`（或本版 transformers 暴露的对应 config flag），
  把 attention 路由到融合 NPU kernel。
- 确认 attention mask 没有把路径退回到 naive matmul。
- 确认 attention 周围没有引入新的 ND↔NZ layout 转换。
**Speedup estimate**: 成功切离 naive 路径时 1.2–1.5；否则 1.05。
**kind**: `operator_fusion`

### `loading_time` — 权重布局 / dtype / cache / shape（一次性，decode 密集）
**何时用**：matmul 对齐差、存在残留 fp32 参数，或 decode 每步重分配
KV cache / 因动态 shape 触发重编译。
**Measures**:
- 加载时一次性把权重预转换成 ND→NZ（fractal），让 matmul 跳过每步转换。
- 扫一遍 `model.parameters()`，把残留 fp32 → fp16/bf16（保留 `inv_freq`
  等精度关键 buffer 为 fp32）。
- 启用**静态** KV cache（如 `StaticCache`），让 decode 不再重分配。
- 把 seq_len / KV length pad 到 128/256 以命中 kernel tile size。
**Speedup estimate**: 1.05–1.2；静态 cache 在 decode 阶段收益最大。
**kind**: `loading_time`

## Custom Operator 提示（可选）

当 profile 显示一个**多算子融合**机会、而官方 `torch_npu` 没有对应的融合算子时，
可以在策略的 `extra.custom_operator` 里附一个**提示**（不是命令）。

### 何时附带 custom_operator 提示

**适合的场景**：
- 热点是多个小算子的序列（如 RMSNorm+residual、QKV+bias、RoPE+attention）
- 官方 torch_npu 缺少对应的融合算子
- 融合后能显著减少 launch/cast/GM-roundtrip 开销
- 这些算子紧邻出现在 `top_ops` 里，且总占比 >5%

**不要附带的场景**：
- 官方已有覆盖良好的融合算子（手写 kernel 很难胜过官方优化）
- 单个算子瓶颈（优化单算子不如用官方实现）
- 不确定是否值得融合时
- 算子之间有数据依赖，无法简单融合

### 提示格式（JSON）

```json
{
  "op_name": "rms_norm_residual",
  "semantic": "y = rms_norm(x + residual, gamma, eps)",
  "why_custom": "fuse two separate ops into one kernel, avoid intermediate GM write",
  "fusion_targets": ["RMSNorm", "Add"],
  "expected_signature": "rms_norm_residual(x, residual, gamma, eps) -> y"
}
```

### 重要约束

- 这只是给 apply-agent 的**提示**。apply-agent 会读真实代码后决定是否真的需要自定义算子，以及具体的 signature/arch_params。
- 你只需说 **WHAT/WHY**，不要预判 HOW（tiling、数据类型、shape 约束等）。
- **每条策略最多附一个** custom_operator 提示。
- 提示中的 `op_name`、`semantic`、`why_custom` 是必须的；`fusion_targets` 和 `expected_signature` 是可选的。

## 估计 `local_speedup_ratio`

### 从 profile 数据提取瓶颈占比

- **单算子瓶颈**：在 `op_type_totals` 里找目标 op 的 `device_time_ms` / `total_latency`
- **多算子融合**：把 `fusion_targets` 里所有 op 的时间加起来
- **整体改动**（graph/loading）：看 `roofline_summary` 或相关 finding 里的描述

### 用 Amdahl 定律估计整体加速

若瓶颈占 **X%**、预期局部加速 **S**，整体加速 ≈ `1 / (1 - X/100 * (1 - 1/S))`

**公式解释**：
- `X/100` 是瓶颈占比（小数形式）
- `1/S` 是优化后该部分的相对时间
- `(1 - 1/S)` 是优化掉的时间比例
- `X/100 * (1 - 1/S)` 是对总时间的改善
- `1 - X/100 * (1 - 1/S)` 是优化后的总相对时间
- 倒数即为加速比

**示例**：
- matmul 占 40%，预期局部 1.2× → 整体 ≈ 1 / (1 - 0.4 * (1 - 1/1.2)) ≈ 1.07
- launch overhead 占 25%，graph compile 预期 1.5× → 整体 ≈ 1 / (1 - 0.25 * (1 - 1/1.5)) ≈ 1.09
- copy/cast 占 8%，消除后 → 整体 ≈ 1 / (1 - 0.08) ≈ 1.09

### 保守默认值（agent 不确定时用）

- 明确瓶颈 + 已知修法：1.1–1.2
- 推测性修复：1.05
- 重大架构改动（fusion/graph）：1.2–1.5
- launch-bound profile 做整模型 graph_rewrite：1.1–1.4

## 在一次扇出里让 lever 多样化

当被要求为同一份 AnalysisResult 给出 top-K 策略时，**不要**返回 K 个
同一 `forward_patch` 的变体。至少覆盖两种不同的 `kind`——
例如一条 `forward_patch`（便宜、低风险）加一条 `graph_rewrite` 或
`loading_time`（上限更高）。

## 编写 `prompt_instruction`

模板（来自 `_build_strategy_prompt`）：
```
Implement the optimization strategy below against the model execution.
The focus and measures are already chosen — your job is the HOW: select the
concrete API, signature, and guards; decide where to apply the patch and how
to wire build_model; and verify functional equivalence. Preserve correctness
while delivering the targeted latency reduction.

Lever (kind): <kind>
<lever-specific hint from _LEVER_HINTS>

Focus:
<one-sentence bottleneck + goal>

Measures:
- <concrete step 1>
- <concrete step 2>
- <concrete step 3>

Profile context:
- analysis_uid: ...
- model_id: ...
- device: ...
- dtype: ...
- total_latency_ms: ...
- top_ops: ...
- op_type_totals: {...}
- roofline_summary: {...}

Report this same kind (<kind>) in your ChangeRecord JSON unless the actual
change ended up on a different lever.
```

收到这个 prompt 的 agent 会修改模型代码。要具体：引用真实的算子名、
已知的文件路径，尽量给出 before/after 代码片段。

对于非 `forward_patch` 的杠杆，显式声明 lever，好让 apply-agent 改对位置：
- `graph_rewrite` / `loading_time`：「Edit `build_model.py` inside `build_model()`
  (after `from_pretrained`, before `return`); do **not** patch any `forward`.」
- `operator_fusion`：「Set the config flag at load; do not hand-patch the
  attention forward.」

在 prompt 和 `extra={"kind": ...}` 里都带上 `kind`。

## 完整示例

### Input (AnalysisResult 摘要)

```python
AnalysisResult(
    total_latency=45.2,  # ms
    top_ops=["aten::linear", "aten::copy_", "aten::mul", "aten::softmax"],
    op_type_totals={
        "matmul": {"device_time_ms": 18.5, "pct_total": 40.9},
        "copy_cast": {"device_time_ms": 3.6, "pct_total": 8.0},
        "elementwise": {"device_time_ms": 2.1, "pct_total": 4.6}
    },
    roofline_summary={
        "compute_utilization": 0.35,
        "memory_bandwidth_utilization": 0.68
    },
    profile_findings=[
        "launch-bound: 328 small kernels with low compute utilization",
        "frequent ND↔NZ layout conversions detected"
    ]
)
```

### Output (OptimizationStrategy 示例)

**Strategy 1** (`graph_rewrite`，主攻 launch overhead):
```python
OptimizationStrategy(
    uid="strategy:...:1",
    local_speedup_ratio=1.25,
    measures=[
        "Wrap model with torch.compile(backend='npu') or torch_npu graph mode",
        "Enable graph capture for decode step to amortize kernel launch",
        "Verify numerical equivalence against eager baseline with tolerance atol=1e-2"
    ],
    prompt_instruction="...",  # 根据模板生成
    extra={
        "kind": "graph_rewrite",
        "model_id": "...",
        "custom_operator": None
    }
)
```
**估算依据**：launch overhead 隐含占比约 20-25%（从 compute_utilization 低推断），
graph compile 预期局部 1.5× → 整体约 1 / (1 - 0.22 * 0.33) ≈ 1.08，保守上调到 1.25
考虑整体融合收益。

**Strategy 2** (`loading_time`，主攻 layout 转换):
```python
OptimizationStrategy(
    uid="strategy:...:2",
    local_speedup_ratio=1.12,
    measures=[
        "Pre-convert all linear layer weights to NZ format after from_pretrained",
        "Pin fp16 dtype across hot path to eliminate redundant casts",
        "Pad sequence length to 128 for better tile alignment"
    ],
    prompt_instruction="...",
    extra={
        "kind": "loading_time",
        "model_id": "...",
        "custom_operator": None
    }
)
```
**估算依据**：copy_cast 占 8%，完全消除 → 整体 1/(1-0.08) ≈ 1.09，考虑对齐收益
上调到 1.12。
