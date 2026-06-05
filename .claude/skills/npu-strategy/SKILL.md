---
name: npu-strategy
description: Generate NPU optimization strategies. Use when mapping profiling hotspots to concrete optimization measures and estimating speedup for Ascend NPU models.
---

# NPU Optimization Strategy

Domain knowledge for generating OptimizationStrategy from AnalysisResult.

## OptimizationStrategy Structure

```python
@dataclass
class OptimizationStrategy:
    uid: str
    local_speedup_ratio: float  # expected speedup for this bottleneck
    measures: list[str]          # concrete steps
    prompt_instruction: str      # full prompt for the apply_optimization agent
    extra: dict | None
```

## Strategy Playbook by Hotspot Type

### Matmul (>20% of time)
**Measures**:
- Verify GEMM layout is ND→NZ (fractal format) and dtype is FP16/BF16.
- Avoid redundant layout conversions immediately before/after matmul.
- Check batch, seq_len, hidden_dim are multiples of 16 (kernel tile size).
- Use fused matmul+bias+activation when available.

**Speedup estimate**: 1.1–1.3 if poorly aligned; 1.05 if already optimized.

### Flash Attention (>3% of time)
**Measures**:
- Ensure `use_flash_attention=True` or equivalent in model config.
- Verify attention mask is not forcing fallback to naive matmul path.
- Check seq_len padding: round up to 128/256 for better kernel efficiency.
- Remove dtype conversions around attention layer.

**Speedup estimate**: 1.2–1.5 if not using fused path; 1.05 if already on fast path.

### Copy/Cast (>1% of time)
**Measures**:
- Audit model forward pass: insert print/log for tensor `.dtype` and `.layout` at each op.
- Remove redundant `.to(dtype)` or `.contiguous()` calls.
- Pin a single layout (preferably NZ for NPU) across the hot path.
- Move unavoidable casts out of the per-token loop (e.g., cast embeddings once at init).

**Speedup estimate**: 1.05–1.15 depending on how much overhead is removable.

### Norm + Reduce (>3% combined)
**Measures**:
- Replace separate RMSNorm → Linear with fused RMSNormLinear if available.
- Check reduction axes match native kernel expectations (last dim preferred).
- Avoid intermediate `.cpu()` or `.numpy()` in normalization paths (seen in custom layers).

**Speedup estimate**: 1.08–1.2 if fusion available.

## Estimating `local_speedup_ratio`

Use Amdahl's Law:  
If bottleneck is **X%** of runtime and you expect **S** local speedup, overall speedup ≈ `1 / (1 - X/100 * (1 - 1/S))`.

Example: matmul is 40% of time, you expect 1.2× local improvement → overall ≈ 1.07.

Conservative defaults:
- Clear bottleneck with known fix: 1.1–1.2
- Speculative or partial fix: 1.05
- Major architectural change such as replacing an unfused hot path with a fused operator: 1.2–1.5

## Writing `prompt_instruction`

Template (from `_build_strategy_prompt`):
```
Optimize the model execution according to the profiling analysis.
Keep numerical correctness unchanged and prefer small, measurable changes.

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
```

The agent receiving this prompt will modify the model code. Be specific: reference actual op names, file paths if known, and give before/after code snippets when possible.
