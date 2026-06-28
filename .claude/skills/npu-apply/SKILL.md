---
name: npu-apply
description: Implement NPU (Ascend) inference optimizations in a model workspace. Use this skill whenever you need to apply an OptimizationStrategy to build_model.py, wire fused torch_npu ops, integrate custom operators from torch.ops.ascendfast, implement torch.compile/graph mode, configure static KV cache, optimize weight layout (ND→NZ), perform dtype cleanup, or verify numerical equivalence. Also use when the user asks to "apply this optimization", "wire this operator", "make this model faster", or mentions editing a forked workspace.
---

# NPU Optimization Apply

在 Ascend 910 NPU 上把一条优化策略真正**落地到代码**所需的实现知识。
策略（WHAT/WHY）已经给定，这里管的是 HOW：选对 API、接进 `build_model()`、
守住数值等价。配套硬约束（输出契约、边界）见 `apply-agent.md`，这里不重复。

## Workspace 纪律（不守会污染别的 workspace）

`workspace_loader.py` 通过在 import 前后快照 `sys.modules` 来隔离每个 workspace。
所以 `from patches import ...` 这类语句**必须写在 `build_model()` 函数体内**，
不能放模块顶层——顶层 import 在 `exec_module()` 时执行，会和之前加载过的同名
`patches` 包串味。

```python
# CORRECT
def build_model():
    model, tok = AutoModelForCausalLM.from_pretrained(...), ...
    from patches import my_patch   # 在函数体内
    my_patch.apply(model)
    return model, tok

# WRONG — causes sys.modules pollution
from patches import my_patch        # 模块顶层，会污染其他 workspace
def build_model():
    ...
```

新代码一律放进 workspace 内（编辑 `build_model.py`，新增 `patches/*.py`、
`config/*`、`graph/*`）。`model/` 下的权重是从父 mode **硬链接**来的，
和父 mode 共享 inode——要改权重（如量化）就写**新文件**，别原地改硬链接文件。

## torch_npu fused op：先 probe，再接进 forward

fused op 的参数错误（dtype / shape / layout）**只在 forward 时才暴露**，
构造期检查抓不到。所以调任何 `npu_*` 算子前，按这个顺序来：

### 标准接入流程（4 步）

1. **确认算子存在**：`hasattr(torch_npu, "npu_rms_norm")`——永远不要假设它在。
2. **查真实签名**，别照搬 `nn.Linear`：
   ```bash
   python -c "import torch_npu; print(torch_npu.npu_linear.__doc__)"
   ```
3. **用小张量 probe**：单独测试能不能算通，再接进 forward。
4. **保留 eager fallback**：`hasattr` 为假或 probe 失败时走原路径。

### 示例：接入 npu_rms_norm

```python
def build_model():
    model, tokenizer = AutoModelForCausalLM.from_pretrained(...)
    
    # 在函数体内 import
    import torch_npu
    
    # 1. 检查算子存在
    has_npu_rms_norm = hasattr(torch_npu, "npu_rms_norm")
    
    if has_npu_rms_norm:
        # 2. Probe：用小张量测试
        try:
            test_x = torch.randn(2, 64, dtype=torch.float16).npu()
            test_gamma = torch.randn(64, dtype=torch.float16).npu()
            result = torch_npu.npu_rms_norm(test_x, test_gamma, 1e-6)
            # npu_rms_norm 返回 tuple，取 [0]
            if isinstance(result, tuple):
                result = result[0]
            probe_ok = result.shape == test_x.shape
        except Exception:
            probe_ok = False
        
        if probe_ok:
            # 3. 接入 forward（patch）
            from patches import rms_norm_fused
            rms_norm_fused.apply(model)
            print("Using fused npu_rms_norm")
        else:
            print("npu_rms_norm probe failed, using eager")
    else:
        print("npu_rms_norm not available, using eager")
    
    model.eval().to("npu:0")
    return model, tokenizer
```

### 已知坑（必读）

| 算子 | 坑 | 解法 |
|------|-----|------|
| `npu_linear` | 要求 input 是 **2D** | transformer 激活是 3D `[B,S,H]`，必须 `x.reshape(-1, H)` 再调用，输出 `reshape(B, S, -1)` 回去 |
| `npu_linear` | weight 布局 `[out, in]` | 与 `nn.Linear` 一致——**不要**转置 |
| `npu_linear` | input/weight dtype 必须一致 | 都 fp16 或都 bf16；混用会报错 |
| `npu_rms_norm` | 返回 tuple | 取 `[0]` 才是结果；`[1]` 是中间值 |
| `npu_rotary_mul` | 可能不存在 | 必须 `hasattr` 守卫 |

## 接入自定义算子（torch.ops.ascendfast.*）

项目内的 `kernels/` 包提供 `torch.ops.ascendfast.*` 命名空间的自定义算子。
接入方式与官方 `torch_npu.npu_*` 类似，但有额外约束。

### 标准接入流程

```python
def build_model():
    model, tokenizer = AutoModelForCausalLM.from_pretrained(...)
    
    # 1. 在函数体内 import
    try:
        import ascendfast_ops  # 自动加载 lib/*.so
    except ImportError:
        print("ascendfast_ops not installed, using eager")
        return model, tokenizer
    
    # 2. 检查算子存在
    has_custom_op = hasattr(torch.ops.ascendfast, "rms_norm_residual")
    
    if has_custom_op:
        # 3. Probe：用真实 shape 测试（arch_params 的规模）
        try:
            h = 896  # 从 model.config.hidden_size 读取
            test_x = torch.randn(4, h, dtype=torch.float16).npu()
            test_res = torch.randn(4, h, dtype=torch.float16).npu()
            test_gamma = torch.randn(h, dtype=torch.float16).npu()
            result = torch.ops.ascendfast.rms_norm_residual(
                test_x, test_res, test_gamma, 1e-6
            )
            probe_ok = result.shape == test_x.shape
        except Exception as e:
            print(f"Custom op probe failed: {e}")
            probe_ok = False
        
        if probe_ok:
            # 4. 接入 forward
            from patches import rms_norm_residual_patch
            rms_norm_residual_patch.apply(model)
            print("Using custom rms_norm_residual")
        else:
            print("Custom op probe failed, using eager")
    else:
        print("Custom op not found, using eager")
    
    model.eval().to("npu:0")
    return model, tokenizer
```

### 关键约束

1. **必须 source 环境脚本**：`scripts/ascend-env.sh` 会 source CANN env 和 custom op 的 `set_env.bash`
2. **Probe shape 必须真实**：用 `model.config.hidden_size` 等真实参数，不要用 toy shape（64）
3. **Custom op 可能返回 tuple**：根据 `OperatorArtifact.usage_note` 处理
4. **Fallback 是必须的**：custom op 失败不应该让整个模型崩溃

### 从 OperatorArtifact 提取信息

当 apply-agent 收到一个 `OperatorArtifact` 时，关键字段：

```python
artifact = OperatorArtifact(
    op_name="rms_norm_residual",
    qualified_name="torch.ops.ascendfast.rms_norm_residual",
    signature="rms_norm_residual(x, residual, gamma, eps) -> y",
    installed=True,
    supported_dtypes=["float16", "float32"],
    numeric_max_rel_err=0.023,
    usage_note="Input shapes must match. Returns single tensor (not tuple). Requires hidden_size >= 128."
)
```

**使用 `usage_note`**：
- 告诉你调用约束（shape、返回类型、限制）
- 例如 "Returns tuple, take [0]" → `result = op(...)[0]`
- 例如 "Requires reshape to 2D" → `x = x.reshape(-1, H); y = op(x); y = y.reshape(B, S, -1)`

## 按杠杆（lever）落地

策略的 `kind` 提示改动该落在哪一层（四个 lever，与 npu-strategy skill 一致）。
不同 lever 的落地手法不同：

### `forward_patch` — monkey-patch 某个 `nn.Module.forward`

最窄的杠杆，治单个算子。在 `build_model()` 里 import 一个 patch 模块，
把目标 module 的 `forward` 换成融合实现。**叠加**而非替换：若 change log
里已经 patch 过同一个 forward，扩展它，别推翻重写。

**transformers 4.57.1 兼容性**：
- `Qwen2Attention` 可用
- `Qwen2FlashAttention2` / `Qwen2SdpaAttention` **不存在**，不要 import

**示例**：patch RMSNorm
```python
# patches/rms_norm_fused.py
import torch
import torch.nn as nn

def apply(model):
    """Replace RMSNorm forward with fused npu_rms_norm."""
    import torch_npu
    
    if not hasattr(torch_npu, "npu_rms_norm"):
        return  # Fallback: no-op
    
    for name, module in model.named_modules():
        if isinstance(module, nn.RMSNorm):  # 假设模型用 nn.RMSNorm
            original_forward = module.forward
            
            def fused_forward(self, x):
                # npu_rms_norm(input, gamma, epsilon) -> (output, rstd)
                result = torch_npu.npu_rms_norm(
                    x, self.weight, self.eps
                )
                return result[0]  # 取第一个元素
            
            module.forward = fused_forward.__get__(module, type(module))
```

### `operator_fusion` — 通过 config 切后端

优先翻 config 开关（如 `attn_implementation`）把整条路径切到融合 kernel，
而不是手 patch forward——这样无需维护 patch，覆盖面更广。确认切换后
attention mask 没把路径退回 naive matmul。

**示例**：启用 flash attention
```python
def build_model():
    # 方法 1：修改 config（推荐）
    config = AutoConfig.from_pretrained(str(_MODEL_DIR))
    config.attn_implementation = "flash_attention_2"  # 或 "eager" / "sdpa"
    
    model = AutoModelForCausalLM.from_pretrained(
        str(_MODEL_DIR),
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # 方法 2：直接传参（如果支持）
    model = AutoModelForCausalLM.from_pretrained(
        str(_MODEL_DIR),
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR))
    model.eval().to("npu:0")
    return model, tokenizer
```

### `graph_rewrite` — 整模型 compile / 图模式

在 `build_model()` 里包一层返回的 model：`torch.compile(model, backend=...)`
或 `torch_npu` 图模式 / ACL graph capture。适合 launch-bound（kernel 又多又小、
`roofline_summary` 算力利用率低）。包完务必和未 compile 的 model 做数值自检，
偏差超阈值就 fallback 回 eager。

**示例**：torch.compile
```python
def build_model():
    model, tokenizer = AutoModelForCausalLM.from_pretrained(...)
    model.eval().to("npu:0")
    
    # 尝试 compile
    try:
        import torch_npu
        if hasattr(torch, "compile"):
            compiled_model = torch.compile(
                model,
                backend="npu",  # 或其他 NPU backend
                mode="reduce-overhead"
            )
            
            # 数值自检
            test_ids = tokenizer("hello world", return_tensors="pt")["input_ids"].npu()
            with torch.no_grad():
                eager_out = model(test_ids).logits
                compiled_out = compiled_model(test_ids).logits
            
            rel_err = (eager_out - compiled_out).abs().max() / (eager_out.abs().max() + 1e-9)
            if rel_err < 1e-2:  # 宽松容差（fp16）
                print(f"Compile OK, rel_err={rel_err:.4f}")
                return compiled_model, tokenizer
            else:
                print(f"Compile numerical mismatch ({rel_err:.4f}), using eager")
    except Exception as e:
        print(f"Compile failed: {e}, using eager")
    
    return model, tokenizer
```

### `loading_time` — 加载期一次性处理

在 `from_pretrained` 之后、`return` 之前一次性做完，让 decode 每步省掉重复开销。
涵盖**静态 KV cache、权重量化、布局/dtype 清理**——它们都是加载期 lever，不要
当成平级 kind。

#### 静态 KV cache

```python
def build_model():
    model, tokenizer = AutoModelForCausalLM.from_pretrained(...)
    model.eval().to("npu:0")
    
    # 启用静态 KV cache（减少 decode 每步的内存分配）
    try:
        from transformers import StaticCache
        max_cache_len = 2048  # 根据实际需求
        
        # 为每层预分配 cache
        model._setup_cache(StaticCache, max_cache_len=max_cache_len)
        print(f"Enabled static KV cache (max_len={max_cache_len})")
    except Exception as e:
        print(f"Static cache setup failed: {e}")
    
    return model, tokenizer
```

#### 权重布局转换（ND → NZ）

```python
def build_model():
    model, tokenizer = AutoModelForCausalLM.from_pretrained(...)
    
    # 预转换权重布局为 NZ（fractal），让 matmul 跳过每步转换
    import torch_npu
    if hasattr(torch_npu, "npu_format_cast"):
        for name, param in model.named_parameters():
            if param.dim() == 2 and param.numel() >= 1024:  # 只转大矩阵
                try:
                    # NZ format = 29（fractal）
                    param.data = torch_npu.npu_format_cast(param.data, 29)
                except Exception:
                    pass  # 转换失败不影响正确性
        print("Converted weights to NZ format")
    
    model.eval().to("npu:0")
    return model, tokenizer
```

#### Dtype 清理

```python
def build_model():
    model, tokenizer = AutoModelForCausalLM.from_pretrained(...)
    
    # 扫描并转换残留的 fp32 参数为 fp16
    target_dtype = torch.float16
    for name, param in model.named_parameters():
        # 保留精度关键的 buffer（如 RoPE 的 inv_freq）
        if "inv_freq" in name or "freqs" in name:
            continue
        
        if param.dtype == torch.float32:
            param.data = param.data.to(target_dtype)
    
    # 同样处理 buffers
    for name, buf in model.named_buffers():
        if "inv_freq" in name or "freqs" in name:
            continue
        if buf.dtype == torch.float32:
            buf.data = buf.data.to(target_dtype)
    
    print(f"Converted residual fp32 params to {target_dtype}")
    model.eval().to("npu:0")
    return model, tokenizer
```

## 收尾验证（两步，缺一不可）

### 第一步：build 能加载

返回 JSON 前先跑这个；抛异常就先修，绝不在 workspace 损坏时返回：

```bash
cd /models/share/userdata/cb/AscendFast && python -c "
import importlib.util, sys
ws = '<absolute workspace path>'
sys.path.insert(0, ws)
spec = importlib.util.spec_from_file_location('bm', ws + '/build_model.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print('smoke OK, build_model:', type(m.build_model))
"
```

### 第二步：forward 能真正跑通

能构造 ≠ 能前向。fused op 的错误只在 forward 时炸，所以必须实际跑一次前向：

```bash
cd <workspace> && python -c "
import torch
import build_model as bm
m, tok = bm.build_model()
ids = tok('hello world', return_tensors='pt')['input_ids'].to(next(m.parameters()).device)
with torch.no_grad(): 
    out = m(ids)
print('forward OK, output shape:', out.logits.shape)
"
```

## 数值等价

优化的前提是输出不变。改完后对同一组输入，比较优化前后 logits（或下一个
token 分布）：fp16/bf16 下用宽松容差（如 `atol=1e-2`），明显发散就说明落地
有 bug——回退到 eager fallback，而不是放过它。

**验证模板**：
```python
# 在优化前保存一个 baseline
baseline_model, _ = build_model_baseline()
optimized_model, tokenizer = build_model_optimized()

test_prompts = ["Hello world", "The quick brown fox"]
for prompt in test_prompts:
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].npu()
    
    with torch.no_grad():
        baseline_out = baseline_model(ids).logits
        optimized_out = optimized_model(ids).logits
    
    max_diff = (baseline_out - optimized_out).abs().max().item()
    rel_err = max_diff / (baseline_out.abs().max().item() + 1e-9)
    
    if rel_err > 1e-2:  # fp16 宽松容差
        print(f"FAIL: rel_err={rel_err:.4f} for '{prompt}'")
        break
else:
    print("Numerical equivalence verified")
```

把"现在模型真实状态 + 对下一轮的约束"写进 ChangeRecord 的 `details`，
下一轮要靠它继续叠加。

## 完整示例

### 示例 1：接入 npu_rms_norm（forward_patch）

```python
# build_model.py
def build_model(device: str | None = None, dtype=torch.float16):
    device = device or "npu:0"
    tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(_MODEL_DIR), torch_dtype=dtype, trust_remote_code=True
    )
    
    # 在函数体内 import
    import torch_npu
    
    # 接入 fused RMSNorm
    if hasattr(torch_npu, "npu_rms_norm"):
        try:
            # Probe
            test_x = torch.randn(2, 128, dtype=dtype).to(device)
            test_g = torch.randn(128, dtype=dtype).to(device)
            result = torch_npu.npu_rms_norm(test_x, test_g, 1e-6)[0]
            
            if result.shape == test_x.shape:
                from patches import rms_norm_fused
                rms_norm_fused.apply(model)
                print("Applied fused RMSNorm")
        except Exception as e:
            print(f"RMSNorm fusion failed: {e}, using eager")
    
    model.eval().to(device)
    return model, tokenizer
```

### 示例 2：接入自定义算子（forward_patch）

```python
# build_model.py
def build_model(device: str | None = None, dtype=torch.float16):
    device = device or "npu:0"
    tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(_MODEL_DIR), torch_dtype=dtype, trust_remote_code=True
    )
    
    # 尝试加载自定义算子
    try:
        import ascendfast_ops
        
        if hasattr(torch.ops.ascendfast, "rms_norm_residual"):
            # 从 config 读取真实 shape
            h = model.config.hidden_size
            
            # Probe
            test_x = torch.randn(4, h, dtype=dtype).to(device)
            test_res = torch.randn(4, h, dtype=dtype).to(device)
            test_g = torch.randn(h, dtype=dtype).to(device)
            result = torch.ops.ascendfast.rms_norm_residual(
                test_x, test_res, test_g, 1e-6
            )
            
            if result.shape == test_x.shape:
                from patches import rms_norm_residual_patch
                rms_norm_residual_patch.apply(model)
                print("Applied custom rms_norm_residual")
    except Exception as e:
        print(f"Custom op integration failed: {e}, using eager")
    
    model.eval().to(device)
    return model, tokenizer
```

### 示例 3：整模型 compile（graph_rewrite）

```python
# build_model.py
def build_model(device: str | None = None, dtype=torch.float16):
    device = device or "npu:0"
    tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(_MODEL_DIR), torch_dtype=dtype, trust_remote_code=True
    )
    model.eval().to(device)
    
    # 尝试 compile
    try:
        if hasattr(torch, "compile"):
            compiled = torch.compile(model, backend="npu", mode="reduce-overhead")
            
            # 数值自检
            test_ids = tokenizer("test", return_tensors="pt")["input_ids"].to(device)
            with torch.no_grad():
                eager_out = model(test_ids).logits
                compiled_out = compiled(test_ids).logits
            
            rel_err = (eager_out - compiled_out).abs().max() / (eager_out.abs().max() + 1e-9)
            if rel_err < 1e-2:
                print(f"Using compiled model (rel_err={rel_err:.4f})")
                return compiled, tokenizer
            else:
                print(f"Compile numerical mismatch ({rel_err:.4f}), fallback to eager")
    except Exception as e:
        print(f"Compile failed: {e}, using eager")
    
    return model, tokenizer
```
