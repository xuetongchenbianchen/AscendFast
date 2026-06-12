---
name: npu-apply
description: Implement NPU (Ascend) inference optimizations in a model workspace. Use whenever applying an OptimizationStrategy to build_model.py — wiring fused torch_npu ops, torch.compile/graph mode, static KV cache, weight layout (ND→NZ), or dtype cleanup — and verifying the result still runs and stays numerically equivalent. Reach for this anytime you are editing a forked Ascend model workspace to make it faster without changing outputs.
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
def build_model():
    model, tok = AutoModelForCausalLM.from_pretrained(...), ...
    from patches import my_patch   # 在函数体内
    my_patch.apply(model)
    return model, tok
```

新代码一律放进 workspace 内（编辑 `build_model.py`，新增 `patches/*.py`、
`config/*`、`graph/*`）。`model/` 下的权重是从父 mode **硬链接**来的，
和父 mode 共享 inode——要改权重（如量化）就写**新文件**，别原地改硬链接文件。

## torch_npu fused op：先 probe，再接进 forward

fused op 的参数错误（dtype / shape / layout）**只在 forward 时才暴露**，
构造期检查抓不到。所以调任何 `npu_*` 算子前，按这个顺序来：

1. 先确认算子存在：`hasattr(torch_npu, "npu_rms_norm")`——永远不要假设它在。
2. 查真实签名，别照搬 `nn.Linear`：
   `python -c "import torch_npu; print(torch_npu.npu_linear.__doc__)"`
3. 用一个小张量单独 probe 它能不能算通，再接进 forward。
4. 一律保留 eager fallback：`hasattr` 为假或 probe 失败时走原路径。

已知坑：
- `npu_linear(input, weight, bias=None)` 要求 **input 是 2D**。transformer 激活是
  3D `[B,S,H]`，必须 `x.reshape(-1, H)` 再调用，输出 `reshape(B, S, -1)` 回去。
- weight 布局 `[out, in]`，和 `nn.Linear` 一致——**不要**转置。
- input 和 weight 的 dtype 必须一致（都 fp16 或都 bf16）。
- `npu_rms_norm(x, gamma, epsilon)` 返回 tuple，取 `[0]` 才是结果。

## 按杠杆（lever）落地

策略的 `kind` 提示改动该落在哪一层。不同 lever 的落地手法不同：

### `forward_patch` — monkey-patch 某个 `nn.Module.forward`
最窄的杠杆，治单个算子。在 `build_model()` 里 import 一个 patch 模块，
把目标 module 的 `forward` 换成融合实现。**叠加**而非替换：若 change log
里已经 patch 过同一个 forward，扩展它，别推翻重写。
transformers 4.57.1 里 `Qwen2Attention` 可用；`Qwen2FlashAttention2` /
`Qwen2SdpaAttention` **不存在**，不要 import。

### `operator_fusion` — 通过 config 切后端
优先翻 config 开关（如 `attn_implementation`）把整条路径切到融合 kernel，
而不是手 patch forward——这样无需维护 patch，覆盖面更广。确认切换后
attention mask 没把路径退回 naive matmul。

### `graph_rewrite` — 整模型 compile / 图模式
在 `build_model()` 里包一层返回的 model：`torch.compile(model, backend=...)`
或 `torch_npu` 图模式 / ACL graph capture。适合 launch-bound（kernel 又多又小、
`roofline_summary` 算力利用率低）。包完务必和未 compile 的 model 做数值自检，
偏差超阈值就 fallback 回 eager。

### `kvcache` — 静态 KV cache
decode 每步重分配 KV cache 是常见浪费。启用 `StaticCache`（或等价）让 decode
不再重分配；注意 cache 容量要覆盖最大 seq len。

### `quantize` / `config` — 加载期一次性处理
在 `from_pretrained` 之后、`return` 之前做：
- 权重一次性预转 ND→NZ（fractal）布局，让 matmul 跳过每步转换。
- 扫 `model.parameters()` 把残留 fp32 → fp16/bf16，但**保留 `inv_freq` 等
  精度关键 buffer 为 fp32**（RoPE 频率精度丢了会坏掉数值）。

## 收尾验证（两步，缺一不可）

**第一步：build 能加载。** 返回 JSON 前先跑这个；抛异常就先修，绝不在
workspace 损坏时返回：

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

**第二步：forward 能真正跑通。** 能构造 ≠ 能前向。fused op 的错误只在
forward 时炸，所以必须实际跑一次前向：

```bash
cd <workspace> && python -c "
import torch, build_model as bm
m, tok = bm.build_model()
ids = tok('hello world', return_tensors='pt')['input_ids'].to(next(m.parameters()).device)
with torch.no_grad(): m(ids)
print('forward OK')"
```

## 数值等价

优化的前提是输出不变。改完后对同一组输入，比较优化前后 logits（或下一个
token 分布）：fp16/bf16 下用宽松容差（如 `atol=1e-2`），明显发散就说明落地
有 bug——回退到 eager fallback，而不是放过它。把"现在模型真实状态 +
对下一轮的约束"写进 ChangeRecord 的 `details`，下一轮要靠它继续叠加。
