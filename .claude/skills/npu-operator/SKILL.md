---
name: npu-operator
description: Design, compile, install, and register a custom AscendC operator into the torch.ops.ascendfast.* namespace on Ascend 910. Use this skill whenever you need to create a custom/fused NPU kernel that official torch_npu.npu_* ops don't cover, implement multi-op fusion (RMSNorm+residual, RoPE+attention, QKV+bias), write AscendC device code, compile and install operators into CANN, or numerically verify custom kernels. Also use when the user mentions "custom operator", "AscendC kernel", "operator fusion", or asks to "implement a fused kernel".
---

# NPU Custom Operator (AscendC)

把一个**自定义/融合算子**从零做到可被 `torch.ops.ascendfast.<op>` 调用所需的全部实现知识。
策略（WHAT/WHY）已给定，这里管 kernel-HOW：设计 tiling、写 AscendC、打包安装、编 adapter、
注册、数值自检。配套硬约束（边界、输出契约）见 `operator-agent.md`，这里不重复。

> **你只动 `kernels/` 树。** 把算子接进某个 `build_model()` 是 apply 步骤的事，不是你的。
> 你的交付物是一个**已编译、已安装、已数值验证**的 `torch.ops.ascendfast.<op>`，外加一条
> 写进 `kernels/registry.json` 的记录。

## 高层决策：值得自己写吗？

在开始编码前，先问三个问题：

### 1. 官方是否已有类似算子？

**检查方法**：
```bash
python -c "import torch_npu; print([x for x in dir(torch_npu) if 'npu_' in x and 'norm' in x.lower()])"
```

**决策**：
- **有且功能匹配** → 不要重写，在 `usage_note` 里说明「官方 `torch_npu.npu_xxx` 更快」
- **有但不支持融合** → 值得写融合版本（如官方有 RMSNorm 但无 RMSNorm+residual）
- **完全没有** → 值得写

### 2. 这是融合还是单算子？

**自定义算子能赢的场景**：
- ✅ **融合**：`RMSNorm+residual`、`RoPE+attention`、`QKV+bias`、`SwiGLU` 融成一个 kernel
  - 收益：省掉中间张量的 GM 往返 + kernel launch overhead
  - 预期加速：1.15-1.3×（针对被融合的部分）
  
- ✅ **特化**：hidden_size、num_heads、eps、dtype 编译期定死
  - 收益：省掉通用算子的运行时分支、shape 检查
  - 预期加速：1.05-1.1×（边际收益）

**不能赢的场景**：
- ❌ **重写官方单算子**：手写 RMSNorm 比官方 `npu_rms_norm` 慢 ~4%
  - 原因：官方是华为深度手工调优的库算子，对齐/分块/调度都到位

### 3. 融合的算子是否紧邻？

**检查 OperatorSpec.torch_reference**：
```python
# 好的融合候选（算子紧邻，数据直通）
class Model(nn.Module):
    def forward(self, x, residual, gamma):
        x = x + residual              # Add
        return rms_norm(x, gamma)     # RMSNorm
        # ↑ 中间结果不被其他地方用，可融合

# 不好的融合候选（算子分离，中间有依赖）
class Model(nn.Module):
    def forward(self, x, residual, gamma):
        x = x + residual
        y = some_other_op(x)          # 中间结果被用了
        return rms_norm(x, gamma)     # 无法简单融合
```

**决策树总结**：
```
官方有类似算子？
├─ 有且功能匹配 → 不写，推荐官方版本
└─ 无或不支持融合 → 继续
    ├─ 是多算子融合？
    │   ├─ 算子紧邻 → 值得写（预期 1.15-1.3×）
    │   └─ 算子分离 → 不值得（复杂度高，收益低）
    └─ 单算子特化 → 边际收益小（1.05-1.1×），优先级低
```

## 全景：一个算子要凑齐两条独立的链

```
A. PyTorch 接入链（让 torch.ops.ascendfast.<op> 能被调用）
   csrc/adapter_<op>.cpp  --build_adapter.py-->  .so  --__init__.py 自动加载-->  注册进 dispatcher

B. device kernel 链（让算子在 NPU 上真正算）
   op_host(tiling) + op_kernel(device)  --build.sh-->  custom_opp_*.run  --安装-->  CANN 找得到 kernel
```

两条都通、且运行时 source 了正确 env，算子才既「调得到」又「算得对」。任一条断，要么
`torch.ops.ascendfast.<op>` 不存在（A 断），要么调用成功但数值全 0 / 等于输入（B 断）。

## 统一工程（不要每个算子 msopgen gen 一个新工程）

所有自定义算子都加进**同一个** msopgen 标准工程
`kernels/ascendc_ops/ascendfast_custom_ops/`。加算子 = 往里加文件，**不重新生成工程**：
`op_host/CMakeLists.txt` 用 `aux_source_directory` 自动收集目录下所有 `.cpp`，所以新增算子
**不用改任何 CMake**。已有 `add_demo`（z=x+y）和 `rms_norm_custom` 两个范例可照抄。

## 环境：每条命令都在「干净子 shell」里跑

本机 shell snapshot 有 `set -u` 坑且 coreutils 不全（`head`/`tr`/`uname` 可能找不到，
`ZSH_VERSION` unbound）。所以编译/安装/验证算子时，**不要**直接在交互 shell 里跑，用这个
干净子 shell 模板（`env -i` 起空环境，自己拼最小 PATH）：

```bash
env -i bash --noprofile --norc -c '
set +u
export PATH=/models/share/userdata/cb/AscendFast/.venv/bin:/usr/bin:/bin:/usr/local/bin
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export VIRTUAL_ENV=/models/share/userdata/cb/AscendFast/.venv
export PATH=$VIRTUAL_ENV/bin:$PATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH=/models/share/userdata/cb/AscendFast
# <你的命令；多行就 && 串起来，或写进 workspace 内的临时 .sh 再 bash 它>
'
```

把任何临时探针/构建脚本写进 `kernels/` 或 `/tmp`，**绝不**写进项目根 `AscendFast/`（根目录
不在 .gitignore 内，散落脚本会被 backup 提交误带进版本库）。

## B 链：ops.json → op_host → op_kernel → build.sh → .run 安装

### B1. ops.json 追加算子原型

编辑 `ascendfast_custom_ops/ops.json`，**追加**一项（参考已有的 `RmsNormCustom`）：大驼峰
算子名、每个 input/output 的 name/dtype/format，attr 写进 `attr` 数组。dtype 列表里每登记
一种（fp16/fp32），框架就为它各编一份 kernel。

### B2. op_host：tiling + shape/dtype 推导 + OpDef

`op_host/<op>.cpp` + `op_host/<op>_tiling.h`。照 `rms_norm_custom.cpp` 改三块：

1. **TilingData 结构**（`_tiling.h`）：host 算给 device 的标量参数。RMSNorm 的例子只需
   `numRows / hidden / epsilon`，并 `REGISTER_TILING_DATA_CLASS(<Op>, <Op>TilingData)`。
2. **TilingFunc**：从 `context->GetInputShape(0)` 推 shape，从 `context->GetAttrs()->GetFloat(0)`
   读 attr，决定 `blockDim`（启动多少核）并 `SetBlockDim`。RMSNorm 按行切：
   `blockDim = min(numRows, 48)`，device 端自算每核的行范围。
3. **OpDef**（`class <Op> : public OpDef`）：声明 Input/Output 的 dtype/format、
   `this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6f)`、`SetInferShape`/`SetInferDataType`，
   **必须** `this->AICore().AddConfig("ascend910_93");`，最后 `OP_ADD(<Op>)`。

### B3. op_kernel：device 三段流水

`op_kernel/<op>.cpp`。kernel 入口签名固定，`GET_TILING_DATA` 取回 host 算的参数，
`DTYPE_X` 由框架按 IR dtype 注入（fp16/fp32 各编一份）：

```cpp
extern "C" __global__ __aicore__ void <op>(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
                                           GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  Kernel<DTYPE_X> op;
  op.Init(x, gamma, y, tiling_data.numRows, tiling_data.hidden, tiling_data.epsilon);
  op.Process();
}
```

kernel 类用 `Init → Process(CopyIn→Compute→CopyOut)` 三段式。RMSNorm 的可复用要点：
- **多核按行切，行内不跨核**：`base = numRows/blockNum; rem = numRows%blockNum;` 前 `rem`
  个核各多分 1 行（尾块处理）。每核只 `SetGlobalBuffer` 到自己那段行的偏移。
- **fp16 全程升 fp32 算**：CopyIn 后 `Cast` 到 fp32，算完 `Cast(CAST_RINT)` 回 fp16。用
  `if constexpr (isHalf)`（`sizeof(T)==2`）编译期分支，fp32 时跳过 Cast。
- **规约后读标量**：`ReduceSum` 到 1 元素 → `Muls(1/hidden)` → `Adds(eps)` → `Rsqrt` →
  `red.GetValue(0)` 读回标量 `rstd` → `Muls(xf, rstd)` → `Mul(xf, gamma)`。
- `redBuf` 按 32B 对齐分配（`InitBuffer(redBuf, 32)`），DataCopy 要 ~32B 对齐。

### B4. 打包成 .run（整个工程一次编出所有算子）

```bash
# 干净子 shell 内：
cd /models/share/userdata/cb/AscendFast/kernels/ascendc_ops/ascendfast_custom_ops
rm -rf build_out          # 关键：清掉旧产物，避开未来时间戳导致的 CPack INSTALL 报错
bash build.sh
```

成功标志：日志出现 `Self-extractable archive "custom_opp_openEuler_aarch64.run" successfully created.`
产物 `build_out/custom_opp_openEuler_aarch64.run` 含工程内**全部**算子。
新 kernel 编译失败会让整包失败——错误日志会定位到 `<op>.cpp` 的行号，照着修。

### B5. 用官方安装器装进项目内（不碰系统）

```bash
cd .../ascendfast_custom_ops/build_out
INSTALL_DIR=/models/share/userdata/cb/AscendFast/kernels/ascendc_ops/_installed_opp
bash custom_opp_openEuler_aarch64.run --quiet --install-path=$INSTALL_DIR
```

成功打印 `SUCCESS`，刷新 `$INSTALL_DIR/vendors/customize/bin/set_env.bash`（运行时靠它）。
**不要手动 cp kernel 到 CANN 目录**——`.run` 安装器会生成 `liboptiling.so`、
`binary_info_config.json` 等手动凑不齐的文件。

## A 链：adapter → build_adapter.py → .so

### A1. 写 adapter（关键：第二个起的算子用 FRAGMENT）

复制 `kernels/csrc/adapter_rms_norm_custom.cpp` 为 `adapter_<op>.cpp`，改：`#include
"aclnn_<op>.h"`、两段式里的 `aclnn<Op>*`、schema 字符串 + 函数签名。

**命名空间冲突坑（必须懂）**：`add_demo` 已经用 `TORCH_LIBRARY(ascendfast, m)` 定义过一次
命名空间。你的新 .so 若再 `TORCH_LIBRARY(ascendfast,...)` 会**命名空间重复定义冲突**。所以
新算子一律用 `TORCH_LIBRARY_FRAGMENT`（允许多个 .so 往同一命名空间**追加** schema）：

```cpp
TORCH_LIBRARY_FRAGMENT(ascendfast, m) { m.def("<op>(Tensor x, Tensor gamma, float eps) -> Tensor"); }
TORCH_LIBRARY_IMPL(ascendfast, PrivateUse1, m) { m.impl("<op>", &ascendfast::<op>); }  // NPU=PrivateUse1
```

workspace 分配 API **本机固定写法**（别用文档常见的 `NPUWorkspaceAllocator::malloc_with_stream`，
本机无此符号）：

```cpp
#include "torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h"
at::Tensor ws;  void* wsPtr = nullptr;       // ws 必须活到 aclnn 执行完
if (wsSize > 0) { ws = at_npu::native::allocate_workspace(wsSize, stream); wsPtr = ws.data_ptr(); }
```

aclnn 两段式：`aclnn<Op>GetWorkspaceSize(<inputs>, <attrs>, <out>, &wsSize, &executor)` 拿
workspace 大小 + executor，再 `aclnn<Op>(wsPtr, wsSize, executor, stream)` 真正下发。attr
（如 `double eps`）插在 tensor 参数和 out 之间。

### A2. 编 .so（无需改 build_adapter.py）

`build_adapter.py` 已遍历 `csrc/adapter_*.cpp` 各编一个 `ascendfast_adapter_<op>.so`，按算子名
检查 `aclnn_<op>.h` 存在。所以**加新算子不用改它**，直接：

```bash
python kernels/csrc/build_adapter.py   # 看到 "built and installed: .../lib/ascendfast_adapter_<op>.so"
```

`__init__.py` 自动遍历 `lib/*.so` 全部加载，A2 编完即注册，**不用动注册入口**。

## 数值自检（收尾，决定 installed 报 true 还是 false）

source 两个 env 后，用 `arch_params` 的真实规模（≥1024 元素，**别用 64**——小 shape 会因
tiling 尾块算错）真调一次，比对 fp32 参考：

```bash
source scripts/ascend-env.sh   # 它会一并 source 上面装好的 set_env.bash
# 然后 python：
import torch, torch_npu, ascendfast_ops
x = torch.randn(64, 896, dtype=torch.float16).npu()
g = torch.randn(896, dtype=torch.float16).npu()
y = torch.ops.ascendfast.<op>(x, g, 1e-6)
torch.npu.synchronize()
ref = (x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+1e-6)*g.float())
rel = (y.float()-ref).abs().max().item()/(ref.abs().max().item()+1e-9)
print("max_rel_err", rel)   # fp16 容差 ~5e-2；超过=kernel 有 bug，回 B2/B3 调
```

过关才往 `kernels/registry.json` 追加记录、报 `installed: true` + 实测 `numeric_max_rel_err`。
跑不通/误差大就如实报 `installed: false`——`gate_operator` 会拦下谎报。

## 排错速查

| 现象 | 原因 | 解法 |
|---|---|---|
| `TORCH_LIBRARY: namespace ascendfast already defined` | 第二个算子又用了 `TORCH_LIBRARY` | 改用 `TORCH_LIBRARY_FRAGMENT`（A1） |
| `cast between floating and unsigned ... not allowed in aicore` | kernel 里 `float`↔`uint` 直接转 | 先经 `int32`：`static_cast<float>(static_cast<int32_t>(x))` |
| `file INSTALL cannot find op_kernel/binary/...`（CPack） | 文件未来时间戳 / 旧产物 | `rm -rf build_out` 后重 `build.sh` |
| `import` 即 `undefined symbol: ...malloc_with_stream` | 用了本机没有的 workspace API | 改 `at_npu::native::allocate_workspace`（A1） |
| `ImportError: ...PyInit_ 未定义` | adapter 当 Python 模块加载 | `load(is_python_module=False)`（已在脚本里） |
| `torch.ops.ascendfast.<op>` 不存在 | .so 没编/没加载 | 重跑 `build_adapter.py`；确认 `lib/*.so` 在 |
| 调用成功但数值全 0 / 等于输入 | CANN 找不到 device kernel | `source set_env.bash`；确认用 `.run` 装的、非手动 cp |
| 小 shape(64元素)算错、大 shape 对 | tiling 对齐/尾块假设整除 | 修尾块（前 rem 核多 1 行）；真实模型规模不受影响 |
| `cmake: command not found` | 机器无系统 cmake | venv 已装，干净子 shell 的 PATH 放 `.venv/bin` |

## 关键文件地图

| 角色 | 路径 |
|---|---|
| 算子工程（含全部 kernel） | `kernels/ascendc_ops/ascendfast_custom_ops/` |
| 算子原型清单 | `.../ascendfast_custom_ops/ops.json` |
| host / kernel 源码 | `.../op_host/<op>.cpp` + `_tiling.h`、`.../op_kernel/<op>.cpp` |
| .run 安装包 | `.../build_out/custom_opp_openEuler_aarch64.run` |
| 安装后的算子库 | `kernels/ascendc_ops/_installed_opp/vendors/customize/` |
| adapter 源码 | `kernels/csrc/adapter_<op>.cpp` |
| adapter 编译脚本 | `kernels/csrc/build_adapter.py`（遍历，无需改） |
| adapter 产物 .so | `kernels/src/ascendfast_ops/lib/*.so` |
| 注册入口（自动加载，无需改） | `kernels/src/ascendfast_ops/__init__.py` |
| 生成-算子清单（幂等用） | `kernels/registry.json` |
| 详细手册（背景/范例） | `kernels/docs/C2_add_operator_howto.md` |
