# C2 自定义算子接入手册 —— 添加一个新算子照着走

> 本文是 **可复现的操作步骤**。背景概念(调用链为什么这么走、PrivateUse1 是什么)见
> [`C2_operator_delivery_contract.md`](./C2_operator_delivery_contract.md)。
>
> 以已跑通的 `add_demo`（`z = x + y`）为范例。把新算子叫 `myop`，照着把
> `add_demo`/`AddDemo` 换成 `myop`/`MyOp` 即可。
>
> **本机事实(写文档时实测)**：
> - soc_version：`Ascend910_9382`，简写 `ascend910_93`
> - CANN：`/usr/local/Ascend/cann-8.5.0`
> - torch_npu：2.7.1.post2（venv `.venv`）
> - 机器**没有系统级 cmake**，cmake 装在项目 venv 里（见步骤 0）

---

## 全景：一个算子要凑齐两条独立的链

```
A. PyTorch 接入链(让 torch.ops.ascendfast.myop 能调用)
   adapter_myop.cpp  --build_adapter.py-->  .so  --__init__.py 加载-->  注册进 dispatcher

B. device kernel 链(让算子在 NPU 上真正算)
   op_host + op_kernel  --build.sh-->  custom_opp_*.run  --安装-->  CANN 能找到 kernel
```

两条都通，且运行时 `source` 了正确的 env，算子才既“调得到”又“算得对”。

---

## 步骤 0：一次性环境准备（已做过可跳过）

```bash
cd /models/share/userdata/cb/AscendFast
# cmake 不是 Python 包但 build.sh 需要它，装进项目 venv（不污染系统）
VIRTUAL_ENV=$PWD/.venv UV_LINK_MODE=copy uv pip install cmake
.venv/bin/cmake --version    # 确认可用
```

每次开新终端跑下面任何命令前，都要有 CANN + venv 环境。本文统一用这个“干净子 shell”
模板（机器的 shell snapshot 有 `set -u` 坑，且 coreutils 不全，故用 `env -i` 起干净环境）：

```bash
env -i bash --noprofile --norc -c '
set +u
export PATH=/models/share/userdata/cb/AscendFast/.venv/bin:/usr/bin:/bin:/usr/local/bin
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export VIRTUAL_ENV=/models/share/userdata/cb/AscendFast/.venv
export PATH=$VIRTUAL_ENV/bin:$PATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH=/models/share/userdata/cb/AscendFast
# <你的命令写这里>
'
```

---

## B 链：写 kernel → 打包 → 安装

> **统一工程（方案 A）**：所有自定义算子都加进**同一个** msopgen 标准工程
> `kernels/ascendc_ops/ascendfast_custom_ops/`，**不再**为每个算子单独 `msopgen gen`。
> 工程已经存在，加算子 = 往里加文件，不重新生成。

### B1. 往算子原型清单里加一条

编辑 `kernels/ascendc_ops/ascendfast_custom_ops/ops.json`，在算子数组里**追加**一项
（参考已有的 `AddDemo`）：定义算子名（大驼峰 `MyOp`）、输入输出的 name/dtype/format。

### B2. 加 host / kernel 实现（每个算子各一份 .cpp）

工程的 `op_host/CMakeLists.txt` 第一行是
`aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ops_srcs)`——它自动收集目录下所有
`.cpp`，所以**新增算子不用改任何 CMake**，把文件放进去即可：

```
ascendfast_custom_ops/
  op_host/
    add_demo.cpp  add_demo_tiling.h     ← 已有
    myop.cpp      myop_tiling.h          ← 新增
  op_kernel/
    add_demo.cpp                        ← 已有
    myop.cpp                            ← 新增
```

> 写新算子可拿 `add_demo.cpp` 当模板照着改：host 里改 tiling/shape 推导和
> `REGISTER_OP` 的算子名，kernel 里改 Compute 的计算逻辑。

确认 `CMakePresets.json` 里这几项对（工程级，只需确认一次）：
- `ASCEND_CANN_PACKAGE_PATH` = `/usr/local/Ascend/cann-8.5.0`
- `ASCEND_COMPUTE_UNIT` = `ascend910_93`
- `vendor_name` = `customize`
- `ENABLE_BINARY_PACKAGE` = `True`

并确认新算子 `op_host/myop.cpp` 里有 `this->AICore().AddConfig("ascend910_93");`

> **kernel tiling 注意**：device 的 `DataCopy` 要求 ~32 字节对齐（fp32 = 8 元素）。
> demo 的 8 核×8 块切分对“正好 64 元素”这种小输入会因切片过小而算错（实测 64 元素
> 返回的是输入本身）。实际模型张量都远大于此（≥1024 元素验证零误差），不受影响；
> 但写新算子时 tiling 要处理对齐和尾块，别照抄 demo 的“假设整除”。

### B3. 重新打包成 .run（整个工程一次编出所有算子）

```bash
# 在干净子 shell 里：
cd /models/share/userdata/cb/AscendFast/kernels/ascendc_ops/ascendfast_custom_ops
bash build.sh
```

成功标志（日志里出现）：
```
Self-extractable archive "custom_opp_openEuler_aarch64.run" successfully created.
```
产物：`ascendfast_custom_ops/build_out/custom_opp_openEuler_aarch64.run`
（这一个 .run 里含工程内**全部**算子的 kernel。）

> **不要手动 cp kernel 文件到 CANN 目录**。`.run` 安装器会生成 `liboptiling.so`、
> `binary_info_config.json` 等手动凑不齐的文件——这是之前手动安装失败的根因。

### B4. 用官方安装器安装（装到项目内，不碰系统）

```bash
# 在干净子 shell 里：
cd .../ascendfast_custom_ops/build_out
INSTALL_DIR=/models/share/userdata/cb/AscendFast/kernels/ascendc_ops/_installed_opp
bash custom_opp_openEuler_aarch64.run --quiet --install-path=$INSTALL_DIR
```
成功打印 `SUCCESS`，并生成
`$INSTALL_DIR/vendors/customize/bin/set_env.bash`（运行时靠它）。

---

## A 链：写 adapter → 编 .so → 注册

### A1. 写 adapter

复制 `kernels/csrc/adapter_add_demo.cpp` 为 `adapter_myop.cpp`，改三处：
1. `#include "aclnn_myop.h"`（B 链编出的 aclnn 头）
2. `EXEC` 两段式里的 `aclnnMyOp*`
3. schema 字符串 + 函数签名（按算子输入输出）

**本机 workspace 分配 API 固定写法**（torch_npu 2.7.1 真实导出的，别用文档里常见的
`c10_npu::NPUWorkspaceAllocator::malloc_with_stream`，本机没有那个符号）：
```cpp
#include "torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h"
// ...
at::Tensor ws;                  // 返回 at::Tensor，必须存活到 aclnn 执行完
void* wsPtr = nullptr;
if (wsSize > 0) {
    ws = at_npu::native::allocate_workspace(wsSize, stream);
    wsPtr = ws.data_ptr();
}
```

末尾注册照抄（命名空间统一 `ascendfast`）：
```cpp
TORCH_LIBRARY(ascendfast, m)              { m.def("myop(Tensor x, Tensor y) -> Tensor"); }
TORCH_LIBRARY_IMPL(ascendfast, PrivateUse1, m) { m.impl("myop", &ascendfast::myop); }
```

### A2. 编译 adapter 成 .so

参照 `kernels/csrc/build_adapter.py`（关键点已踩平，照抄即可）：
- `load(..., is_python_module=False)` —— adapter 不是 Python 模块，只靠 TORCH_LIBRARY 注册；
  不加这个会报 `PyInit_ 未定义`。
- 产物 .so 复制到 `kernels/src/ascendfast_ops/lib/`。

```bash
# 在干净子 shell 里：
python kernels/csrc/build_adapter.py     # 看到 "built and installed: .../lib/xxx.so" 即成功
```

### A3. 注册入口（无需改动）

`kernels/src/ascendfast_ops/__init__.py` 会**自动遍历 `lib/*.so` 全部加载**，
所以 A2 把新算子的 .so 编进 `lib/` 后，这里**不用动**。

---

## 运行 & 验证

每次新终端、跑任何用到算子的代码前，**source 两个 env**：
```bash
source scripts/ascend-env.sh
source kernels/ascendc_ops/_installed_opp/vendors/customize/bin/set_env.bash   # 设 ASCEND_CUSTOM_OPP_PATH
```

验证（用真实规模 ≥1024 元素，别用 64）：
```python
import torch, torch_npu, ascendfast_ops
x = torch.randn(512, 512).npu()
y = torch.randn(512, 512).npu()
r = torch.ops.ascendfast.myop(x, y)
torch.npu.synchronize()
print((r - (x + y)).abs().max().item())   # 期望 0.0
```

---

## 排错速查（按现象）

| 现象 | 原因 | 解法 |
|---|---|---|
| `import` 即 `undefined symbol: ...malloc_with_stream` | adapter 用了本机没有的 workspace API | 改成 `at_npu::native::allocate_workspace`（见 A1） |
| `ImportError: ...PyInit_ 未定义` | adapter 当成 Python 模块加载 | `load(is_python_module=False)` |
| `torch.ops.ascendfast.xxx` 不存在 | .so 没加载 / `__init__.py` 没加载它 | 确认 `lib/*.so` 在、`load_library` 调了 |
| `Could not run with arguments from 'CPU' backend` | 用了 CPU tensor 调 NPU 算子 | 输入 `.npu()` |
| 调用成功但数值全 0 / 等于输入 | CANN 找不到 device kernel | `source set_env.bash`；确认用 `.run` 装的、不是手动 cp |
| `LoadSo: liboptiling.so ... No such file` | 手动 cp 漏了 tiling 库 | 别手动装，用 `.run`（B4/B5） |
| 小 shape(64元素)算错、大 shape 对 | kernel tiling 对齐/尾块 | 真实规模不受影响；写新算子时修 tiling |
| `cmake: command not found` | 机器无系统 cmake | venv 已装，PATH 放 `.venv/bin`（步骤 0） |

---

## 关键文件地图

| 角色 | 路径 |
|---|---|
| adapter 源码 | `kernels/csrc/adapter_<op>.cpp` |
| adapter 编译脚本 | `kernels/csrc/build_adapter.py` |
| adapter 产物 .so | `kernels/src/ascendfast_ops/lib/*.so` |
| 注册入口 | `kernels/src/ascendfast_ops/__init__.py` |
| 算子工程(含全部 kernel) | `kernels/ascendc_ops/ascendfast_custom_ops/` |
| 算子原型清单 | `.../ascendfast_custom_ops/ops.json` |
| .run 安装包 | `.../ascendfast_custom_ops/build_out/custom_opp_openEuler_aarch64.run` |
| 安装后的算子库 | `kernels/ascendc_ops/_installed_opp/vendors/customize/` |
| 运行时 env | `.../_installed_opp/vendors/customize/bin/set_env.bash` |
