# ascendfast_ops —— AscendFast 自定义高性能算子库

把自己写的 Ascend C 算子注册进 `torch.ops.ascendfast.*` 命名空间，和官方
`torch.ops.npu.*` 在 PyTorch dispatcher 面前平起平坐。任意 ExecutionMode
workspace 的 `build_model()` 里 `import ascendfast_ops` 一次即可调用。

当前实现是 **C2**：算子在 NPU 上由编译好的 Ascend C device kernel 真正执行
（不是 Python 占位）。已跑通的范例算子是 `add_demo`（`z = x + y`）。

## 目录结构：两条独立的链 + 一个注册入口

```
kernels/
├── src/ascendfast_ops/          【A·注册入口】venv Python 包（uv pip install -e）
│   ├── __init__.py              #  import 它 → 自动遍历加载 lib/*.so → 算子进 dispatcher
│   └── lib/*.so                 #  ← A 链产物（PyTorch 加载它）
│
├── csrc/                        【A·适配层】C++ 胶水
│   ├── adapter_add_demo.cpp     #  at::Tensor ↔ aclTensor，调 aclnn 两段式
│   └── build_adapter.py         #  把上面的 .cpp 编成 lib/*.so
│
├── ascendc_ops/                 【B·device kernel】真正在 NPU 上算
│   ├── ascendfast_custom_ops/   #  msopgen 标准工程（所有算子都加进这一个）
│   │   ├── ops.json             #    算子原型清单（名字/输入输出/dtype）
│   │   ├── op_host/<op>.cpp      #    host: tiling（多核切分）+ shape 推导
│   │   ├── op_kernel/<op>.cpp    #    kernel: CopyIn→Compute→CopyOut
│   │   └── build_out/*.run       #    ← B 链产物：算子安装包
│   └── _installed_opp/          #  .run 装进来的运行时算子库（CANN 加载它）
│
└── docs/                        操作手册 + 交付契约
```

- **A 链产物** = `src/ascendfast_ops/lib/*.so` —— PyTorch `load_library` 加载，
  靠 `TORCH_LIBRARY` 宏把算子注册进 dispatcher。
- **B 链产物** = `build_out/*.run` → 装进 `_installed_opp/` —— CANN 运行时按
  `ASCEND_CUSTOM_OPP_PATH` 找到它，提供真正的 tiling + device kernel。
- 两链的**接缝**：adapter 编译时链接 B 链编出的 `libcust_opapi.so` 和
  `aclnn_<op>.h`。所以**必须先 build B 链，再 build A 链**。

## 为什么是独立的 venv 包，而不是放进 patches/

每个 fork 的 workspace 会**独立拷贝** `patches/`。编译好的 `.so`、需重编的 kernel
源码是重资产，绝不能每 fork 拷一份。本包装在 venv（`uv pip install -e kernels/`）：

- 不在 `adaptations/` 下 → fork 不拷贝，算子库本体全局一份、所有 mode 共享。
- `workspace_loader` 的 sys.modules 隔离只清理路径落在 workspace 内的模块；本包
  路径在 venv 里 → 整个 run 常驻、**只注册一次**（重复注册会撞 duplicate def）。

## 安装

```bash
VIRTUAL_ENV=$PWD/.venv UV_LINK_MODE=copy uv pip install -e kernels/
```

本 venv 由 **uv** 管理，没有 pip。装依赖一律用 `uv pip install`。

## 运行前置：source 两个 env

```bash
source scripts/ascend-env.sh                                                   # CANN + venv + torch_npu
source kernels/ascendc_ops/_installed_opp/vendors/customize/bin/set_env.bash   # ASCEND_CUSTOM_OPP_PATH
```

第二个让 CANN 能找到自定义算子的 device kernel；不 source 会"调得到但算不对"。

## 在 workspace 里使用

`build_model()` 函数体内（遵 CLAUDE.md，import 不放模块顶层）：

```python
def build_model(...):
    import ascendfast_ops            # 触发一次注册
    ...
    out = torch.ops.ascendfast.add_demo(x, y)   # 调用点
```

## 从 PyTorch 一路调到 NPU 的链条

完整链路演示见 `docs/` 同目录的调用脚本说明，简版：

```
torch.ops.ascendfast.add_demo(x,y)            # ① Python
  → PyTorch dispatcher 查 (ascendfast, add_demo, PrivateUse1)   # ②
  → ascendfast::add_demo()  [csrc/adapter_add_demo.cpp]         # ③ C++ 胶水
  → aclnnAddDemoGetWorkspaceSize + aclnnAddDemo                 # ④ aclnn 两段式
  → CANN 按 ASCEND_CUSTOM_OPP_PATH 找到算子                       # ⑤
  → optiling::TilingFunc()  [op_host/add_demo.cpp]              # ⑥ 多核切分
  → add_demo(...)  [op_kernel/add_demo.cpp]                     # ⑦ device kernel
```

## 当前算子

| 算子 | schema | 说明 |
|---|---|---|
| `add_demo` | `(Tensor x, Tensor y) -> Tensor` | `z = x + y`，逐元素，fp16/fp32，ND。C2 device kernel。范例用，验证整条接入链。 |

> `add_demo` 是教学范例：tiling 假设元素数能被 `8*8` 整除，对极小输入（如 64 元素）
> 会算错；真实规模（≥1024 元素）实测零误差。写自己的算子时 tiling 要处理对齐和尾块。

## 加一个新算子

照 `docs/C2_add_operator_howto.md` 走，三步：

```
B链：ops.json 加一条 + op_host/<op>.cpp + op_kernel/<op>.cpp → build.sh → 装 .run
A链：csrc/adapter_<op>.cpp → python csrc/build_adapter.py → 出 lib/*.so
注册：__init__.py 不用动（已自动遍历 lib/*.so）
```
