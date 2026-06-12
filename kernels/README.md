# ascendfast_ops —— AscendFast 自定义高性能算子库

把自己写的 NPU 算子注册进 `torch.ops.ascendfast.*` 命名空间，和官方
`torch.ops.npu.*` 在 PyTorch dispatcher 面前平起平坐。任意 ExecutionMode
workspace 的 `build_model()` 里 `import ascendfast_ops` 一次即可调用。

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

## 测试

```bash
.venv/bin/python -m pytest kernels/tests/ -v
```

算子级数值测试，与模型级 correctness 解耦：对每个算子和 PyTorch 参考实现逐元素
`allclose`。C1 占位应全绿；切到 C2 device kernel 后这套测试不变，直接当回归基线。

## 在 workspace 里使用

见 `examples/use_my_linear_patch.py`。要点：`import ascendfast_ops` 写在
`build_model()` 函数体内（遵 CLAUDE.md），调用点为 `torch.ops.ascendfast.<op>(...)`。

## 当前算子

| 算子 | schema | 说明 |
|---|---|---|
| `my_linear` | `(Tensor x, Tensor w, Tensor? b) -> Tensor` | `x @ w.T + b`，支持任意前导维。当前为 C1 纯 Python 占位。 |

## C1 → C2 路线

- **C1（当前）**：`src/ascendfast_ops/<op>.py` 里实现体用纯 torch，验证接入链路 +
  当数值基线。
- **C2（目标）**：Ascend C 写 kernel → `msopgen` 建工程 → 编出 custom op 包 →
  C++ 适配层（`EXEC_NPU_CMD` + `TORCH_LIBRARY` 注册到 PrivateUse1）→ 编成 `.so`
  放 `lib/`。届时只改各算子模块的实现体（`_impl` 里的 TODO 切换点），**对外
  schema 与调用点 `torch.ops.ascendfast.<op>` 永不变**。

新增算子：在 `src/ascendfast_ops/` 加 `<op>.py`（custom_op + register_fake），在
`__init__.py` 追加一行 import，在 `tests/` 加对应数值测试，更新上表。
