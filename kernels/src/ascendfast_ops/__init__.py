"""AscendFast 自定义算子库 —— 统一注册入口。

设计：所有自定义高性能算子都注册进 `torch.ops.ascendfast.*` 命名空间，和官方
`torch.ops.npu.*` 在 PyTorch dispatcher 面前平起平坐。任意 ExecutionMode workspace
的 build_model() 里 `import ascendfast_ops` 一次，即可 `torch.ops.ascendfast.<op>(...)`。

本包装在 venv（pip install -e kernels/），**不在 adaptations/ 下**：
- fork workspace 时不会被拷贝（算子库本体一份，所有 mode 共享）。
- workspace_loader 的 sys.modules 隔离只清理路径落在 workspace 内的模块，本包路径
  在 venv 里，因此整个 run 常驻、只注册一次（重复注册会撞 duplicate def 错）。

两类实现走同一注册路径：
- C1（当前）：纯 Python 占位实现，验证接入链路、当数值参考基线。
- C2（目标）：加载编译好的 Ascend C 算子（lib/*.so），把同名 op 的 PrivateUse1
  内核换成真 device kernel；Python 占位降级为 CPU/Meta 回退或删除。

切换 C1→C2 只改各算子模块的实现体，**对外 schema 与调用点 `torch.ops.ascendfast.<op>`
永不变**——这是整个结构可复用的关键。
"""
from __future__ import annotations

# 注册是 import 副作用：导入各算子模块即把它们登记进 dispatcher。
# 新增算子时在此追加一行 import，保持"import 包 = 拿到全部算子"的契约。
from . import my_linear as _my_linear  # noqa: F401

__all__ = ["registered_ops"]


def registered_ops() -> list[str]:
    """返回本库已注册的算子全名，供 smoke / 调试确认注册成功。"""
    import torch

    ns = torch.ops.ascendfast
    # 只取真正的算子（OpOverloadPacket），滤掉命名空间对象自带的属性（如 .name）。
    names = []
    for n in dir(ns):
        if n.startswith("_"):
            continue
        if isinstance(getattr(ns, n), torch._ops.OpOverloadPacket):
            names.append(f"ascendfast.{n}")
    return names
