"""AscendFast 自定义算子库 —— 统一注册入口。

设计：所有自定义高性能算子都注册进 `torch.ops.ascendfast.*` 命名空间，和官方
`torch.ops.npu.*` 在 PyTorch dispatcher 面前平起平坐。任意 ExecutionMode workspace
的 build_model() 里 `import ascendfast_ops` 一次，即可 `torch.ops.ascendfast.<op>(...)`。

本包装在 venv（pip install -e kernels/），**不在 adaptations/ 下**：
- fork workspace 时不会被拷贝（算子库本体一份，所有 mode 共享）。
- workspace_loader 的 sys.modules 隔离只清理路径落在 workspace 内的模块，本包路径
  在 venv 里，因此整个 run 常驻、只注册一次（重复注册会撞 duplicate def 错）。

C2 实现：通过编译好的 C++ 适配层（.so）加载 Ascend C kernel。
- 每个算子的适配层在 kernels/csrc/adapter_<op>.cpp
- 编译脚本：kernels/csrc/build_adapter.py
- import 本包时自动加载编译好的 .so，把算子注册进 PyTorch dispatcher
"""
from __future__ import annotations
import sys
from pathlib import Path

# C2 算子：动态加载 lib/ 下所有编译好的 .so（每个 adapter_<op>.cpp 编一份）。
# 编译：cd kernels && source /path/to/ascend-env.sh && python csrc/build_adapter.py
# 自动遍历——加新算子只要把它的 .so 编进 lib/，无需改这里。
_LIB_DIR = Path(__file__).parent / "lib"
_adapter_sos = sorted(_LIB_DIR.glob("*.so"))

if _adapter_sos:
    # 加载 .so 会触发其中的 TORCH_LIBRARY / TORCH_LIBRARY_IMPL，把算子注册进
    # PyTorch dispatcher。load_library 专门用于加载这种"非 Python 模块"的算子库。
    import torch
    for _so in _adapter_sos:
        torch.ops.load_library(str(_so))
else:
    print(f"[ascendfast_ops] Warning: no *.so in {_LIB_DIR}. "
          f"Run 'python kernels/csrc/build_adapter.py' to compile C2 kernels.",
          file=sys.stderr)

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
