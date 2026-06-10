"""从 ExecutionMode 的 workspace 物化 (model, tokenizer) 的公共加载器。

每个 ExecutionMode 都是一个自包含、可运行的目录，通过 entrypoint（默认
build_model.py）暴露统一入口：

    build_model() -> (model, tokenizer)

无论 workspace 里嵌的是哪种优化（forward patch / 算子融合 / 量化 / ...），
correctness / profile / benchmark 都**只**通过这个入口加载模型——这是全项目
唯一的模型真相源。加载逻辑本身与 profiling/benchmark 无关，所以独立成模块，
避免各功能去 import profile_runner.py 的私有实现。

注意：本模块不依赖 torch，只负责 import workspace 的 build_model.py 并调用它；
具体 device / dtype 由 build_model() 自身决定。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from models import ExecutionMode


def load_build_model(mode: ExecutionMode) -> tuple[Any, Any]:
    """加载 mode.workspace_dir 的 build_model()，返回 (model, tokenizer)。

    Args:
        mode: 一个自包含可运行的 ExecutionMode；其 entrypoint 必须暴露
              build_model() -> (model, tokenizer)。

    Raises:
        FileNotFoundError: entrypoint 文件不存在。
        ImportError:       entrypoint 无法作为模块加载。
        AttributeError:    entrypoint 未暴露 build_model()。
    """
    ws = Path(mode.workspace_dir).resolve()
    entry = ws / mode.entrypoint
    if not entry.is_file():
        raise FileNotFoundError(f"entrypoint not found: {entry}")

    # 隔离：每个 fork 的 build_model.py 可能 import 同名辅助文件（patches/config/...）。
    # 若把 ws 永久留在 sys.path、把这些裸名模块永久留在 sys.modules，下一个 fork 会
    # 拿到上一个 fork 缓存的同名模块——profile 到错的代码，静默污染"比较延迟"本身。
    # 因此进来前快照、用完后还原：只清掉本次 import 新引入的裸名模块与本次加进的路径。
    path_added = str(ws) not in sys.path
    if path_added:
        sys.path.insert(0, str(ws))
    modules_before = set(sys.modules)
    try:
        spec = importlib.util.spec_from_file_location(f"_mode_entry_{ws.name}", entry)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load entrypoint: {entry}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "build_model"):
            raise AttributeError(f"{entry} does not expose build_model()")
        model, tokenizer = module.build_model()
        return model, tokenizer
    finally:
        # 还原：只删"文件位于本 workspace 内"的新缓存模块（即 fork 自带的入口与裸名
        # 辅助文件 patches/config/...），让下一个 fork 不会拿到本 fork 的同名模块。
        # 刻意不动 transformers_modules.* 等 trust_remote_code 动态模块——刚建好的
        # model 仍依赖它们留在 sys.modules，全删会反噬当前模型。
        ws_str = str(ws)
        for name in set(sys.modules) - modules_before:
            mod = sys.modules.get(name)
            mod_file = getattr(mod, "__file__", None)
            if mod_file and str(Path(mod_file).resolve()).startswith(ws_str):
                del sys.modules[name]
        if path_added and str(ws) in sys.path:
            sys.path.remove(str(ws))
