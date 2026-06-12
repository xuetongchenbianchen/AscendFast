"""示例：在一个 ExecutionMode workspace 里接入 ascendfast 自定义算子库。

这是"轻胶水"层 —— 自定义算子库（kernels/ascendfast_ops，装在 venv）是重资产、
全局一份；本文件演示 patch 怎么把 nn.Linear 的 forward 改调 torch.ops.ascendfast.my_linear。
真实使用时把本文件拷进某个 workspace 的 patches/，并在 build_model() 里调用 apply()。

注意（遵 CLAUDE.md）：import ascendfast_ops 必须写在 build_model() 函数体内，
不能放模块顶层 —— 与 workspace 隔离约定一致。本文件的 import 写在 apply() 内即满足。
"""
import types

import torch
import torch.nn as nn


def apply(model):
    # import 即触发算子注册（venv 包，全 run 常驻、只注册一次）。
    import ascendfast_ops  # noqa: F401

    my_linear = torch.ops.ascendfast.my_linear

    def _new_forward(self, x):
        # my_linear 的 schema 支持任意前导维，无需像 npu_linear 那样手动 reshape 2D。
        return my_linear(x, self.weight, self.bias)

    patched = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.forward = types.MethodType(_new_forward, module)
            patched += 1
    return patched
