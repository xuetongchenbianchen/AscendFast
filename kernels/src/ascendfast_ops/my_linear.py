"""ascendfast::my_linear —— 自定义 linear 算子（C1 占位实现）。

语义：out = x @ w.T + b，与 nn.Linear 一致。
- x:  [..., in_features]   任意前导维（forward patch 里会 reshape 成 2D 再调）
- w:  [out_features, in_features]
- b:  [out_features] 或 None
- out:[..., out_features]

这是 C1（纯 Python）阶段：实现体用 torch 原生算子，作用有二——
  1. 验证"注册→dispatcher→build_model 调用→门禁→correctness"整条接入链路；
  2. 当 C2 device kernel 的数值参考基线（单元测试拿它对 allclose）。

C2 切换点：见下方 `_impl` 里的 TODO。届时实现体改为
  torch.ops.ascendfast._my_linear_npu(x, w, b)   # 由编译好的 .so 注册
其余（schema、register_fake、对外算子名）一律不动。
"""
from __future__ import annotations

import torch

# 算子的对外 schema。命名空间 ascendfast，算子名 my_linear。
# Tensor? b 表示 bias 可选。这份 schema 是 C1/C2 共同的不变契约。
_SCHEMA = "ascendfast::my_linear(Tensor x, Tensor w, Tensor? b) -> Tensor"


@torch.library.custom_op("ascendfast::my_linear", mutates_args=())
def my_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    return _impl(x, w, b)


def _impl(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    # ---- C1 占位：纯 torch 实现 -------------------------------------------
    # ---- C2 替换：return torch.ops.ascendfast._my_linear_npu(x, w, b) -----
    out = torch.matmul(x, w.t())
    if b is not None:
        out = out + b
    return out


@my_linear.register_fake
def _(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    """shape/dtype 推断（torch.compile / torchair 图捕获时用，不实际算）。

    输出形状 = x 的前导维 + [w.shape[0]]，dtype/device 跟随 x。
    必须与 _impl 的真实输出形状一致，否则图模式下会形状不符。
    """
    return x.new_empty((*x.shape[:-1], w.shape[0]))
