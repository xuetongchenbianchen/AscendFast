"""ascendfast::my_linear 的算子级数值测试。

定位：算子库的数值守门，与模型级 correctness（last-token logits 余弦）解耦。
- 模型级太粗，抓不住单算子的数值 bug；这里对算子直接逐元素 allclose。
- C1 占位实现下应全绿；切到 C2 device kernel 后，这套测试不变，直接当回归基线
  ——kernel 数值错会在这里暴露，而不是等到模型 forward 才发现。

参考实现固定为 PyTorch 原生 x@w.T(+b)，是"正确"的金标准。
"""
from __future__ import annotations

import pytest
import torch

import ascendfast_ops  # noqa: F401 —— import 触发算子注册


def _ref(x, w, b):
    out = torch.matmul(x, w.t())
    return out + b if b is not None else out


def test_op_is_registered():
    assert "ascendfast.my_linear" in ascendfast_ops.registered_ops()


@pytest.mark.parametrize("shape", [(4, 16), (2, 3, 16), (1, 1, 16)])
def test_my_linear_matches_reference(shape):
    torch.manual_seed(0)
    in_features = shape[-1]
    out_features = 8
    x = torch.randn(*shape)
    w = torch.randn(out_features, in_features)
    b = torch.randn(out_features)

    out = torch.ops.ascendfast.my_linear(x, w, b)
    ref = _ref(x, w, b)

    assert tuple(out.shape) == (*shape[:-1], out_features)
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_my_linear_no_bias():
    torch.manual_seed(1)
    x = torch.randn(4, 16)
    w = torch.randn(8, 16)
    out = torch.ops.ascendfast.my_linear(x, w, None)
    assert torch.allclose(out, _ref(x, w, None), rtol=1e-4, atol=1e-5)


def test_fake_shape_inference():
    """register_fake 的形状推断须与真实输出一致（torch.compile/图模式依赖它）。"""
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(2, 3, 16)
        w = torch.empty(8, 16)
        out = torch.ops.ascendfast.my_linear(x, w, None)
        assert tuple(out.shape) == (2, 3, 8)
