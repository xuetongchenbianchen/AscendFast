#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""演示：从 PyTorch 一路调到自己写的 Ascend C 算子，看清整条链。

这个脚本不是测试，是"链路解剖"——每一步都打印它在调用链的哪一层，
让你直观看到 torch.ops.ascendfast.add_demo(x, y) 到底经过了什么。

────────────────────────────────────────────────────────────────────
运行前必须 source 两个 env（缺第二个会"调得到但算不对"）：

    cd /models/share/userdata/cb/AscendFast
    source scripts/ascend-env.sh
    source kernels/ascendc_ops/_installed_opp/vendors/customize/bin/set_env.bash
    python kernels/examples/call_chain_demo.py
────────────────────────────────────────────────────────────────────

完整调用链（脚本会逐层验证）：

  ① Python          torch.ops.ascendfast.add_demo(x, y)
        ↓           x,y 是 .npu() 张量 → 后端是 PrivateUse1
  ② dispatcher      按 (命名空间 ascendfast, op add_demo, 后端 PrivateUse1) 查表
        ↓           表是 import ascendfast_ops 时 load_library(.so) 填的
  ③ adapter(C++)    ascendfast::add_demo()  [kernels/csrc/adapter_add_demo.cpp]
        ↓           at::Tensor → aclTensor，拿当前 NPU 流，调 aclnn
  ④ aclnn 两段式    aclnnAddDemoGetWorkspaceSize(...) + aclnnAddDemo(...)
        ↓           符号来自 libcust_opapi.so（B 链编出，adapter 链接它）
  ⑤ CANN runtime   按 ASCEND_CUSTOM_OPP_PATH 找到 _installed_opp 里的算子
        ↓
  ⑥ Tiling(host)   optiling::TilingFunc()  [op_host/add_demo.cpp] 决定多核切分
        ↓
  ⑦ device kernel  add_demo(...)  [op_kernel/add_demo.cpp]
                   每个 AI Core: CopyIn(GM→Local)→Compute(Add)→CopyOut(Local→GM)
"""
import os
import sys


def banner(step: str, msg: str) -> None:
    print(f"\n{'─' * 70}\n[{step}] {msg}\n{'─' * 70}")


def main() -> int:
    # ── 层①前置：import 触发注册 ───────────────────────────────────
    # 加载 ascendfast_ops 包时，__init__.py 会遍历 lib/*.so 并 load_library。
    # 加载 .so 这个动作本身会执行其中 TORCH_LIBRARY / TORCH_LIBRARY_IMPL 宏
    # 展开的全局构造函数，把 add_demo 注册进 dispatcher。没有显式的 register()。
    banner("②注册", "import ascendfast_ops —— 加载 .so，宏在加载副作用里完成注册")
    import torch
    import torch_npu  # noqa: F401  # 引入 PrivateUse1(npu) 后端
    import ascendfast_ops

    # 确认算子真的进了 dispatcher。注意：PyTorch 算子是懒加载的，dir() 在首次
    # 访问前可能看不到它——所以这里直接尝试取出 OpOverloadPacket。
    op = torch.ops.ascendfast.add_demo
    print(f"  dispatcher 里拿到算子对象: {op}")
    print(f"  类型: {type(op).__name__}  (OpOverloadPacket = 注册成功)")

    # ── 层①：构造 NPU 输入 ────────────────────────────────────────
    # 用 ≥1024 元素的真实规模（demo kernel 的 tiling 对极小输入会算错）。
    banner("①Python", "构造 .npu() 张量 —— 后端标记为 PrivateUse1，dispatcher 才会路由到 NPU 实现")
    shape = (512, 512)
    x = torch.randn(*shape, dtype=torch.float32).npu()
    y = torch.randn(*shape, dtype=torch.float32).npu()
    print(f"  x.device={x.device}  x.dtype={x.dtype}  shape={tuple(x.shape)}")
    # 张量在哪个后端，决定 dispatcher 选哪份实现。CPU 张量会落到 CPU 实现（不存在 → 报错）。
    print(f"  x 的后端 key: {x.device.type}  → dispatcher 会找 PrivateUse1 的 impl")

    # ── 层②③④⑤⑥⑦：一次调用，穿透全链 ───────────────────────────
    banner("③→⑦", "torch.ops.ascendfast.add_demo(x, y) —— 一行穿透 adapter→aclnn→CANN→tiling→device kernel")
    print("  调用中… (dispatcher→C++ adapter→aclnn 两段式→device kernel)")
    z = torch.ops.ascendfast.add_demo(x, y)
    # NPU 是异步下发，必须 synchronize 等 device kernel 真正算完再读结果。
    torch.npu.synchronize()
    print(f"  返回 z.device={z.device}  shape={tuple(z.shape)}")

    # ── 验证：device kernel 算对了 ────────────────────────────────
    banner("验证", "对照 PyTorch 自带的 x+y，确认我们的 device kernel 数值正确")
    ref = x + y  # 走官方 npu 加法，作为参考真值
    torch.npu.synchronize()
    max_diff = (z - ref).abs().max().item()
    print(f"  max|ours - (x+y)| = {max_diff:.3e}")
    ok = max_diff == 0.0
    print(f"  结果: {'PASS —— 自己写的 kernel 与官方逐元素一致' if ok else 'FAIL'}")

    # ── 反证：CPU 张量会被 dispatcher 拦下 ────────────────────────
    # 这一步是为了让你看清"后端决定路由"：同一个 op，喂 CPU 张量会因为没有
    # CPU impl 而报错——这恰恰证明我们只注册了 PrivateUse1(NPU) 那一份。
    banner("反证", "用 CPU 张量调同一个 op —— 预期报错，证明算子只在 NPU 后端注册")
    try:
        cx = torch.randn(8, 8)
        cy = torch.randn(8, 8)
        torch.ops.ascendfast.add_demo(cx, cy)
        print("  （意外：CPU 居然没报错）")
    except (RuntimeError, NotImplementedError) as e:
        first_line = str(e).strip().splitlines()[0]
        print(f"  如期报错: {first_line}")
        print("  → 说明 dispatcher 按后端选实现，我们只给了 NPU(PrivateUse1) 一份")

    return 0 if ok else 1


if __name__ == "__main__":
    # 友好提示：若没 source 第二个 env，device kernel 找不到，结果会错。
    if "ASCEND_CUSTOM_OPP_PATH" not in os.environ:
        print("[警告] 未设 ASCEND_CUSTOM_OPP_PATH。先 source "
              "kernels/ascendc_ops/_installed_opp/vendors/customize/bin/set_env.bash，"
              "否则 CANN 找不到 device kernel，结果会算错。", file=sys.stderr)
    sys.exit(main())
