"""把 add_demo 的 PyTorch 适配层 (adapter_add_demo.cpp) 即时编成扩展 .so。

路线 A：adapter 手写 aclnn 两段式，只依赖公开头。本脚本负责把它和
torch_npu / CANN / 本地编出的 libcust_opapi.so 链在一起。

用法（必须先 source CANN 环境，让 ASCEND_TOOLKIT_HOME 生效）：
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    .venv/bin/python kernels/csrc/build_adapter.py

import 返回的扩展后，torch.ops.ascendfast.add_demo 即注册进 PyTorch。
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# 本地 add_demo 算子工程的编译产物。
_ADDDEMO = _HERE.parent / "ascendc_ops" / "add_demo" / "AddDemo" / "build_out"


def _cann_home() -> Path:
    home = os.environ.get("ASCEND_TOOLKIT_HOME")
    if not home:
        raise RuntimeError(
            "ASCEND_TOOLKIT_HOME 未设置。先 source "
            "/usr/local/Ascend/ascend-toolkit/set_env.sh"
        )
    return Path(home)


def _torch_npu_dir() -> Path:
    import torch_npu

    return Path(torch_npu.__file__).resolve().parent


def build():
    from torch.utils.cpp_extension import load

    cann = _cann_home()
    tnpu = _torch_npu_dir()

    aclnn_inc = _ADDDEMO / "op_api" / "include"      # aclnn_add_demo.h
    opapi_lib = _ADDDEMO / "op_api" / "lib"            # libcust_opapi.so
    for p in (aclnn_inc / "aclnn_add_demo.h", opapi_lib / "libcust_opapi.so"):
        if not p.exists():
            raise RuntimeError(f"缺少算子产物：{p}（先在 AddDemo/ 下 build.sh）")
    # __CONTINUE_HERE__

    include_dirs = [
        str(tnpu / "include"),
        str(tnpu / "include" / "third_party" / "acl" / "inc"),
        str(cann / "include"),
        str(cann / "include" / "aclnn"),
        str(aclnn_inc),
    ]
    library_dirs = [
        str(tnpu / "lib"),
        str(cann / "lib64"),
        str(opapi_lib),
    ]
    # cust_opapi: 本地算子的 host 入口（aclnnAddDemo*）。
    # ascendcl/nnopbase: aclCreateTensor / aclnn 执行器底座。
    # torch_npu: getCurrentNPUStream / NPUWorkspaceAllocator。
    libraries = ["torch_npu", "cust_opapi", "ascendcl", "nnopbase"]

    ext = load(
        name="ascendfast_adapter_add_demo",
        sources=[str(_HERE / "adapter_add_demo.cpp")],
        extra_include_paths=include_dirs,
        extra_ldflags=(
            [f"-L{d}" for d in library_dirs]
            + [f"-Wl,-rpath,{d}" for d in library_dirs]
            + [f"-l{l}" for l in libraries]
        ),
        verbose=True,
    )
    print(f"[build_adapter] built {ext.__file__}")
    return ext


if __name__ == "__main__":
    build()
