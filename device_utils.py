"""设备运行时工具：profile / benchmark / correctness 三个评测模块共用的公共 API。

这些函数原先私藏在 profile_npu.py（1397 行，名字暗示「只做 profile」）里，但
benchmark / correctness / profile_runner 都需要它们来：导入 torch、读模型真实所在设备、
跑一次 forward、forward 后同步、释放显存。它们与 profiling 本身无关，是纯粹的设备工具，
因此独立成本模块作为单一真相源——评测模块依赖这里，不再去 import profiler 的私有实现。

profile_npu.py 通过 re-import 复用这里的实现（并保留其内部惯用的下划线别名），所以
profiler 一侧行为完全不变。
"""
from __future__ import annotations

import importlib.util
import os
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DeviceSpec:
    kind: str
    device: str
    name: str
    count: int = 1
    memory_gb: float = 0.0
    peak_tflops_fp16: float = 0.0
    peak_bandwidth_gb_s: float = 0.0


def _ensure_stdlib_profile_module() -> None:
    """Ensure torch_npu sees the stdlib profile module."""
    current = sys.modules.get("profile")
    current_file = Path(getattr(current, "__file__", "")) if current is not None else None
    stdlib_dir = Path(sysconfig.get_path("stdlib") or "")
    stdlib_profile = stdlib_dir / "profile.py"
    if current_file is not None and current_file == stdlib_profile:
        return
    if not stdlib_profile.exists():  # pragma: no cover - defensive for unusual Python builds
        return
    spec = importlib.util.spec_from_file_location("profile", stdlib_profile)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["profile"] = module
    spec.loader.exec_module(module)


def import_torch() -> Any:
    _ensure_stdlib_profile_module()
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on host env
        raise RuntimeError(
            "PyTorch is required to capture a profile. Install the ascend or cuda extra first."
        ) from exc
    return torch


def _npu_device_spec(torch: Any, index: int, torch_npu: Any) -> tuple[DeviceSpec, Any]:
    """Build a DeviceSpec for npu:<index>, reading that card's own properties."""
    props = torch.npu.get_device_properties(index)
    total_memory = float(getattr(props, "total_memory", 0.0) or 0.0)
    return (
        DeviceSpec(
            kind="npu",
            device=f"npu:{index}",
            name=str(torch.npu.get_device_name(index)),
            count=int(torch.npu.device_count()),
            memory_gb=round(total_memory / (1024 ** 3), 1) if total_memory else 0.0,
        ),
        torch_npu,
    )


def _cuda_device_spec(torch: Any, index: int) -> tuple[DeviceSpec, None]:
    """Build a DeviceSpec for cuda:<index>, reading that card's own properties."""
    props = torch.cuda.get_device_properties(index)
    total_memory = float(getattr(props, "total_memory", 0.0) or 0.0)
    return (
        DeviceSpec(
            kind="cuda",
            device=f"cuda:{index}",
            name=str(torch.cuda.get_device_name(index)),
            count=int(torch.cuda.device_count()),
            memory_gb=round(total_memory / (1024 ** 3), 1) if total_memory else 0.0,
        ),
        None,
    )


def device_spec_for(model: Any) -> tuple[DeviceSpec, Any | None]:
    """以模型真身所在设备为权威，构造 DeviceSpec —— 不再独立猜测放哪。

    build_model() 已经把模型 .to(device) 了；消费端（correctness/benchmark/profile）
    应当读模型真实所在的设备，而非再 detect_device("auto") 重新猜一遍——后者在多卡
    下可能猜到另一张卡，触发静默跨卡搬运，污染 benchmark/profile 数据甚至 OOM。

    这里直接取模型参数所在 device（ground truth），并读该卡自己的属性填充
    name/memory_gb（不再硬编码 0 号卡）。CPU 模型走 cpu 分支。
    """
    torch = import_torch()
    dev = next(model.parameters()).device
    index = dev.index if dev.index is not None else 0
    if dev.type == "npu":
        try:
            import torch_npu as loaded_torch_npu  # type: ignore[import-not-found]
        except Exception:
            loaded_torch_npu = None
        return _npu_device_spec(torch, index, loaded_torch_npu)
    if dev.type == "cuda":
        return _cuda_device_spec(torch, index)
    return DeviceSpec(kind="cpu", device="cpu", name="CPU", count=1), None


def detect_device(prefer: str = "auto", *, allow_cpu: bool = False) -> tuple[DeviceSpec, Any | None]:
    """Detect the active profiling device and return optional torch_npu module."""
    torch = import_torch()
    preferred = prefer.lower().strip()
    torch_npu = None
    if preferred in {"auto", "npu"}:
        try:
            import torch_npu as loaded_torch_npu  # type: ignore[import-not-found]

            torch_npu = loaded_torch_npu
        except Exception:
            torch_npu = None
        if torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available():
            return _npu_device_spec(torch, 0, torch_npu)
        if preferred == "npu":
            raise RuntimeError("Requested NPU profiling, but torch_npu/torch.npu is not available.")

    if preferred in {"auto", "cuda"} and torch.cuda.is_available():
        return _cuda_device_spec(torch, 0)
    if preferred == "cuda":
        raise RuntimeError("Requested CUDA profiling, but CUDA is not available.")

    if allow_cpu or preferred == "cpu":
        return DeviceSpec(kind="cpu", device="cpu", name="CPU", count=1), None
    raise RuntimeError("No Ascend NPU or CUDA device is available for profiling.")


def synchronize(torch: Any, device_kind: str) -> None:
    if device_kind == "npu" and hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
        torch.npu.synchronize()
    elif device_kind == "cuda" and hasattr(torch, "cuda"):
        torch.cuda.synchronize()


def release_device_memory(torch: Any, device_kind: str) -> None:
    """gc + 清设备显存缓存。调用方须先 del 掉自己持有的 model/大张量引用再调本函数。

    每个 ExecutionMode 节点都会 benchmark/profile/correctness 各加载一次完整模型上
    设备；递归向下时若不释放，显存只增不减，深树必然 OOM。本函数只负责"已无引用后
    把缓存还给设备"，与 synchronize 同层、同 device_kind 风格；解引用由调用方做，
    因为 Python 里 helper 删不掉调用方作用域的变量。
    """
    import gc
    gc.collect()
    if device_kind == "npu" and hasattr(torch, "npu") and hasattr(torch.npu, "empty_cache"):
        torch.npu.empty_cache()
    elif device_kind == "cuda" and hasattr(torch, "cuda") and hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()


def run_forward(model: Any, inputs: dict[str, Any]) -> None:
    if "input_ids" in inputs:
        try:
            model(input_ids=inputs["input_ids"])
        except TypeError:
            model(inputs["input_ids"])
        return
    if "x" in inputs:
        try:
            model(inputs["x"])
        except TypeError:
            model(**inputs)
        return
    model(**inputs)
