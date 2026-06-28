"""Model profile capture, command builders, and profile_report.json readers.

The profiler is self-contained in this project. New reports use
device-neutral timing fields while readers remain compatible with older
``gpu_*`` profile_report.json artifacts.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import inspect
import json
import os
import re
import sys
import sysconfig
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from dataset import load_prompt_dataset, load_tokenizer, tokenize_prompts

# 设备运行时工具已抽到 device_utils（profile/benchmark/correctness 三方共用的公共 API）。
from device_utils import (
    DeviceSpec,
    detect_device,
    device_spec_for,
    import_torch,
    release_device_memory,
    run_forward,
    synchronize,
)


_STDLIB_PROFILE_MODULE: Any | None = None


def _load_stdlib_profile_module() -> Any:
    global _STDLIB_PROFILE_MODULE
    if _STDLIB_PROFILE_MODULE is not None:
        return _STDLIB_PROFILE_MODULE
    stdlib_profile = Path(sysconfig.get_path("stdlib") or "") / "profile.py"
    spec = importlib.util.spec_from_file_location("_stdlib_profile", stdlib_profile)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load stdlib profile module from {stdlib_profile}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _STDLIB_PROFILE_MODULE = module
    return module


def run(statement: str, filename: str | None = None, sort: int | str = -1) -> Any:
    """Compatibility shim for stdlib cProfile."""

    return _load_stdlib_profile_module().run(statement, filename, sort)


def runctx(
    statement: str,
    globals: dict[str, Any],
    locals: dict[str, Any],
    filename: str | None = None,
    sort: int | str = -1,
) -> Any:
    """Compatibility shim for stdlib cProfile."""

    return _load_stdlib_profile_module().runctx(statement, globals, locals, filename, sort)


def __getattr__(name: str) -> Any:
    if name in {"Profile", "_Utils"}:
        return getattr(_load_stdlib_profile_module(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


DEFAULT_WARMUP_ITERS = 5
DEFAULT_PROFILE_ITERS = 10

_KERNEL_CLASSIFICATION: tuple[tuple[tuple[str, ...], str], ...] = (
    (("flashattentionscore", "fusion_attention", "flash_attention", "flash", "fmha", "attention"), "flash_attention"),
    (("aclnnmatmul", "matmul", "batchmatmul", "gemm", "cublas"), "matmul"),
    (("softmax",), "softmax"),
    (("layer_norm", "layernorm"), "layernorm"),
    (("rms_norm", "rmsnorm", "powtensorscalar", "rsqrt"), "rmsnorm"),
    (("swiglu", "silu", "swish", "gelu", "mlp"), "fused_mlp"),
    (("cross_entropy", "nll"), "cross_entropy"),
    (("rotary", "rope", "repeatinterleave"), "rotary_embedding"),
    (("allreduce", "all_reduce", "reduce"), "reduce"),
    (("cast", "_to_copy", "copy", "memcpy"), "copy_cast"),
)

_ANOMALY_REQUIRED_ARTIFACTS = ("kernel_details", "op_summary", "trace_view")


@dataclass(frozen=True)
class ProfileOptimizationSummary:
    top5_pct: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProfileOp:
    """profile_report 中归一化后的单条算子/Kernel 热点记录。

    字段含义：
        rank: 在 profile 热点列表中的排序，通常越小表示越靠前、耗时越高。
        name: profile 报告中的原始算子或 kernel 名称。
        op_type: 归一化后的算子类型；报告缺失时由 ``classify_kernel(name)`` 推断。
        duration_ms: 该算子/kernel 的设备侧总耗时，单位毫秒。
        pct_total: 该算子耗时占总设备耗时或总延迟的百分比。
        cumulative_pct: 按热点排序累计到当前算子的耗时占比。
        call_count: 该算子/kernel 在 profile 窗口内的调用次数。
        avg_time_us: 单次调用的平均耗时，单位微秒。
        shape_info: 报告中记录的输入/输出 shape 或其他形状描述。
        roofline: roofline/性能瓶颈分类，例如 compute、memory 或 unknown。
        optimization_priority: profile 报告给出的优化优先级，例如 HIGH/MEDIUM/LOW。
        source: 该记录的来源；默认是 profile_report，加载文件后通常写入报告路径。
    """

    rank: int
    name: str
    op_type: str
    duration_ms: float
    pct_total: float | None = None
    cumulative_pct: float | None = None
    call_count: int | None = None
    avg_time_us: float | None = None
    shape_info: str = ""
    roofline: str = "unknown"
    optimization_priority: str = "LOW"
    source: str = "profile_report"


@dataclass(frozen=True)
class ProfileReportData:
    """``load_profile_report`` 读取 profile_report.json 后得到的中间证据结构。

    它不是 pipeline 最终用于决策的 ``AnalysisReport``，而是把可选的
    profile_report.json 规范化成轻量 profile 摘要；后续
    ``merge_profile_report`` 会把 ``ops`` 转成 ``TraceOpSummary`` 并合入
    ``ModelRunArtifacts.trace_ops``，再统一生成 ``AnalysisReport``。

    数据结构形状：
        ProfileReportData(
            path=Path(...),                         # profile_report.json 路径
            model="...",                            # 报告里的 model_name/model
            device_name="...",                      # 报告里的 device_name
            total_latency_ms=...,                   # total_device_time_ms/total_latency_ms
            optimization_summary=ProfileOptimizationSummary(...),
            ops=[ProfileOp(...)],                   # top_kernels/kernels/bottleneck_kernels 归一化结果
            notes=["..."],                          # 缺失、解析失败或格式不对时的提示
        )
    """

    path: Path
    model: str
    device_name: str
    total_latency_ms: float | None
    optimization_summary: ProfileOptimizationSummary = field(default_factory=ProfileOptimizationSummary)
    ops: list[ProfileOp] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def trace_ops(self) -> list[dict[str, Any]]:
        return [
            {"name": op.name, "duration_ms": op.duration_ms, "source": op.source}
            for op in self.ops
        ]


@dataclass(frozen=True)
class ProfileCommand:
    argv: list[str]
    cwd: Path
    writes: list[Path]

    def shell_display(self) -> str:
        return " ".join(self.argv)


@dataclass(frozen=True)
class ProfileConfig:
    model: str | None = None
    module: str | None = None
    class_name: str = ""
    pretrained: str | None = None
    input_shape: tuple[int, ...] = (1, 2048)
    dtype: str = "float16"
    device: str = "auto"
    profiler_level: str = "L0"
    analysis_preset: str = "standard"
    warmup_iters: int = DEFAULT_WARMUP_ITERS
    profile_iters: int = DEFAULT_PROFILE_ITERS
    output: Path = Path("workspace/profile_report.json")
    profiler_output_dir: Path = Path("workspace/npu_profiler")
    dataset_path: Path | None = None
    prompt_field: str = "prompt"
    max_samples: int | None = None
    max_input_tokens: int | None = None
    profile_mode: str = "forward"
    max_new_tokens: int = 1
    trust_remote_code: bool = True
    export_trace: bool = False
    record_shapes: bool | None = None
    skip_first: int = 0
    allow_cpu: bool = False


@dataclass(frozen=True)
class KernelRecord:
    name: str
    op_type: str
    device_time_us: float
    call_count: int
    input_shapes: str = ""
    roofline: str = ""


@dataclass(frozen=True)
class ProfileRunResult:
    report: dict[str, Any]
    output_path: Path
    profiler_output_dir: Path | None = None
    trace_path: Path | None = None
    manifest_path: Path | None = None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _profile_kernels(data: dict[str, Any]) -> list[dict[str, Any]]:
    kernels = data.get("top_kernels") or data.get("kernels") or data.get("bottleneck_kernels") or []
    return [kernel for kernel in kernels if isinstance(kernel, dict)]


def _profile_optimization_summary(data: dict[str, Any]) -> ProfileOptimizationSummary:
    raw = data.get("optimization_summary")
    if not isinstance(raw, dict):
        return ProfileOptimizationSummary()
    return ProfileOptimizationSummary(
        top5_pct=_to_float(raw.get("top5_pct")),
        raw=dict(raw),
    )


def load_profile_report(path: str | Path) -> ProfileReportData:
    """Load a profile_report.json without importing torch."""

    report_path = Path(path)
    if not report_path.exists():
        return ProfileReportData(
            path=report_path,
            model="unknown",
            device_name="",
            total_latency_ms=None,
            notes=[f"Profile report missing: {report_path}"],
        )

    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return ProfileReportData(
            path=report_path,
            model="unknown",
            device_name="",
            total_latency_ms=None,
            notes=[f"Failed to parse profile report {report_path}: {exc}"],
        )
    if not isinstance(data, dict):
        return ProfileReportData(
            path=report_path,
            model="unknown",
            device_name="",
            total_latency_ms=None,
            notes=[f"Profile report is not a JSON object: {report_path}"],
        )

    ops: list[ProfileOp] = []
    for idx, item in enumerate(_profile_kernels(data), 1):
        name = str(item.get("name") or item.get("kernel_name") or item.get("op") or "")
        duration = _to_float(item.get("device_time_ms") or item.get("total_device_time_ms"))
        if duration is None:
            duration_us = _to_float(item.get("device_time_us") or item.get("self_device_time_total"))
            duration = duration_us / 1000.0 if duration_us is not None else None
        if not name or duration is None:
            continue
        ops.append(
            ProfileOp(
                rank=int(item.get("rank") or idx),
                name=name,
                op_type=str(item.get("op_type") or classify_kernel(name)),
                duration_ms=duration,
                pct_total=_to_float(item.get("pct_total")),
                cumulative_pct=_to_float(item.get("cumulative_pct")),
                call_count=int(item.get("call_count") or 0) if item.get("call_count") is not None else None,
                avg_time_us=_to_float(item.get("avg_time_us")),
                shape_info=str(item.get("shape_info") or item.get("shape") or ""),
                roofline=str(item.get("roofline") or "unknown"),
                optimization_priority=str(item.get("optimization_priority") or "LOW"),
                source=str(report_path),
            )
        )

    return ProfileReportData(
        path=report_path,
        model=str(data.get("model_name") or data.get("model") or "unknown"),
        device_name=str(data.get("device_name") or ""),
        total_latency_ms=_to_float(data.get("total_device_time_ms") or data.get("total_latency_ms")),
        optimization_summary=_profile_optimization_summary(data),
        ops=ops,
    )


def build_profile_command(
    *,
    repo_dir: Path,
    model: str | None = None,
    class_name: str | None = None,
    pretrained: str | None = None,
    module: str | None = None,
    input_shape: str = "1,256",
    dtype: str = "float16",
    runner: list[str] | None = None,
    profiler_level: str = "L0",
    analysis_preset: str = "standard",
    output: str | Path | None = None,
    dataset_path: str | Path | None = None,
    prompt_field: str = "prompt",
    max_samples: int | None = None,
    max_input_tokens: int | None = None,
    profile_mode: str = "forward",
    max_new_tokens: int | None = None,
) -> ProfileCommand:
    """Build, but do not execute, this module's profile capture command."""

    argv = list(runner or ["python"])
    argv.append("profile_npu.py")
    if pretrained:
        argv.extend(["--module", module or "transformers", "--class-name", class_name or "AutoModelForCausalLM", "--pretrained", pretrained])
    else:
        argv.extend(["--model", model or "models/gpt2.py", "--class-name", class_name or "GPT2"])
    if input_shape:
        argv.extend(["--input-shape", input_shape])
    if dtype:
        argv.extend(["--dtype", dtype])
    if profiler_level:
        argv.extend(["--profiler-level", profiler_level])
    if analysis_preset and analysis_preset != "standard":
        argv.extend(["--analysis-preset", analysis_preset])
    if output is not None:
        argv.extend(["--output", str(output)])
    if dataset_path is not None:
        argv.extend(["--dataset-path", str(dataset_path), "--prompt-field", prompt_field])
    if max_samples is not None:
        argv.extend(["--max-samples", str(max_samples)])
    if max_input_tokens is not None:
        argv.extend(["--max-input-tokens", str(max_input_tokens)])
    if profile_mode != "forward":
        argv.extend(["--profile-mode", profile_mode])
    if max_new_tokens is not None:
        argv.extend(["--max-new-tokens", str(max_new_tokens)])
    writes = [repo_dir / Path(output)] if output is not None and not Path(output).is_absolute() else [Path(output)] if output is not None else [repo_dir / "workspace" / "profile_report.json"]
    return ProfileCommand(argv=argv, cwd=repo_dir, writes=writes)


def apply_analysis_preset(config: ProfileConfig) -> ProfileConfig:
    """Apply profiling presets derived from external Ascend analysis skills."""

    preset = config.analysis_preset.lower().strip()
    if preset == "standard":
        return config
    if preset == "anomaly":
        return replace(
            config,
            profiler_level="L1" if config.profiler_level == "L0" else config.profiler_level,
            record_shapes=True,
            export_trace=True,
        )
    if preset == "deep":
        return replace(
            config,
            profiler_level="L2",
            record_shapes=True,
            export_trace=True,
        )
    raise ValueError(f"Unsupported analysis preset: {config.analysis_preset}")


def classify_kernel(kernel_name: str) -> str:
    """Map framework/device event names to a model-level op type."""

    name_lower = kernel_name.lower()
    for fragments, op_type in _KERNEL_CLASSIFICATION:
        if any(fragment in name_lower for fragment in fragments):
            return op_type
    if "mm" in name_lower and re.search(r"(?:^|[^a-z])mm(?:$|[^a-z])", name_lower):
        return "matmul"
    return "other"


def _priority_label(pct: float) -> str:
    if pct >= 10.0:
        return "HIGH"
    if pct >= 3.0:
        return "MEDIUM"
    return "LOW"


def estimate_roofline_position(op_type: str, device_time_us: float) -> str:
    compute_bound = {"matmul", "flash_attention"}
    memory_bound = {"softmax", "layernorm", "rmsnorm", "reduce", "rotary_embedding", "fused_mlp", "cross_entropy", "copy_cast"}
    if op_type in compute_bound:
        return "compute-bound"
    if op_type in memory_bound:
        return "memory-bound"
    return "likely compute-bound" if device_time_us > 100.0 else "likely memory-bound"


def _shape_string(input_shapes: Any) -> str:
    if not input_shapes:
        return ""
    try:
        return str(input_shapes)
    except Exception:
        return ""


def _relative_or_abs(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def _find_named_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    if not root.exists():
        return []
    found: list[Path] = []
    for pattern in patterns:
        found.extend(path for path in root.rglob(pattern) if path.is_file())
    return sorted(set(found))


def build_profile_artifact_manifest(
    profiler_output_dir: Path,
    *,
    report_path: Path | None = None,
    config: ProfileConfig | None = None,
) -> dict[str, Any]:
    """Inventory artifacts needed by anomaly analysis skills."""

    root = profiler_output_dir
    kernel_details = _find_named_files(root, ("kernel_details*.csv",))
    op_summary = _find_named_files(root, ("op_summary*.csv", "op_statistic*.csv"))
    trace_view = _find_named_files(root, ("trace_view.json", "*trace_view*.json", "trace.json"))
    communication = _find_named_files(root, ("communication.json", "*communication*.json"))
    profiler_info = _find_named_files(root, ("profiler_info*.json",))

    anomaly_artifacts = {
        "kernel_details": [_relative_or_abs(path, root) for path in kernel_details],
        "op_summary": [_relative_or_abs(path, root) for path in op_summary],
        "trace_view": [_relative_or_abs(path, root) for path in trace_view],
        "communication": [_relative_or_abs(path, root) for path in communication],
    }
    readiness = {
        "anomaly_ready": bool(kernel_details and op_summary and trace_view),
        "anomaly_missing": [
            name
            for name, present in (
                ("kernel_details", bool(kernel_details)),
                ("op_summary", bool(op_summary)),
                ("trace_view", bool(trace_view)),
            )
            if not present
        ],
    }
    recommendations: list[str] = []
    if readiness["anomaly_missing"]:
        recommendations.append("For ascend-profiling-anomaly, rerun with --analysis-preset anomaly or --profiler-level L1 --record-shapes.")

    return {
        "schema_version": 1,
        "producer": "profile_npu.py",
        "profiler_output_dir": str(root),
        "profile_report": str(report_path) if report_path is not None else None,
        "capture": {
            "profiler_level": config.profiler_level if config is not None else None,
            "analysis_preset": config.analysis_preset if config is not None else None,
            "record_shapes": config.record_shapes if config is not None else None,
            "export_trace": config.export_trace if config is not None else None,
            "input_shape": list(config.input_shape) if config is not None else None,
            "dtype": config.dtype if config is not None else None,
            "device": config.device if config is not None else None,
            "warmup_iters": config.warmup_iters if config is not None else None,
            "profile_iters": config.profile_iters if config is not None else None,
            "skip_first": config.skip_first if config is not None else None,
            "max_new_tokens": config.max_new_tokens if config is not None else None,
            "dataset_path": str(config.dataset_path) if config is not None and config.dataset_path is not None else None,
            "prompt_field": config.prompt_field if config is not None else None,
            "max_samples": config.max_samples if config is not None else None,
            "max_input_tokens": config.max_input_tokens if config is not None else None,
        },
        "profiler_info": [_relative_or_abs(path, root) for path in profiler_info],
        "anomaly_analysis": {
            "required_by_skill": list(_ANOMALY_REQUIRED_ARTIFACTS),
            "artifacts": anomaly_artifacts,
        },
        "readiness": readiness,
        "recommendations": recommendations,
    }


def _event_device_time_us(evt: Any, device_kind: str) -> float:
    if device_kind == "npu":
        attrs = ("self_device_time_total", "self_privateuse1_time_total", "privateuse1_time_total", "device_time_total")
    elif device_kind == "cuda":
        attrs = ("self_device_time_total", "self_cuda_time_total", "cuda_time_total", "device_time_total")
    else:
        attrs = ("self_cpu_time_total", "cpu_time_total")
    for attr in attrs:
        value = getattr(evt, attr, 0.0)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        if numeric > 0.0:
            return numeric
    return 0.0


def _parse_input_shape(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid input shape '{text}'. Expected comma-separated integers.") from exc
    if not values:
        raise ValueError("input shape must contain at least one dimension")
    return values


def _resolve_dtype(torch: Any, dtype_text: str) -> Any:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    key = dtype_text.lower().strip()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_text}'. Supported: {', '.join(sorted(mapping))}")
    return mapping[key]


def _load_model_from_file(model_path: str, class_name: str) -> Any:
    path = Path(model_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    if not hasattr(module, class_name):
        available = [name for name in dir(module) if not name.startswith("_")]
        raise AttributeError(f"Class '{class_name}' not found in {path}. Available: {available}")
    return getattr(module, class_name)()


def _from_pretrained_kwargs(module_name: str, trust_remote_code: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"torch_dtype": "auto"}
    if module_name == "transformers":
        kwargs["trust_remote_code"] = trust_remote_code
    return kwargs


def _is_local_pretrained_path(pretrained: str | None) -> bool:
    if not pretrained:
        return False
    return Path(pretrained).expanduser().exists()


def _load_model_from_module(module_name: str, class_name: str, pretrained: str | None, *, trust_remote_code: bool = True) -> Any:
    module = importlib.import_module(module_name)
    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in module '{module_name}'.")
    cls = getattr(module, class_name)
    if pretrained:
        if not hasattr(cls, "from_pretrained"):
            raise RuntimeError(f"{class_name} does not provide from_pretrained().")
        kwargs = _from_pretrained_kwargs(module_name, trust_remote_code)
        if module_name == "transformers" and _is_local_pretrained_path(pretrained):
            kwargs["local_files_only"] = True
        return cls.from_pretrained(pretrained, **kwargs)
    return cls()


def load_model(config: ProfileConfig) -> tuple[Any, str]:
    if config.model:
        return _load_model_from_file(config.model, config.class_name), f"{config.class_name} from {config.model}"
    if config.module:
        model = _load_model_from_module(
            config.module,
            config.class_name,
            config.pretrained,
            trust_remote_code=config.trust_remote_code,
        )
        desc = f"{config.class_name} from {config.module}"
        if config.pretrained:
            desc += f" (pretrained: {config.pretrained})"
        return model, desc
    raise ValueError("Must specify either model or module.")


def _is_language_model(model: Any) -> bool:
    cls_name = type(model).__name__.lower()
    lm_markers = ("causal", "lm", "gpt", "llama", "bert", "t5", "opt", "falcon", "mistral", "gemma", "phi", "qwen", "bloom", "mpt")
    if any(marker in cls_name for marker in lm_markers):
        return True
    config = getattr(model, "config", None)
    if config is not None and getattr(config, "vocab_size", None):
        return True
    try:
        signature = inspect.signature(model.forward)
        if "input_ids" in signature.parameters:
            return True
    except (TypeError, ValueError):
        pass
    for name, child in model.named_children():
        if "embed" in name.lower() or "embedding" in type(child).__name__.lower():
            return True
        break
    return False


def _vocab_size(model: Any) -> int:
    config = getattr(model, "config", None)
    if config is not None:
        value = getattr(config, "vocab_size", None)
        if isinstance(value, int) and value > 0:
            return value
    for module in model.modules():
        value = getattr(module, "num_embeddings", None)
        if isinstance(value, int) and value > 0:
            return value
    return 32000


def generate_input(torch: Any, model: Any, input_shape: tuple[int, ...], dtype: Any, device: str) -> dict[str, Any]:
    if _is_language_model(model):
        batch = input_shape[0] if len(input_shape) >= 1 else 1
        seq_len = input_shape[1] if len(input_shape) >= 2 else 512
        return {"input_ids": torch.randint(0, _vocab_size(model), (batch, seq_len), device=device, dtype=torch.long)}
    return {"x": torch.randn(*input_shape, device=device, dtype=dtype)}

# 推理模式路由器
def _run_profile_step(model: Any, inputs: dict[str, Any], config: ProfileConfig) -> None:
    if config.profile_mode == "forward":
        run_forward(model, inputs)
        return
    if config.profile_mode != "generate":
        raise ValueError(f"Unsupported profile_mode: {config.profile_mode}")
    if "input_ids" not in inputs:
        raise RuntimeError("--profile-mode generate requires language-model input_ids.")
    if not hasattr(model, "generate"):
        raise RuntimeError("--profile-mode generate requires a model with generate().")

    generate_inputs = {key: value for key, value in inputs.items() if key in {"input_ids", "attention_mask", "token_type_ids"}}
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max(1, int(config.max_new_tokens)),
        "do_sample": False,
        "use_cache": True,
    }
    model_config = getattr(model, "config", None)
    pad_token_id = getattr(model_config, "pad_token_id", None)
    eos_token_id = getattr(model_config, "eos_token_id", None)
    if pad_token_id is None:
        if isinstance(eos_token_id, (list, tuple)):
            pad_token_id = eos_token_id[0] if eos_token_id else None
        else:
            pad_token_id = eos_token_id
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = pad_token_id
    model.generate(**generate_inputs, **generate_kwargs)


def _dataset_inputs(torch: Any, model: Any, config: ProfileConfig, device: DeviceSpec) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if config.dataset_path is None:
        return None, None
    if not _is_language_model(model):
        raise ValueError("--dataset-path replay currently supports language models with tokenizer-backed input_ids.")
    dataset = load_prompt_dataset(
        config.dataset_path,
        prompt_field=config.prompt_field,
        max_samples=config.max_samples,
    )
    tokenizer = load_tokenizer(config.module or "transformers", config.pretrained, trust_remote_code=config.trust_remote_code)
    if tokenizer is None:
        raise RuntimeError("--dataset-path requires an AutoTokenizer-loadable pretrained model directory.")
    max_length = config.max_input_tokens or (config.input_shape[1] if len(config.input_shape) >= 2 else 512)
    inputs = tokenize_prompts(
        torch,
        tokenizer,
        dataset.prompts,
        device=device.device,
        max_length=max_length,
    )
    manifest = dataset.manifest()
    manifest["max_input_tokens"] = max_length
    manifest["tokenized_keys"] = sorted(inputs)
    if "input_ids" in inputs:
        manifest["input_shape"] = list(inputs["input_ids"].shape)
    return inputs, manifest


def prepare_model_and_input(torch: Any, model: Any, config: ProfileConfig, device: DeviceSpec) -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
    dtype = _resolve_dtype(torch, config.dtype)
    model = model.to(device=device.device)
    if dtype in (torch.float16, torch.bfloat16):
        model = model.to(dtype=dtype)
    model.eval()

    dataset_inputs, dataset_manifest = _dataset_inputs(torch, model, config, device)
    if dataset_inputs is not None:
        with torch.no_grad():
            _run_profile_step(model, dataset_inputs, config)
        synchronize(torch, device.kind)
        return model, dataset_inputs, dataset_manifest

    batch = config.input_shape[0] if config.input_shape else 1
    for attempt_batch in (batch, max(1, batch // 2), 1):
        current_shape = (attempt_batch, *config.input_shape[1:])
        inputs = generate_input(torch, model, current_shape, dtype, device.device)
        try:
            with torch.no_grad():
                _run_profile_step(model, inputs, config)
            synchronize(torch, device.kind)
            return model, inputs, None
        except Exception:
            if device.kind == "cuda" and hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
            if attempt_batch == 1:
                raise
    raise RuntimeError("Could not run a forward pass for profiling.")


def _build_npu_profile_kwargs(torch_npu: Any, config: ProfileConfig) -> dict[str, Any]:
    level = config.profiler_level.upper()
    activities = [torch_npu.profiler.ProfilerActivity.NPU]
    if level in {"L1", "L2"}:
        activities.insert(0, torch_npu.profiler.ProfilerActivity.CPU)
    # 兼容不同版本的 torch_npu
    try:
        schedule = torch_npu.profiler.schedule(wait=0, 
                                               warmup=0, 
                                               active=config.profile_iters, 
                                               repeat=1, 
                                               skip_first=config.skip_first)
    except TypeError:
        schedule = torch_npu.profiler.schedule(wait=0, 
                                               warmup=0, 
                                               active=config.profile_iters, 
                                               repeat=1)

    kwargs: dict[str, Any] = {
        "activities": activities,
        "schedule": schedule,
        "on_trace_ready": torch_npu.profiler.tensorboard_trace_handler(str(config.profiler_output_dir)),
    }

    if level in {"L1", "L2"}:
        kwargs["record_shapes"] = config.record_shapes if config.record_shapes is not None else True
        kwargs["with_stack"] = level == "L2"
        kwargs["profile_memory"] = level == "L2"
        if hasattr(torch_npu.profiler, "_ExperimentalConfig") and hasattr(torch_npu.profiler, "ProfilerLevel"):
            kwargs["experimental_config"] = torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1
            )
    else:
        if config.record_shapes is not None:
            kwargs["record_shapes"] = config.record_shapes
    return kwargs


def _build_torch_profile_kwargs(torch: Any, config: ProfileConfig, device: DeviceSpec) -> dict[str, Any]:
    level = config.profiler_level.upper()
    if device.kind == "cuda":
        activities = [torch.profiler.ProfilerActivity.CUDA]
        if level in {"L1", "L2"}:
            activities.insert(0, torch.profiler.ProfilerActivity.CPU)
    else:
        activities = [torch.profiler.ProfilerActivity.CPU]

    kwargs: dict[str, Any] = {
        "activities": activities,
        "record_shapes": config.record_shapes if config.record_shapes is not None else level in {"L1", "L2"},
        "with_stack": level == "L2",
        "profile_memory": level == "L2",
    }
    if config.export_trace:
        kwargs["on_trace_ready"] = torch.profiler.tensorboard_trace_handler(str(config.profiler_output_dir))
    return kwargs


def _warmup_phase(
    model: Any,
    inputs: dict[str, Any],
    config: ProfileConfig,
    torch: Any,
    device: DeviceSpec,
) -> None:
    """Warmup to eliminate cold-start overhead.

    Runs the model several times without recording any metrics to ensure:
    - Memory is allocated
    - Kernels are compiled/selected
    - Device drivers are initialized
    """
    with torch.no_grad():
        for _ in range(config.warmup_iters):
            _run_profile_step(model, inputs, config)
            synchronize(torch, device.kind)
    # Final sync to ensure warmup is completely done
    synchronize(torch, device.kind)


def _export_trace(prof: Any, config: ProfileConfig) -> Path | None:
    """Export Chrome trace if supported and configured.

    Returns:
        Path to trace file if successful, None otherwise
    """
    if not config.export_trace or not hasattr(prof, "export_chrome_trace"):
        return None

    trace_path = config.profiler_output_dir / "trace.json"
    try:
        prof.export_chrome_trace(str(trace_path))
        return trace_path
    except Exception:
        return None


def _extract_kernel_records(
    prof: Any,
    device: DeviceSpec,
    config: ProfileConfig,
) -> list[KernelRecord]:
    """Extract kernel records from profiler or fallback to CSV files.

    Tries multiple sources in order:
    1. profiler.key_averages() API
    2. NPU kernel_details.csv (NPU only)
    3. NPU op_statistic.csv (NPU only)

    Returns:
        List of KernelRecord sorted by device time (descending)
    """
    records: list[KernelRecord] = []

    # Try profiler API first
    if hasattr(prof, "key_averages"):
        for evt in prof.key_averages(group_by_input_shape=True):
            device_time_us = _event_device_time_us(evt, device.kind)
            if device_time_us <= 0.0:
                continue

            name = str(getattr(evt, "key", "") or "")
            if not name:
                continue

            records.append(
                KernelRecord(
                    name=name,
                    op_type=classify_kernel(name),
                    device_time_us=device_time_us,
                    call_count=int(getattr(evt, "count", 1) or 1),
                    input_shapes=_shape_string(getattr(evt, "input_shapes", "")),
                )
            )

    # NPU fallback: try CSV files if profiler API returned nothing
    if not records and device.kind == "npu":
        records = _records_from_npu_kernel_details(config.profiler_output_dir)
    if not records and device.kind == "npu":
        records = _records_from_npu_op_statistic(config.profiler_output_dir)

    records.sort(key=lambda r: r.device_time_us, reverse=True)
    return records


def _profiling_phase(
    model: Any,
    inputs: dict[str, Any],
    config: ProfileConfig,
    torch: Any,
    device: DeviceSpec,
    torch_npu: Any | None,
) -> tuple[list[KernelRecord], dict[str, Any]]:
    """Profiler collection for operator-level performance data and latency estimation.

    Runs the profiler to collect kernel-level metrics using continuous submission
    (no per-iteration sync) to better reflect async execution.

    Latency is estimated from profiler's total device time, not from manual timing,
    to avoid disrupting the async pipeline with frequent synchronization.

    Returns:
        Tuple of (kernel_records, profiler_artifacts)
    """
    # Select profiler based on device type
    if device.kind == "npu":
        if torch_npu is None:
            raise RuntimeError("torch_npu is required for NPU profiling.")
        profile_factory = torch_npu.profiler.profile
        profile_kwargs = _build_npu_profile_kwargs(torch_npu, config)
    else:
        profile_factory = torch.profiler.profile
        profile_kwargs = _build_torch_profile_kwargs(torch, config, device)

    # Run profiler
    with torch.no_grad():
        with profile_factory(**profile_kwargs) as prof:
            # NPU may need extra skip_first steps
            total_steps = config.profile_iters + max(config.skip_first, 0) \
                          if device.kind == "npu" else config.profile_iters

            for _ in range(total_steps):
                _run_profile_step(model, inputs, config)
                if hasattr(prof, "step"):
                    prof.step()

            # Single sync at the end to ensure all operations complete
            synchronize(torch, device.kind)

    # Export trace if configured
    trace_path = _export_trace(prof, config)

    # Extract kernel records
    records = _extract_kernel_records(prof, device, config)

    # Build artifacts dict
    artifacts: dict[str, Any] = {}
    if trace_path is not None:
        artifacts["trace_path"] = str(trace_path)

    return records, artifacts


def profile_model(
    model: Any,
    inputs: dict[str, Any],
    *,
    config: ProfileConfig,
    device: DeviceSpec,
    torch_module: Any | None = None,
    torch_npu: Any | None = None,
) -> tuple[list[KernelRecord], dict[str, Any]]:
    """Run profiling to collect operator-level performance data.

    This function focuses solely on identifying performance bottlenecks at the
    operator level. For accurate end-to-end latency measurement, use benchmark.py
    which runs without profiler overhead.

    Workflow:
    1. Warmup - eliminate cold-start overhead (not recorded)
    2. Profiler collection - operator-level performance data

    Returns:
        Tuple of (kernel_records, artifacts_dict)
        - kernel_records: sorted by device time (slowest first)
        - artifacts_dict: profiler output paths and metadata
    """
    torch = torch_module or import_torch()
    config.profiler_output_dir.mkdir(parents=True, exist_ok=True)

    # Warmup phase
    _warmup_phase(model, inputs, config, torch, device)

    # Profiler collection phase
    records, profiler_artifacts = _profiling_phase(
        model, inputs, config, torch, device, torch_npu
    )

    # Build final artifacts
    artifacts: dict[str, Any] = {
        "profiler_output_dir": str(config.profiler_output_dir),
        **profiler_artifacts,
    }

    return records, artifacts


def _records_from_npu_kernel_details(profiler_output_dir: Path) -> list[KernelRecord]:
    totals: dict[str, tuple[float, int]] = {}
    for path in sorted(profiler_output_dir.glob("**/kernel_details.csv")):
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    name = str(row.get("Name") or "").strip()
                    duration_text = str(row.get("Duration(us)") or "").strip()
                    if not name or not duration_text:
                        continue
                    try:
                        duration_us = float(duration_text)
                    except ValueError:
                        continue
                    total, count = totals.get(name, (0.0, 0))
                    totals[name] = (total + duration_us, count + 1)
        except OSError:
            continue
    records = [
        KernelRecord(
            name=name,
            op_type=classify_kernel(name),
            device_time_us=duration_us,
            call_count=count,
        )
        for name, (duration_us, count) in totals.items()
        if duration_us > 0.0
    ]
    records.sort(key=lambda item: item.device_time_us, reverse=True)
    return records


def _records_from_npu_op_statistic(profiler_output_dir: Path) -> list[KernelRecord]:
    """Fallback for NPU profiler exports that only include op_statistic.csv."""

    totals: dict[str, tuple[float, int]] = {}
    for path in sorted(profiler_output_dir.glob("**/op_statistic.csv")):
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    name = str(row.get("OP Type") or row.get("Name") or "").strip()
                    duration_text = str(row.get("Total Time(us)") or row.get("Duration(us)") or "").strip()
                    if not name or not duration_text:
                        continue
                    try:
                        duration_us = float(duration_text)
                    except ValueError:
                        continue
                    count = _csv_count(row.get("Count"))
                    total, calls = totals.get(name, (0.0, 0))
                    totals[name] = (total + duration_us, calls + count)
        except OSError:
            continue
    records = [
        KernelRecord(
            name=name,
            op_type=classify_kernel(name),
            device_time_us=duration_us,
            call_count=max(count, 1),
        )
        for name, (duration_us, count) in totals.items()
        if duration_us > 0.0
    ]
    records.sort(key=lambda item: item.device_time_us, reverse=True)
    return records


def _csv_count(value: Any) -> int:
    count = _to_float(value)
    if count is None:
        return 1
    return max(int(count), 1)


def build_profile_report(
    records: list[KernelRecord],
    device: DeviceSpec,
    config: ProfileConfig,
    model_desc: str,
    *,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_time_us = sum(record.device_time_us for record in records)
    total_time_ms = total_time_us / 1000.0
    annotated: list[KernelRecord] = []
    for record in records:
        annotated.append(
            KernelRecord(
                name=record.name,
                op_type=record.op_type,
                device_time_us=record.device_time_us,
                call_count=record.call_count,
                input_shapes=record.input_shapes,
                roofline=estimate_roofline_position(record.op_type, record.device_time_us),
            )
        )

    top_kernels: list[dict[str, Any]] = []
    cumulative_pct = 0.0
    for idx, record in enumerate(annotated, 1):
        pct = (record.device_time_us / total_time_us * 100.0) if total_time_us > 0.0 else 0.0
        cumulative_pct += pct
        top_kernels.append(
            {
                "rank": idx,
                "name": record.name,
                "op_type": record.op_type,
                "shape_info": record.input_shapes,
                "device_time_ms": round(record.device_time_us / 1000.0, 3),
                "device_time_us": round(record.device_time_us, 3),
                "call_count": record.call_count,
                "avg_time_us": round(record.device_time_us / max(record.call_count, 1), 2),
                "pct_total": round(pct, 1),
                "cumulative_pct": round(cumulative_pct, 1),
                "roofline": record.roofline,
                "optimization_priority": _priority_label(pct),
            }
        )

    top5_time_us = sum(record.device_time_us for record in annotated[:5])
    top5_pct = (top5_time_us / total_time_us * 100.0) if total_time_us > 0.0 else 0.0

    report: dict[str, Any] = {
        "schema_version": 4,
        "producer": "profile_npu.py",
        "model": config.model or config.module,
        "model_desc": model_desc,
        "class_name": config.class_name,
        "pretrained": config.pretrained,
        "input_shape": list(config.input_shape),
        "dtype": config.dtype,
        "max_new_tokens": config.max_new_tokens,
        "device_kind": device.kind,
        "device_name": device.name,
        "device_peak_tflops_fp16": device.peak_tflops_fp16,
        "device_peak_bandwidth_gb_s": device.peak_bandwidth_gb_s,
        "profiler_level": config.profiler_level.upper(),
        "analysis_preset": config.analysis_preset,
        "record_shapes": config.record_shapes,
        "export_trace": config.export_trace,
        "profiler_output_dir": str(config.profiler_output_dir),
        "total_device_time_ms": round(total_time_ms, 3),
        "total_kernels": len(annotated),
        "profile_iters": config.profile_iters,
        "top_kernels": top_kernels,
        "optimization_summary": {
            "top5_pct": round(top5_pct, 1),
        },
    }
    if artifacts:
        report["artifacts"] = dict(artifacts)

    return report


def print_report(report: dict[str, Any]) -> None:
    print()
    print("=" * 72)
    print("  AscendFast Profiler")
    print("=" * 72)
    print(f"  Model:  {report.get('model_desc') or report.get('model')}")
    print(f"  Input:  shape={report.get('input_shape')}, dtype={report.get('dtype')}")
    print(f"  Device: {report.get('device_name')} ({report.get('device_kind')})")
    print(f"  Level:  {report.get('profiler_level')}")
    print(f"  Total device time: {report.get('total_device_time_ms'):.3f} ms")
    print()
    print(f"{'Rank':>4} | {'Op Type':<18} | {'Time ms':>9} | {'Calls':>6} | {'Pct':>6} | Name")
    print("-" * 72)
    for item in report.get("top_kernels", [])[:20]:
        print(
            f"{item['rank']:>4} | {item['op_type']:<18} | {item['device_time_ms']:>9.3f} | "
            f"{item['call_count']:>6} | {item['pct_total']:>5.1f}% | {item['name']}"
        )
    summary = report.get("optimization_summary", {})
    print()
    print(f"  Top-5 time:     {summary.get('top5_pct', 0.0):.1f}%")


# 加载模型、执行 NPU profiling、汇总 kernel/latency 数据
def generate_profile_report(config: ProfileConfig) -> ProfileRunResult:
    config = apply_analysis_preset(config)
    device, torch_npu = detect_device(config.device, allow_cpu=config.allow_cpu)
    torch = import_torch()
    model, model_desc = load_model(config)
    model, inputs, dataset_manifest = prepare_model_and_input(torch, model, config, device)
    records, artifacts = profile_model(
        model,
        inputs,
        config=config,
        device=device,
        torch_module=torch,
        torch_npu=torch_npu,
    )
    if dataset_manifest is not None:
        artifacts["dataset"] = dataset_manifest
    if not records:
        raise RuntimeError("No device events were captured. Check device placement, input shape, and profiler level.")
    report = build_profile_report(records, device, config, model_desc, artifacts=artifacts)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_profile_artifact_manifest(
        config.profiler_output_dir,
        report_path=config.output,
        config=config,
    )
    manifest_path = config.profiler_output_dir / "profile_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report.setdefault("artifacts", {})
    report["artifacts"]["profile_manifest"] = str(manifest_path)
    report["analysis_readiness"] = manifest["readiness"]
    config.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return ProfileRunResult(
        report=report,
        output_path=config.output,
        profiler_output_dir=Path(artifacts["profiler_output_dir"]) if "profiler_output_dir" in artifacts else None,
        trace_path=Path(artifacts["trace_path"]) if "trace_path" in artifacts else None,
        manifest_path=manifest_path,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture model-level PyTorch/NPU profile data.")
    parser.add_argument("--model", help="Path to a Python file containing the model class.")
    parser.add_argument("--module", help="Python module containing the model class, e.g. transformers.")
    parser.add_argument("--class-name", required=True, help="Model class name.")
    parser.add_argument("--pretrained", help="Pretrained model name/path for classes with from_pretrained().")
    parser.add_argument("--input-shape", required=True, help="Comma-separated input shape, e.g. 1,2048.")
    parser.add_argument("--dtype", default="float16", help="float16, bfloat16, or float32.")
    parser.add_argument("--device", default="auto", choices=["auto", "npu", "cuda", "cpu"], help="Profiling device preference.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU-only profiling fallback.")
    parser.add_argument("--profiler-level", default="L0", choices=["L0", "L1", "L2"], help="NPU profiler level. L0 is the default lightweight capture.")
    parser.add_argument(
        "--analysis-preset",
        default="standard",
        choices=["standard", "anomaly", "deep"],
        help="Capture preset for downstream external profiling skills. anomaly uses L1+shapes+trace; deep uses L2.",
    )
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERS)
    parser.add_argument("--profile-iters", type=int, default=DEFAULT_PROFILE_ITERS)
    parser.add_argument("--skip-first", type=int, default=0, help="Profiler schedule skip_first steps.")
    parser.add_argument("--record-shapes", action="store_true", help="Record input shapes even for L0.")
    parser.add_argument("--export-trace", action="store_true", help="Export TensorBoard/Chrome trace artifacts when supported.")
    parser.add_argument("--profiler-output-dir", type=Path, default=Path("workspace/npu_profiler"))
    parser.add_argument("--output", type=Path, default=Path("workspace/profile_report.json"))
    parser.add_argument("--dataset-path", type=Path, help="Local JSONL prompt dataset for real input replay.")
    parser.add_argument("--prompt-field", default="prompt", help="JSONL field containing the prompt text.")
    parser.add_argument("--max-samples", type=int, help="Maximum prompt rows to replay.")
    parser.add_argument("--max-input-tokens", type=int, help="Tokenizer truncation length for prompt replay.")
    parser.add_argument("--profile-mode", choices=["forward", "generate"], default="forward", help="Profile a raw forward pass or autoregressive generate/decode path.")
    parser.add_argument("--max-new-tokens", type=int, default=1, help="New tokens to generate when --profile-mode generate is used.")
    parser.add_argument("--no-trust-remote-code", action="store_true", help="Disable trust_remote_code for Transformers local/modelscope model loading.")
    args = parser.parse_args(argv)
    if not args.model and not args.module:
        parser.error("Must specify either --model or --module.")
    return args


def config_from_args(args: argparse.Namespace) -> ProfileConfig:
    return ProfileConfig(
        model=args.model,
        module=args.module,
        class_name=args.class_name,
        pretrained=args.pretrained,
        input_shape=_parse_input_shape(args.input_shape),
        dtype=args.dtype,
        device=args.device,
        profiler_level=args.profiler_level,
        analysis_preset=args.analysis_preset,
        warmup_iters=args.warmup_iters,
        profile_iters=args.profile_iters,
        output=args.output,
        profiler_output_dir=args.profiler_output_dir,
        dataset_path=args.dataset_path,
        prompt_field=args.prompt_field,
        max_samples=args.max_samples,
        max_input_tokens=args.max_input_tokens,
        profile_mode=getattr(args, "profile_mode", "forward"),
        max_new_tokens=getattr(args, "max_new_tokens", 1),
        trust_remote_code=not getattr(args, "no_trust_remote_code", False),
        export_trace=args.export_trace,
        record_shapes=True if args.record_shapes else None,
        skip_first=args.skip_first,
        allow_cpu=args.allow_cpu,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        result = generate_profile_report(config_from_args(_parse_args(argv)))
    except Exception as exc:  # pragma: no cover - exercised on real hosts
        print(f"ERROR: {exc}")
        traceback.print_exc()
        return 1
    print_report(result.report)
    print()
    print(f"Profile saved to {result.output_path}")
    if result.profiler_output_dir is not None:
        print(f"Profiler artifacts: {result.profiler_output_dir}")
    if result.trace_path is not None:
        print(f"Chrome trace: {result.trace_path}")
    if result.manifest_path is not None:
        print(f"Artifact manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
