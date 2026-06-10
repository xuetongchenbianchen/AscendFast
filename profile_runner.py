"""run_profile：对一个 ExecutionMode 在真实 NPU 上做诊断性 profile。

与 analysis.py / strategy.py 同范式：Agent 优先 + 确定性 fallback。

定位（与 run_real_benchmark 区分）：
- run_profile      —— 用 data/ 模拟数据做**诊断**，产出 profile_report.json，
                       喂给 analyze_profile → 生成下一轮策略。
- run_real_benchmark —— 用真实领域数据集测**目标延迟**（2x 标尺），见 optimization.py。

唯一真相源：模型与 tokenizer 都来自 workspace 的 build_model()，profile 不再
从原始权重目录重载 tokenizer，从而和 apply / benchmark 保持一致。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from agent_client import AGENT_ENABLED, call_agent_json
from dataset import load_prompt_dataset, tokenize_prompts
from models import ExecutionMode, ProfileResult
from workspace_loader import load_build_model
from profile_npu import (
    ProfileConfig,
    build_profile_report,
    detect_device,
    profile_model,
    _import_torch,
    _release_device_memory,
)

_PROJECT_ROOT = Path(__file__).parent
# profile 用的"模拟数据"：与真实领域 benchmark 数据集刻意区分开。
_PROFILE_DATASET = _PROJECT_ROOT / "data" / "prompts_real.jsonl"


def run_profile(mode: ExecutionMode) -> ProfileResult:
    """对 mode 做 profile，返回 ProfileResult。Agent 优先，失败回退确定性实现。"""
    if mode.correctness_passed is not True:
        raise ValueError(
            f"run_profile requires correctness_passed=True, got {mode.correctness_passed}"
        )
    if AGENT_ENABLED:
        result = _llm_run_profile(mode)
        if result is not None:
            return result
    return _deterministic_profile(mode)


# --------------------------------------------------------------------------- #
# Agent 版：Agent 读 change_log 自行决定 profile_mode / input_shape / 重试，
# 但模型与 tokenizer 仍由 workspace 的 build_model() 锁定。
# --------------------------------------------------------------------------- #
def _llm_run_profile(mode: ExecutionMode) -> ProfileResult | None:
    change_log = _format_change_log(mode)
    prompt = (
        "Profile this optimized model variant on Ascend NPU and report where the\n"
        "profile_report.json landed plus the measured latency.\n\n"
        f"Workspace directory (absolute): {mode.workspace_dir}\n"
        f"Unified entrypoint: build_model.py :: build_model() -> (model, tokenizer)\n"
        f"Profile dataset (simulated, for diagnosis): {_PROFILE_DATASET}\n\n"
        "## Optimizations already applied (pick the right profile mode)\n"
        f"{change_log}\n\n"
        "Load the model ONLY through build_model() (its optimization lives there).\n"
        "Choose --profile-mode generate for kvcache/decode/generation work, else\n"
        "forward. Write the report under <workspace>/profile/profile_report.json.\n\n"
        "Return ONLY this JSON:\n"
        '{"profile_report_path": "<abs path>",\n'
        ' "profiler_output_dir": "<abs path or null>",\n'
        ' "latency_after_ms": <mean ms from latency_stats_ms.mean>,\n'
        ' "profile_mode": "forward|generate",\n'
        ' "notes": "<one line>"}'
    )
    result = call_agent_json("profile-agent", prompt, timeout=1000)
    if not isinstance(result, dict) or "profile_report_path" not in result:
        return None
    report_path = Path(str(result["profile_report_path"]))
    report = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = None
    latency = _to_float(result.get("latency_after_ms"))
    if latency <= 0.0 and isinstance(report, dict):
        latency = _to_float((report.get("latency_stats_ms") or {}).get("mean"))
    return ProfileResult(
        uid=f"profile:{mode.uid}",
        execution_mode_uid=mode.uid,
        latency_before=0.0,
        latency_after=latency,
        profile_report=report,
        profile_report_path=str(report_path),
        profiler_output_dir=_opt_str(result.get("profiler_output_dir")),
        extra={"source": "profile_agent", "profile_mode": result.get("profile_mode"),
               "notes": result.get("notes")},
    )


# --------------------------------------------------------------------------- #
# 确定性版：in-process 编排 profile_npu 的中层积木，build_model() 为唯一真相源。
# --------------------------------------------------------------------------- #
def _deterministic_profile(
    mode: ExecutionMode,
    *,
    profile_mode: str = "forward",
    input_shape: tuple[int, ...] = (1, 512),
) -> ProfileResult:
    torch = _import_torch()
    model, tokenizer = load_build_model(mode)
    device, torch_npu = detect_device("auto", allow_cpu=True)
    model = model.to(device.device).eval()

    out_dir = Path(mode.workspace_dir) / "profile"
    cfg = ProfileConfig(
        module="build_model",
        class_name="build_model",
        pretrained=str(Path(mode.workspace_dir) / "model"),
        input_shape=input_shape,
        profile_mode=profile_mode,
        dataset_path=_PROFILE_DATASET,
        output=out_dir / "profile_report.json",
        profiler_output_dir=out_dir / "npu_profiler",
        allow_cpu=True,
    )

    max_len = input_shape[1] if len(input_shape) >= 2 else 512
    prompts = load_prompt_dataset(_PROFILE_DATASET).prompts
    inputs = tokenize_prompts(torch, tokenizer, prompts, device=device.device, max_length=max_len)

    try:
        records, artifacts = profile_model(
            model, inputs, config=cfg, device=device, torch_module=torch, torch_npu=torch_npu,
        )
        if not records:
            raise RuntimeError("No device events captured during profile.")
        report = build_profile_report(records, device, cfg, "build_model", artifacts=artifacts)
        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        cfg.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        # 本节点 profile 完即释放模型，防止递归深处显存累积 OOM。
        del model
        _release_device_memory(torch, device.kind)

    latency = _to_float((report.get("latency_stats_ms") or {}).get("mean"))
    return ProfileResult(
        uid=f"profile:{mode.uid}",
        execution_mode_uid=mode.uid,
        latency_before=0.0,
        latency_after=latency,
        profile_report=report,
        profile_report_path=str(cfg.output),
        profiler_output_dir=str(cfg.profiler_output_dir),
        extra={"source": "deterministic", "profile_mode": profile_mode},
    )


def _format_change_log(mode: ExecutionMode) -> str:
    if not mode.change_log:
        return "(none — baseline model)"
    return "\n".join(f"{i}. [{r.kind}] {r.summary}" for i, r in enumerate(mode.change_log, 1))


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text and text.lower() != "null" else None
