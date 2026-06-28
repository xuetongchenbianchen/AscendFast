"""run_profile：对一个 ExecutionMode 在真实 NPU 上做诊断性 profile。

确定性实现：in-process 编排 profile_npu 的中层积木，build_model() 为唯一真相源。
profile_mode 由 _resolve_profile_mode() 决定——当前一律 forward，generate 作为
扩展分支（底层 _run_profile_step 已支持），不再走 LLM agent。

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

from dataset import load_prompt_dataset, tokenize_prompts
from models import ExecutionMode, ProfileResult
from workspace_loader import load_build_model
from profile_npu import (
    ProfileConfig,
    build_profile_report,
    profile_model,
)
from device_utils import device_spec_for, import_torch, release_device_memory

_PROJECT_ROOT = Path(__file__).parent
# profile 用的"模拟数据"：与真实领域 benchmark 数据集刻意区分开。
_PROFILE_DATASET = _PROJECT_ROOT / "data" / "prompts_real.jsonl"


def run_profile(mode: ExecutionMode) -> ProfileResult:
    """对 mode 做 profile，返回 ProfileResult。

    走确定性实现，profile_mode 由 _resolve_profile_mode() 决定
    （默认 forward，generate 作为扩展分支）。
    """
    if mode.correctness_passed is not True:
        raise ValueError(
            f"run_profile requires correctness_passed=True, got {mode.correctness_passed}"
        )
    return _deterministic_profile(mode, profile_mode=_resolve_profile_mode(mode))


def _resolve_profile_mode(mode: ExecutionMode) -> str:
    """选择本次 profile 用 forward 还是 generate。

    从简单开始：当前一律返回 "forward"。

    扩展点（generate）：将来按 mode.change_log 里是否包含 kvcache / decode /
    generation 类优化返回 "generate"，否则 "forward"。底层 _run_profile_step()
    已实现 generate 分支，这里只负责"何时启用"。
    """
    return "forward"


# --------------------------------------------------------------------------- #
# 确定性版：in-process 编排 profile_npu 的中层积木，build_model() 为唯一真相源。
# --------------------------------------------------------------------------- #
def _deterministic_profile(
    mode: ExecutionMode,
    *,
    profile_mode: str = "forward",
    input_shape: tuple[int, ...] = (1, 512),
) -> ProfileResult:
    torch = import_torch()
    model, tokenizer = load_build_model(mode)
    device, torch_npu = device_spec_for(model)       # 模型在哪就用哪，不二次搬运
    model = model.eval()

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
        release_device_memory(torch, device.kind)

    return ProfileResult(
        uid=f"profile:{mode.uid}",
        execution_mode_uid=mode.uid,
        profile_report=report,
        profile_report_path=str(cfg.output),
        profiler_output_dir=str(cfg.profiler_output_dir),
        extra={"source": "deterministic", "profile_mode": profile_mode},
    )
