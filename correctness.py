"""正确性检验：判断一个优化后的 ExecutionMode 输出是否仍与 baseline 等价。

每个 fork 都与 baseline 金标准比（非父 mode），指标为 last-token
logits 余弦相似度，golden 缓存到 baseline workspace。通过统一入口加载。

定位（与 benchmark.py 对称）：
- benchmark.py    —— 测一个 mode 跑得**多快**（forward 延迟）。
- correctness.py  —— 判一个 mode 算得**对不对**（输出与 baseline 是否一致）。

两者都是"对一个已物化的 mode 做 NPU 评测"，都通过 workspace_loader.load_build_model
取模型，互不依赖 profiler，也不依赖 apply（apply 只负责"造"出 mode，本模块负责"评"）。

判定指标：**last-token logits 的余弦相似度**。
- 对每条固定 prompt，取最后一个有效 token 位置的 next-token logits 分布；
- 与 baseline 在同一 prompt、同一位置的 logits 算余弦相似度，逐样本取均值；
- 均值 ≥ threshold → correctness_passed=True。
选 last-token 而非全序列：最贴近"模型下一步输出什么"的生成行为，对量化/算子融合
带来的合法数值漂移容忍恰当，且天然避开 padding 位置的噪声。

baseline 的金标准 logits 算一次即缓存到 baseline workspace（correctness_golden.pt），
之后每个 fork 都跟这份金标准比，而非逐层与父 mode 比——避免误差沿优化链累积。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dataset import load_prompt_dataset, tokenize_prompts
from models import ExecutionMode
from mode_store import write_manifest
from workspace_loader import load_build_model
from device_utils import device_spec_for, import_torch, release_device_memory, synchronize

_PROJECT_ROOT = Path(__file__).parent
# 正确性检验用的固定 prompt 集：与 benchmark 同源真实数据，口径一致。
_CORRECTNESS_DATASET = _PROJECT_ROOT / "data" / "prompts_sharegpt.jsonl"
_GOLDEN_NAME = "correctness_golden.pt"

_DEFAULT_THRESHOLD = 0.85
_DEFAULT_SAMPLES = 32
_DEFAULT_MAX_TOKENS = 512


def run_correctness_test(
    mode: ExecutionMode,
    baseline_mode: ExecutionMode | None = None,
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    max_samples: int = _DEFAULT_SAMPLES,
    max_input_tokens: int = _DEFAULT_MAX_TOKENS,
    dataset_path: str | None = None,
) -> ExecutionMode:
    """检验 mode 与 baseline 输出是否等价，填写 correctness_passed 并写回 manifest。

    Args:
        mode:             待检验的执行模式（apply 产出的、correctness_passed=None）。
        baseline_mode:    金标准来源；None 时要求 baseline golden 已缓存。
        threshold:        last-token logits 余弦相似度阈值，≥ 即判通过。
        max_samples:      固定 prompt 条数（与 benchmark 同源数据集前 N 条）。
        max_input_tokens: tokenize 的 max_length / padding 上限。
        dataset_path:     prompt 集 jsonl；None 用默认 ShareGPT 转出文件。

    Returns:
        同一个 mode，已填 correctness_passed（并落 manifest）。
    """
    # baseline 自身按定义正确，直接放行（且其 golden 由首个 fork 触发缓存）。
    if mode.parent_uid is None:
        mode.correctness_passed = True
        write_manifest(mode)
        return mode

    ds_path = Path(dataset_path) if dataset_path else _CORRECTNESS_DATASET
    if not ds_path.exists():
        raise FileNotFoundError(
            f"correctness 数据集不存在: {ds_path}\n"
            "先用 data/sharegpt_to_jsonl.py 从 ShareGPT 生成，或显式传 dataset_path。"
        )

    torch = import_torch()
    golden = _ensure_golden(
        torch, baseline_mode, ds_path,
        max_samples=max_samples, max_input_tokens=max_input_tokens,
    )

    logits = _last_token_logits(
        torch, mode, ds_path,
        max_samples=max_samples, max_input_tokens=max_input_tokens,
    )
    score = _mean_cosine(torch, golden, logits)

    mode.correctness_passed = bool(score >= threshold)
    extra = dict(mode.extra or {})
    extra["correctness"] = {"metric": "last_token_logits_cosine",
                            "score": round(float(score), 6), "threshold": threshold}
    mode.extra = extra
    write_manifest(mode)
    return mode


# --------------------------------------------------------------------------- #
# golden 缓存：baseline 的 last-token logits 算一次，存到 baseline workspace
# --------------------------------------------------------------------------- #
def _ensure_golden(
    torch: Any,
    baseline_mode: ExecutionMode | None,
    ds_path: Path,
    *,
    max_samples: int,
    max_input_tokens: int,
) -> Any:
    """返回 baseline 的金标准 last-token logits（[N, vocab]）；无缓存则算一次落盘。

    缓存键还包含数据集/采样配置，配置变了会重算，避免拿旧 golden 比新口径。
    """
    if baseline_mode is None:
        raise ValueError(
            "run_correctness_test needs baseline_mode the first time (to build golden)."
        )
    golden_path = Path(baseline_mode.workspace_dir) / _GOLDEN_NAME
    meta_path = golden_path.with_suffix(".json")
    want = {"dataset": str(ds_path), "max_samples": max_samples,
            "max_input_tokens": max_input_tokens}

    if golden_path.exists() and meta_path.exists():
        try:
            if json.loads(meta_path.read_text(encoding="utf-8")) == want:
                return torch.load(golden_path)
        except Exception:
            pass  # 缓存损坏/不匹配 → 重算

    golden = _last_token_logits(
        torch, baseline_mode, ds_path,
        max_samples=max_samples, max_input_tokens=max_input_tokens,
    )
    try:
        torch.save(golden, golden_path)
        meta_path.write_text(json.dumps(want, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass  # 落盘失败不致命，本轮直接用内存里的 golden
    return golden


# --------------------------------------------------------------------------- #
# 取一个 mode 在固定 prompt 集上的 last-token logits（每条取最后有效位置）
# --------------------------------------------------------------------------- #
def _last_token_logits(
    torch: Any,
    mode: ExecutionMode,
    ds_path: Path,
    *,
    max_samples: int,
    max_input_tokens: int,
) -> Any:
    model, tokenizer = load_build_model(mode)
    device, _ = device_spec_for(model)               # 模型在哪就用哪，不二次搬运
    model = model.eval()

    # 决定性：固定 prompt、固定顺序（不排序），与 golden 一一对应。
    prompts = load_prompt_dataset(ds_path, max_samples=max_samples).prompts
    inputs = tokenize_prompts(
        torch, tokenizer, prompts, device=device.device, max_length=max_input_tokens
    )

    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits if hasattr(out, "logits") else out[0]
    synchronize(torch, device.kind)

    # 每条样本最后一个有效 token 的位置：优先用 attention_mask，否则取末列。
    mask = inputs.get("attention_mask")
    if mask is not None:
        last_idx = mask.long().sum(dim=1) - 1            # [N]
        last_idx = last_idx.clamp(min=0)
    else:
        last_idx = torch.full((logits.shape[0],), logits.shape[1] - 1, dtype=torch.long)
    rows = torch.arange(logits.shape[0])
    last = logits[rows, last_idx.to(logits.device)]      # [N, vocab]
    result = last.float().cpu()                          # 先把要留的搬到 CPU
    # 设备上的 model/out/logits 用完即释放，避免与 golden 计算同时在显存里压两份模型。
    del model, out, logits, last
    release_device_memory(torch, device.kind)
    return result


# --------------------------------------------------------------------------- #
# last-token logits 余弦相似度（逐样本算再取均值）
# --------------------------------------------------------------------------- #
def _mean_cosine(torch: Any, golden: Any, candidate: Any) -> float:
    if golden.shape != candidate.shape:
        # 形状不一致（如 vocab 改变）直接判不等价。
        return 0.0
    cos = torch.nn.functional.cosine_similarity(
        golden.float(), candidate.float(), dim=-1, eps=1e-8
    )
    return float(cos.mean().item())
