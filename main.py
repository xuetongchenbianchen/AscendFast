"""AscendFast 优化管线的命令行入口。

用法（先 source 环境，再跑）：
    source scripts/ascend-env.sh
    python main.py --model-id Qwen2.5-0.5B-Instruct

首跑建议小规模（先把管线跑通、看 ledger，再放开整棵树）：
    python main.py --model-id Qwen2.5-0.5B-Instruct --top-k 2 --max-depth 2

跑完会打印最优 mode / 延迟 / 加速比，并指向本次 run 的 ledger（runs/<run_uid>.json）——
那里记着这次探索了哪棵树、每个环节成败、为什么停。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from optimization import run

_PROJECT_ROOT = Path(__file__).resolve().parent
_RUNS_DIR = _PROJECT_ROOT / "runs"


def _default_model_dir(model_id: str) -> str:
    """约定：原始权重放在 model/<model_id>/ 下（与 ensure_baseline_mode 镜像它）。"""
    return str(_PROJECT_ROOT / "model" / model_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Iteratively optimize a causal LM on Ascend NPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-id", required=True,
        help="模型标识，对应 model/<model-id>/（如 Qwen2.5-0.5B-Instruct）。",
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="原始权重目录；缺省取 model/<model-id>/。",
    )
    parser.add_argument(
        "--top-k", type=int, default=2,
        help="每个节点最多尝试的策略数（扇出宽度）。首跑建议 2。",
    )
    parser.add_argument(
        "--max-depth", type=int, default=2,
        help="优化链最大深度（baseline 为 depth=0）。首跑建议 2。",
    )
    args = parser.parse_args()

    model_dir = args.model_dir or _default_model_dir(args.model_id)
    if not Path(model_dir).exists():
        parser.error(f"模型目录不存在: {model_dir}（用 --model-dir 显式指定）")

    print("=" * 70)
    print(f"AscendFast run  |  model={args.model_id}")
    print(f"  model_dir = {model_dir}")
    print(f"  top_k     = {args.top_k}    max_depth = {args.max_depth}")
    print("=" * 70)

    best_mode, best_lat = run(
        args.model_id, model_dir, top_k=args.top_k, max_depth=args.max_depth,
    )

    print("\n" + "=" * 70)
    print("DONE")
    print(f"  best mode    : {best_mode.uid}")
    print(f"  best latency : {best_lat:.4f} ms")
    print(f"  workspace    : {best_mode.workspace_dir}")
    print(f"  优化步数      : {len(best_mode.change_log)} 步叠加")
    # 指向本次 run 的 ledger（最新那个）：stop_reason / 每环节成败都在里面。
    if _RUNS_DIR.exists():
        ledgers = sorted(_RUNS_DIR.glob("run_*.json"))
        if ledgers:
            print(f"  ledger       : {ledgers[-1]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
