from __future__ import annotations

import json
from pathlib import Path

from agent_client import AGENT_ENABLED, call_agent_json
from models import AppliedArtifact, ExecutionMode, OptimizationStrategy

_PROJECT_ROOT = Path(__file__).parent


def apply_optimization(
    strategy: OptimizationStrategy,
    model_id: str,
) -> ExecutionMode:
    """Apply strategy via LLM agent; fallback returns ExecutionMode with empty artifacts."""
    work_dir = _PROJECT_ROOT / "adaptations" / model_id / strategy.uid
    work_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[AppliedArtifact] = []
    if AGENT_ENABLED:
        artifacts = _llm_apply_optimization(strategy, work_dir) or []

    return ExecutionMode(
        uid=f"mode:{strategy.uid}",
        model_id=model_id,
        strategy_uid=strategy.uid,
        artifacts=artifacts,
    )


def _llm_apply_optimization(
    strategy: OptimizationStrategy,
    work_dir: Path,
) -> list[AppliedArtifact] | None:
    prompt = (
        f"{strategy.prompt_instruction}\n\n"
        f"Work directory (absolute): {work_dir}\n"
        "Apply the optimization, then return a JSON object describing every file you "
        "created or modified:\n"
        '{"artifacts": [{"kind": "patch|config|weight|graph|custom", '
        '"paths": ["<relative-to-work-dir>"], '
        '"revert_cmd": "<shell cmd to undo, or null>", '
        '"metadata": {}}]}'
    )
    result = call_agent_json("optimization-agent", prompt, timeout=300)
    if not isinstance(result, dict):
        return None
    raw = result.get("artifacts")
    if not isinstance(raw, list) or not raw:
        return None
    artifacts = []
    for item in raw:
        if not isinstance(item, dict) or "kind" not in item or "paths" not in item:
            continue
        artifacts.append(AppliedArtifact(
            kind=str(item["kind"]),
            paths=[str(p) for p in item["paths"]] if isinstance(item["paths"], list) else [],
            revert_cmd=item.get("revert_cmd") or None,
            metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else None,
        ))
    return artifacts or None
