"""Thin bridge: Python → claude CLI headless agent calls."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Set ASCENDFAST_USE_LLM_AGENT=0 to force rule-based fallback (offline / tests).
AGENT_ENABLED: bool = os.environ.get("ASCENDFAST_USE_LLM_AGENT", "1") != "0"

_PROJECT_ROOT = Path(__file__).parent


def call_agent(agent_name: str, prompt: str, *, timeout: int = 120) -> str | None:
    """Run `claude -p <prompt> --agent <agent_name>` headless; return text or None."""
    if not AGENT_ENABLED:
        return None
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--agent", agent_name, "--output-format", "json",
             "--permission-mode", "acceptEdits"],
            capture_output=True, text=True, timeout=timeout, cwd=_PROJECT_ROOT,
        )
        outer = json.loads(result.stdout)
        if outer.get("is_error") or outer.get("subtype") != "success":
            return None
        return outer.get("result")
    except Exception:
        return None


def call_agent_json(agent_name: str, prompt: str, *, timeout: int = 120) -> dict | list | None:
    """Like call_agent but parse .result as JSON; return None on any failure."""
    prompt_with_constraint = (
        prompt + "\n\nIMPORTANT: Reply with ONLY valid JSON, no markdown fences."
    )
    raw = call_agent(agent_name, prompt_with_constraint, timeout=timeout)
    if raw is None:
        return None
    # Strip optional ```json … ``` fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    # Find first { or [
    m = re.search(r"[\[{]", cleaned)
    if m:
        cleaned = cleaned[m.start():]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None
