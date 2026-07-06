"""Thin bridge: Python pipeline -> selectable headless agent backend."""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from verify import record_agent_call
from trace_store import record_agent_io

# Set ASCENDFAST_USE_LLM_AGENT=0 to force rule-based fallback/offline tests.
AGENT_ENABLED: bool = os.environ.get("ASCENDFAST_USE_LLM_AGENT", "1") != "0"

_PROJECT_ROOT = Path(__file__).parent
_AGENT_DIR = _PROJECT_ROOT / ".claude" / "agents"
_SKILL_DIR = _PROJECT_ROOT / ".claude" / "skills"
_PROJECT_CONTEXT = _PROJECT_ROOT / "AGENTS.md"

# Claude remains the default upstream-compatible backend. Codex is opt-in for
# environments where Claude Code is unavailable but Codex CLI is installed.
_DEFAULT_BACKEND = "claude"
_CLAUDE_CLI_PATH = "/root/miniconda3/envs/llm_test/bin/claude"
_LAST_CALL_STATUS: tuple[str, str] = ("unknown", "")

_AGENT_SKILLS = {
    "strategy-agent": ("npu-strategy",),
    "apply-agent": ("npu-apply",),
    "analysis-agent": ("npu-analysis",),
    "operator-agent": ("npu-operator",),
}


class CodexAgentError(RuntimeError):
    """Raised when Codex CLI cannot complete an agent request."""


def _agent_backend() -> str:
    backend = os.environ.get("ASCENDFAST_AGENT_BACKEND", _DEFAULT_BACKEND).strip().lower()
    aliases = {
        "cc": "claude",
        "claude-code": "claude",
        "claude_code": "claude",
        "codex-cli": "codex",
        "codex_cli": "codex",
    }
    return aliases.get(backend, backend)


async def _run_claude_agent(agent_name: str, prompt: str) -> str | None:
    """Drive a single Claude Code headless agent query."""
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    opts = ClaudeAgentOptions(
        cwd=str(_PROJECT_ROOT),
        permission_mode="acceptEdits",
        setting_sources=["project"],
        extra_args={"agent": agent_name},
        cli_path=os.environ.get("ASCENDFAST_CLAUDE_CLI", _CLAUDE_CLI_PATH),
    )
    result: ResultMessage | None = None
    # Hold the async generator explicitly so timeout cancellation closes the
    # underlying Claude subprocess promptly instead of relying on event-loop
    # shutdown hooks.
    agen = query(prompt=prompt, options=opts)
    try:
        async for msg in agen:
            if isinstance(msg, ResultMessage):
                result = msg
    finally:
        await agen.aclose()
    if result is None or result.is_error or result.subtype != "success":
        return None
    return result.result


def _is_claude_sdk_error(exc: BaseException) -> bool:
    try:
        from claude_agent_sdk import ClaudeSDKError
    except Exception:
        return False
    return isinstance(exc, ClaudeSDKError)


def _codex_cli_path() -> str:
    configured = os.environ.get("ASCENDFAST_CODEX_CLI", "").strip()
    if configured:
        return configured

    # Prefer the native Codex binary over the npm wrapper. The wrapper starts
    # with `#!/usr/bin/env node`, so it fails in NPU experiment shells where
    # Node.js is not on PATH even though Codex itself is installed.
    native = Path(
        "/models/share/userdata/chenjunyang/node_modules/"
        "@openai/codex-linux-arm64/vendor/aarch64-unknown-linux-musl/bin/codex"
    )
    if native.exists() and os.access(native, os.X_OK):
        return str(native)

    found = shutil.which("codex")
    if found:
        return found

    wrapper = Path("/models/share/userdata/chenjunyang/node_modules/.bin/codex")
    if wrapper.exists():
        return str(wrapper)
    raise FileNotFoundError(
        "Codex CLI not found. Set ASCENDFAST_CODEX_CLI or add `codex` to PATH."
    )


def _agent_role_prompt(agent_name: str) -> str:
    """Load the existing role prompt for a named project agent."""
    agent_path = _AGENT_DIR / f"{agent_name}.md"
    if not agent_path.exists():
        raise FileNotFoundError(f"agent prompt not found: {agent_path}")
    return agent_path.read_text(encoding="utf-8")


def _project_context_prompt() -> str:
    if _PROJECT_CONTEXT.exists():
        return _PROJECT_CONTEXT.read_text(encoding="utf-8")
    legacy_context = _PROJECT_ROOT / "CLAUDE.md"
    if legacy_context.exists():
        return legacy_context.read_text(encoding="utf-8")
    return ""


def _skill_context_prompt(agent_name: str) -> str:
    """Inline legacy Claude skills so Codex sub-agents see the original playbook."""
    chunks: list[str] = []
    for skill_name in _AGENT_SKILLS.get(agent_name, ()):  # unknown agents get no skill context
        skill_path = _SKILL_DIR / skill_name / "SKILL.md"
        if skill_path.exists():
            chunks.append(f"## Skill: {skill_name}\n{skill_path.read_text(encoding='utf-8')}")
    return "\n\n".join(chunks)


def _build_codex_prompt(agent_name: str, prompt: str) -> str:
    role = _agent_role_prompt(agent_name)
    context = _project_context_prompt()
    skills = _skill_context_prompt(agent_name)
    skill_section = f"## Legacy skill playbook\n{skills}\n\n" if skills else ""
    return (
        "You are running as a named AscendFast pipeline agent.\n"
        "Follow the role contract exactly. The final answer must satisfy the "
        "pipeline prompt; do not add markdown fences around machine-readable JSON.\n\n"
        "## Project context\n"
        f"{context}\n\n"
        "## Agent role file\n"
        f"{role}\n\n"
        f"{skill_section}"
        "## Pipeline request\n"
        f"{prompt}"
    )


def _run_codex_agent(agent_name: str, prompt: str, *, timeout: int) -> str:
    """Run `codex exec` and return its final message."""
    cli = _codex_cli_path()
    sandbox = os.environ.get("ASCENDFAST_CODEX_SANDBOX", "workspace-write").strip()
    approval = os.environ.get("ASCENDFAST_CODEX_APPROVAL", "never").strip()
    model = os.environ.get("ASCENDFAST_CODEX_MODEL", "").strip()

    with tempfile.NamedTemporaryFile(
        prefix=f"ascendfast_{agent_name}_", suffix=".txt", delete=False
    ) as output_file:
        output_path = Path(output_file.name)
    cmd = [
        cli,
        "--ask-for-approval",
        approval,
        "exec",
        "--cd",
        str(_PROJECT_ROOT),
        "--sandbox",
        sandbox,
        "--ephemeral",
        "--output-last-message",
        str(output_path),
    ]
    if model:
        cmd.extend(["--model", model])
    cmd.append("-")

    env = os.environ.copy()
    # Avoid leaking provider-specific Claude variables into the Codex subprocess.
    env.pop("CLAUDECODE", None)

    try:
        completed = subprocess.run(
            cmd,
            input=_build_codex_prompt(agent_name, prompt),
            text=True,
            capture_output=True,
            cwd=str(_PROJECT_ROOT),
            env=env,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            detail = stderr or stdout or f"exit status {completed.returncode}"
            raise CodexAgentError(detail[-4000:])
        if output_path.exists():
            result = output_path.read_text(encoding="utf-8").strip()
            if result:
                return result
        return completed.stdout.strip()
    finally:
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass


def _record_agent_trace(
    agent_name: str,
    stage_name: str,
    prompt: str,
    raw_response: str | None,
    parsed_response: Any | None,
    status: str,
    *,
    detail: str = "",
    duration_ms: float | None = None,
) -> None:
    record_agent_io(
        agent_name,
        stage_name,
        prompt,
        raw_response,
        parsed_response,
        status,
        detail=detail,
        duration_ms=duration_ms,
    )


def call_agent(
    agent_name: str,
    prompt: str,
    *,
    timeout: int = 120,
    trace: bool = True,
    trace_stage: str | None = None,
) -> str | None:
    """Run a named project agent through the selected backend; return text or None.

    Default backend is Claude Code for upstream compatibility. Set
    ASCENDFAST_AGENT_BACKEND=codex to use Codex CLI instead.
    """
    started = time.perf_counter()
    stage_name = trace_stage or _stage_from_agent(agent_name)
    if not AGENT_ENABLED:
        _set_last_call_status("disabled", "agent runtime disabled")
        record_agent_call(agent_name, "disabled")
        if trace:
            _record_agent_trace(
                agent_name, stage_name, prompt, None, None, "disabled",
                detail="agent runtime disabled", duration_ms=_elapsed_ms(started),
            )
        return None

    backend = _agent_backend()
    try:
        if backend == "claude":
            result = asyncio.run(asyncio.wait_for(_run_claude_agent(agent_name, prompt), timeout))
        elif backend == "codex":
            result = _run_codex_agent(agent_name, prompt, timeout=timeout)
        else:
            raise ValueError(
                "Unsupported ASCENDFAST_AGENT_BACKEND "
                f"{backend!r}; expected 'claude' or 'codex'."
            )
    except (asyncio.TimeoutError, subprocess.TimeoutExpired):
        detail = f"exceeded {timeout}s"
        _set_last_call_status("timeout", detail)
        record_agent_call(agent_name, "timeout", detail)
        if trace:
            _record_agent_trace(
                agent_name, stage_name, prompt, None, None, "timeout",
                detail=detail, duration_ms=_elapsed_ms(started),
            )
        return None
    except (FileNotFoundError, CodexAgentError, OSError) as exc:
        detail = f"{type(exc).__name__}: {exc}"
        _set_last_call_status("subprocess_error", detail)
        record_agent_call(agent_name, "subprocess_error", detail)
        if trace:
            _record_agent_trace(
                agent_name, stage_name, prompt, None, None, "subprocess_error",
                detail=detail, duration_ms=_elapsed_ms(started),
            )
        return None
    except Exception as exc:  # noqa: BLE001 - distinguish backend process errors from client bugs
        detail = f"{type(exc).__name__}: {exc}"
        status = "subprocess_error" if backend == "claude" and _is_claude_sdk_error(exc) else "unexpected"
        _set_last_call_status(status, detail)
        record_agent_call(agent_name, status, detail)
        if trace:
            _record_agent_trace(
                agent_name, stage_name, prompt, None, None, status,
                detail=detail, duration_ms=_elapsed_ms(started),
            )
        return None

    if result is None:
        detail = "no successful agent result"
        _set_last_call_status("agent_error", detail)
        record_agent_call(agent_name, "agent_error", detail)
        if trace:
            _record_agent_trace(
                agent_name, stage_name, prompt, None, None, "agent_error",
                detail=detail, duration_ms=_elapsed_ms(started),
            )
        return None
    record_agent_call(agent_name, "ok")
    _set_last_call_status("ok", "")
    if trace:
        _record_agent_trace(
            agent_name, stage_name, prompt, result, None, "ok",
            duration_ms=_elapsed_ms(started),
        )
    return result


def call_agent_json(agent_name: str, prompt: str, *, timeout: int = 1200) -> dict | list | None:
    """Like call_agent but parse the agent result as JSON; return None on failure."""
    started = time.perf_counter()
    stage_name = _stage_from_agent(agent_name)
    prompt_with_constraint = (
        prompt + "\n\nIMPORTANT: Reply with ONLY valid JSON, no markdown fences."
    )
    raw = call_agent(
        agent_name, prompt_with_constraint, timeout=timeout,
        trace=False, trace_stage=stage_name,
    )
    if raw is None:
        status, detail = _LAST_CALL_STATUS
        _record_agent_trace(
            agent_name, stage_name, prompt_with_constraint, None, None, status,
            detail=detail or "call_agent returned None", duration_ms=_elapsed_ms(started),
        )
        return None

    cleaned = _extract_json_text(raw)
    try:
        parsed = json.loads(cleaned)
        _record_agent_trace(
            agent_name, stage_name, prompt_with_constraint, raw, parsed, "ok",
            duration_ms=_elapsed_ms(started),
        )
        return parsed
    except json.JSONDecodeError:
        detail = "result was not parseable JSON"
        record_agent_call(agent_name, "bad_json", detail)
        _record_agent_trace(
            agent_name, stage_name, prompt_with_constraint, raw, None, "bad_json",
            detail=detail, duration_ms=_elapsed_ms(started),
        )
        return None


def _extract_json_text(raw: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    start = min(
        (idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx >= 0),
        default=-1,
    )
    if start < 0:
        return cleaned
    cleaned = cleaned[start:]

    # Keep only the first balanced top-level JSON value. This tolerates a short
    # trailing note while still rejecting malformed JSON.
    stack: list[str] = []
    in_string = False
    escape = False
    for index, char in enumerate(cleaned):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or stack[-1] != char:
                break
            stack.pop()
            if not stack:
                return cleaned[: index + 1]
    return cleaned


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _set_last_call_status(status: str, detail: str = "") -> None:
    global _LAST_CALL_STATUS
    _LAST_CALL_STATUS = (status, detail)


def _stage_from_agent(agent_name: str) -> str:
    if agent_name.endswith("-agent"):
        return agent_name[: -len("-agent")]
    return agent_name
