"""Thin bridge: Python → claude headless agent calls via claude-agent-sdk."""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
from pathlib import Path

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKError,
    ResultMessage,
    query,
)

from verify import record_agent_call
from trace_store import record_agent_io

# Set ASCENDFAST_USE_LLM_AGENT=0 to force rule-based fallback (offline / tests).
AGENT_ENABLED: bool = os.environ.get("ASCENDFAST_USE_LLM_AGENT", "1") != "0"

_PROJECT_ROOT = Path(__file__).parent

# Pin the claude binary so calls don't depend on PATH (the project venv does not
# expose `claude` itself; it lives in the conda env that ships Claude Code).
_CLI_PATH = "/root/miniconda3/envs/llm_test/bin/claude"

# SDK process-layer failures all map to the existing "subprocess_error" ledger
# status — the label predates the SDK migration but its meaning is unchanged:
# "failed to invoke / talk to the claude CLI process".
# CLINotFoundError / CLIConnectionError / ProcessError / CLIJSONDecodeError are
# all subclasses of ClaudeSDKError, so the base class alone covers them.
_PROCESS_ERRORS = (ClaudeSDKError,)
_LAST_CALL_STATUS: tuple[str, str] = ("unknown", "")


async def _run_agent(agent_name: str, prompt: str) -> str | None:
    """Drive a single headless agent query; return its final text or None.

    Returns None when the run produced no successful ResultMessage (is_error or
    subtype != "success"). Raises SDK exceptions on process-layer failures —
    call_agent maps those to ledger statuses.
    """
    opts = ClaudeAgentOptions(
        cwd=str(_PROJECT_ROOT),
        permission_mode="acceptEdits",
        setting_sources=["project"],        # 必须：否则读不到 .claude/agents/*.md
        extra_args={"agent": agent_name},   # 必须：ClaudeAgentOptions 无 agent 字段，透传 --agent
        cli_path=_CLI_PATH,
    )
    result: ResultMessage | None = None
    # 显式持有 query() 这个 async generator，并在 finally 里 aclose()。
    # 为什么不能只写 `async for ... in query(...)`：
    # SDK 在 generator 内层用 `finally: await transport.close()` 回收它 spawn 的
    # claude 子进程（SIGTERM→SIGKILL 三级兜底）。但 wait_for 超时取消时，裸 async for
    # 的 generator 清理依赖 asyncio.run() 退出时的 shutdown_asyncgens() 隐式钩子——
    # 一旦有人把它跑在持久 event loop 上（不每次 shutdown_asyncgens），那条 finally
    # 就不保证即时执行，claude 子进程会泄漏成僵尸（长 run 下每次 1000s 超时累积）。
    # 显式 aclose() 把这份回收从"借 asyncio.run 的隐式钩子"变成我方控制、即时触发。
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


def call_agent(
    agent_name: str,
    prompt: str,
    *,
    timeout: int = 120,
    trace: bool = True,
    trace_stage: str | None = None,
) -> str | None:
    """Run a headless `--agent <agent_name>` query via claude-agent-sdk; return text or None.

    Before every None return, log an agent_call StageOutcome distinguishing the
    failure kind (disabled/timeout/subprocess_error/agent_error) so "为什么没效果"
    stops being a black box. The None contract itself is unchanged; callers don't move.
    """
    started = time.perf_counter()
    stage_name = trace_stage or _stage_from_agent(agent_name)
    if not AGENT_ENABLED:
        _set_last_call_status("disabled", "agent runtime disabled")
        record_agent_call(agent_name, "disabled")
        if trace:
            record_agent_io(
                agent_name, stage_name, prompt, None, None, "disabled",
                detail="agent runtime disabled",
                duration_ms=_elapsed_ms(started),
            )
        return None
    try:
        result = asyncio.run(asyncio.wait_for(_run_agent(agent_name, prompt), timeout))
    except asyncio.TimeoutError:
        _set_last_call_status("timeout", f"exceeded {timeout}s")
        record_agent_call(agent_name, "timeout", f"exceeded {timeout}s")
        if trace:
            record_agent_io(
                agent_name, stage_name, prompt, None, None, "timeout",
                detail=f"exceeded {timeout}s",
                duration_ms=_elapsed_ms(started),
            )
        return None
    except _PROCESS_ERRORS as exc:
        detail = f"{type(exc).__name__}: {exc}"
        _set_last_call_status("subprocess_error", detail)
        record_agent_call(agent_name, "subprocess_error", detail)
        if trace:
            record_agent_io(
                agent_name, stage_name, prompt, None, None, "subprocess_error",
                detail=detail,
                duration_ms=_elapsed_ms(started),
            )
        return None
    except Exception as exc:  # noqa: BLE001 - 非进程层异常：多半是 agent_client 自身的 bug
        detail = f"{type(exc).__name__}: {exc}"
        _set_last_call_status("unexpected", detail)
        record_agent_call(agent_name, "unexpected", detail)
        if trace:
            record_agent_io(
                agent_name, stage_name, prompt, None, None, "unexpected",
                detail=detail,
                duration_ms=_elapsed_ms(started),
            )
        return None
    if result is None:
        _set_last_call_status("agent_error", "no successful ResultMessage")
        record_agent_call(agent_name, "agent_error", "no successful ResultMessage")
        if trace:
            record_agent_io(
                agent_name, stage_name, prompt, None, None, "agent_error",
                detail="no successful ResultMessage",
                duration_ms=_elapsed_ms(started),
            )
        return None
    record_agent_call(agent_name, "ok")
    _set_last_call_status("ok", "")
    if trace:
        record_agent_io(
            agent_name, stage_name, prompt, result, None, "ok",
            duration_ms=_elapsed_ms(started),
        )
    return result


def call_agent_json(agent_name: str, prompt: str, *, timeout: int = 1200) -> dict | list | None:
    """Like call_agent but parse .result as JSON; return None on any failure."""
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
        record_agent_io(
            agent_name, stage_name, prompt_with_constraint, None, None, status,
            detail=detail or "call_agent returned None",
            duration_ms=_elapsed_ms(started),
        )
        return None
    # Strip optional ```json … ``` fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    # Find first { or [
    m = re.search(r"[\[{]", cleaned)
    if m:
        cleaned = cleaned[m.start():]
    try:
        parsed = json.loads(cleaned)
        record_agent_io(
            agent_name, stage_name, prompt_with_constraint, raw, parsed, "ok",
            duration_ms=_elapsed_ms(started),
        )
        return parsed
    except json.JSONDecodeError:
        record_agent_call(agent_name, "bad_json", "result was not parseable JSON")
        record_agent_io(
            agent_name, stage_name, prompt_with_constraint, raw, None, "bad_json",
            detail="result was not parseable JSON",
            duration_ms=_elapsed_ms(started),
        )
        return None


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _set_last_call_status(status: str, detail: str = "") -> None:
    global _LAST_CALL_STATUS
    _LAST_CALL_STATUS = (status, detail)


def _stage_from_agent(agent_name: str) -> str:
    if agent_name.endswith("-agent"):
        return agent_name[: -len("-agent")]
    return agent_name
