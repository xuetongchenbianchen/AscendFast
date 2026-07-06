# Optional Codex Agent Backend

AscendFast defaults to the upstream Claude Code backend through
`claude-agent-sdk`. This document describes the optional Codex CLI backend for
environments where Claude Code is unavailable but Codex CLI is installed.

## Backend Selection

Default behavior remains unchanged:

```bash
ASCENDFAST_AGENT_BACKEND=claude
```

To opt into Codex:

```bash
ASCENDFAST_AGENT_BACKEND=codex
```

The public Python interface is unchanged:

```python
from agent_client import call_agent_json

result = call_agent_json("strategy-agent", prompt, timeout=1000)
```

## Claude Backend

The Claude backend keeps using `claude-agent-sdk` and `.claude/agents/*.md` as
before. `ASCENDFAST_CLAUDE_CLI` can override the Claude Code binary path.

## Codex Backend

When `ASCENDFAST_AGENT_BACKEND=codex`, `agent_client.py` runs:

```bash
codex --ask-for-approval never exec \
  --cd /models/share/userdata/chenjunyang/workspace/26Infer/AscendFast \
  --sandbox workspace-write \
  --ephemeral \
  --output-last-message /tmp/ascendfast_<agent>_*.txt \
  -
```

The prompt is sent over stdin. The final Codex message is read from
`--output-last-message` instead of stdout so that progress logs or transcript
text cannot corrupt JSON parsing.

Codex does not automatically load the legacy Claude role files or skills, so
`agent_client.py` explicitly injects:

- `AGENTS.md` project context, if present
- `.claude/agents/<agent>.md` role prompt
- the matching `.claude/skills/*/SKILL.md` playbook, if present

## Environment Variables

- `ASCENDFAST_USE_LLM_AGENT=0`: disable LLM agent calls.
- `ASCENDFAST_AGENT_BACKEND=claude|codex`: choose backend; default is `claude`.
- `ASCENDFAST_CLAUDE_CLI=/path/to/claude`: override Claude Code binary.
- `ASCENDFAST_CODEX_CLI=/path/to/codex`: override Codex binary discovery.
- `ASCENDFAST_CODEX_MODEL=<model>`: pass `--model` to Codex.
- `ASCENDFAST_CODEX_SANDBOX=workspace-write`: choose child-agent sandbox.
- `ASCENDFAST_CODEX_APPROVAL=never`: choose Codex approval policy.

## Failure Interpretation

Both backends preserve the existing ledger and trace status contract:

- `subprocess_error`: backend CLI/SDK could not run or exited non-zero.
- `timeout`: backend exceeded the per-agent timeout.
- `agent_error`: backend returned no successful final message.
- `bad_json`: final text could not be parsed as JSON.
- `ok`: the backend produced a final message; for `call_agent_json`, JSON
  parsing then decides whether the caller receives a structured object.

## NPU Runtime Note

For NPU experiments, run the parent pipeline in the same environment where
`npu-smi info` and `torch.npu.is_available()` work. If a child Codex process
cannot see NPU devices because of sandboxing, apply-agent should still return
its ChangeRecord after static validation; the parent pipeline performs the real
forward gate before accepting the candidate.
