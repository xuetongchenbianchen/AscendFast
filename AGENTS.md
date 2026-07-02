# AscendFast Agent Context

This repository runs an iterative NPU inference optimization pipeline for
PyTorch causal LMs on Ascend.

## Runtime

- Project root: `/models/share/userdata/chenjunyang/workspace/26Infer/AscendFast`
- Python: `.venv/bin/python` using Python 3.11
- NPU runtime: Ascend 910, torch 2.7.1, torch_npu 2.7.1.post2
- Current validated model: `model/Qwen2.5-0.5B-Instruct`
- Benchmark data: `data/prompts_sharegpt.jsonl`
- Profile data: `data/prompts_real.jsonl`

Before running Ascend/NPU Python commands, source the runtime environment:

```bash
source scripts/ascend-env.sh
```

## Agent Backend

The pipeline calls named agents through `agent_client.py`. The default backend
remains Claude Code via `claude-agent-sdk`. Codex CLI is available as an
optional backend for environments where Claude Code is unavailable.

Important environment switches:

- `ASCENDFAST_USE_LLM_AGENT=0`: disable LLM agents.
- `ASCENDFAST_AGENT_BACKEND=claude|codex`: choose backend; default is `claude`.
- `ASCENDFAST_CLAUDE_CLI=/path/to/claude`: override Claude Code binary.
- `ASCENDFAST_CODEX_CLI=/path/to/codex`: override Codex CLI path.
- `ASCENDFAST_CODEX_MODEL=<model>`: choose a Codex model.
- `ASCENDFAST_CODEX_SANDBOX=workspace-write`: default child-agent sandbox.
- `ASCENDFAST_CODEX_APPROVAL=never`: default child-agent approval policy.

## Build Model Contract

Every execution mode workspace must expose:

```python
def build_model() -> tuple[object, object]:
    ...
```

All benchmark, profile, correctness, and apply gates load optimized models only
through this function.

Patch imports must stay inside `build_model()`:

```python
def build_model():
    from patches import my_patch
    my_patch.apply()
    ...
```

Do not put workspace patch imports at module top level, because
`workspace_loader.py` isolates workspaces by controlling import state.

## Agent Responsibilities

| Agent | Responsibility | Must Not Do |
|---|---|---|
| profile-agent | Return profiler paths and latency metadata | Interpret optimization strategy |
| analysis-agent | Identify where time is spent | Write code or choose implementation |
| strategy-agent | Choose WHAT/WHY based on profile and analysis evidence | Write code or mutate workspaces |
| apply-agent | Implement the selected strategy in a forked workspace | Change parent workspaces or root project files during candidate application |

## Workspace Discipline

Candidate changes must stay inside the forked execution-mode workspace unless a
later pipeline stage explicitly asks for project-level operator scaffolding.
Root-level probe scripts and temporary files should not be created during normal
strategy application.
