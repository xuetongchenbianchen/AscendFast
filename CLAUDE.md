# AscendFast — Project Context for All Agents

## Hardware & Runtime

- Device: Ascend 910 NPU (`npu:0`)
- torch_npu: 2.7.1.post2
- transformers: 4.57.1
- Python: 3.10
- venv: `/models/share/userdata/cb/AscendFast/.venv`
- Run commands from project root: `/models/share/userdata/cb/AscendFast`

## transformers 4.57.1 — Available Qwen2 classes

```python
# AVAILABLE
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

# NOT AVAILABLE — do not import, do not reference
# Qwen2FlashAttention2   (added in transformers ≥ 4.40, not present here)
# Qwen2SdpaAttention     (added in transformers ≥ 4.40, not present here)
```

## torch_npu fused ops (check before use)

```python
import torch_npu
hasattr(torch_npu, "npu_rms_norm")      # True — safe to use
hasattr(torch_npu, "npu_rotary_mul")    # check at runtime
# Always guard with hasattr() — never assume a fused op exists
```

## Critical import rule for build_model.py

All `from patches import ...` statements MUST be inside the `build_model()`
function body, NOT at module top-level:

```python
# CORRECT
def build_model(...):
    from patches import my_patch   # inside function
    my_patch.apply()
    ...

# WRONG — causes sys.modules pollution across workspace isolation
from patches import my_patch       # top-level
```

This is because `workspace_loader.py` isolates each workspace's modules by
snapshotting `sys.modules` before and after import. A top-level patch import
runs at `exec_module()` time and can collide with a same-named `patches`
package from a previously loaded workspace.

## Mandatory smoke test for optimization-agent

After writing any code, run this before returning JSON:

```bash
cd /models/share/userdata/cb/AscendFast && python -c "
import importlib.util, sys
ws = '<absolute workspace path>'
sys.path.insert(0, ws)
spec = importlib.util.spec_from_file_location('bm', ws + '/build_model.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print('smoke OK, build_model:', type(m.build_model))
"
```

If it raises any exception, fix the code first. Do not return JSON with a
broken workspace.

## Project layout

```
model/                  # original weights (read-only)
adaptations/            # ExecutionMode workspaces
  <model_id>/
    baseline/           # baseline ExecutionMode
    mode_<...>/         # optimized forks
data/
  prompts_real.jsonl    # profile dataset (simulated, small)
  prompts_sharegpt.jsonl # benchmark dataset (real, 64 samples)
runs/                   # RunLedger JSON files (one per optimization run)
```

## Agent roles (do not cross boundaries)

| Agent | Does | Does NOT |
|---|---|---|
| profile-agent | run profiler, return paths + latency | interpret results |
| analysis-agent | identify WHERE time is spent | propose fixes |
| strategy-agent | propose WHAT to optimize | write code |
| optimization-agent | write + verify code | invent strategies |
