"""Environment probe — run this before writing any optimization code.

Usage:
    python /models/share/userdata/cb/AscendFast/env_probe.py

Prints a structured summary of the actual runtime environment so agents
never have to guess what APIs exist.
"""
import sys

def _version(pkg):
    try:
        import importlib.metadata
        return importlib.metadata.version(pkg)
    except Exception:
        return "not installed"

def _qwen2_attn_classes():
    try:
        import importlib
        mod = importlib.import_module("transformers.models.qwen2.modeling_qwen2")
        candidates = ["Qwen2Attention", "Qwen2FlashAttention2", "Qwen2SdpaAttention"]
        return [c for c in candidates if hasattr(mod, c)]
    except Exception as e:
        return [f"ERROR: {e}"]

def _npu_ops():
    try:
        import torch_npu
        candidates = [
            "npu_rms_norm", "npu_rotary_mul", "npu_scaled_masked_softmax",
            "npu_flash_attention", "npu_fusion_attention", "npu_apply_rotary_pos_emb",
        ]
        return [op for op in candidates if hasattr(torch_npu, op)]
    except Exception as e:
        return [f"torch_npu unavailable: {e}"]

def main():
    print("=== AscendFast env_probe ===")
    print(f"python:        {sys.version.split()[0]}")
    print(f"transformers:  {_version('transformers')}")
    print(f"torch:         {_version('torch')}")
    print(f"torch_npu:     {_version('torch_npu')}")
    print()
    attn = _qwen2_attn_classes()
    print(f"Qwen2 attn classes available:  {attn}")
    print(f"  NOT available: {[c for c in ['Qwen2Attention','Qwen2FlashAttention2','Qwen2SdpaAttention'] if c not in attn]}")
    print()
    ops = _npu_ops()
    print(f"torch_npu fused ops available: {ops}")
    print()
    print("=== import rules (enforced by workspace_loader isolation) ===")
    print("  patches imports MUST be inside build_model() body, NOT module top-level.")
    print("  Guard any transformers internal class: try/except ImportError.")

if __name__ == "__main__":
    main()
