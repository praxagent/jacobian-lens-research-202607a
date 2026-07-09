"""Shared model loader for the behavioral runners.

Keeps two infra quirks out of the runners:
- **torch.accelerator shim.** transformers 5.13's mxfp4 quantizer calls
  ``torch.accelerator.current_accelerator()`` (added in torch 2.6); older torch lacks it.
- **mxfp4 dequant.** gpt-oss ships mxfp4-packed, and the packed backward is incompatible
  with jlens's repeated-backward, so we dequantize to bf16 at load.

For every non-mxfp4 model this reproduces the previous plain
``from_pretrained(..., torch_dtype=...).to(dev)`` exactly, so existing results are unchanged.
"""
from __future__ import annotations

import types

import torch
import transformers

if not hasattr(torch, "accelerator"):  # torch < 2.6 compat for the mxfp4 quantizer
    torch.accelerator = types.SimpleNamespace(
        current_accelerator=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        is_available=lambda: torch.cuda.is_available(),
    )


def _is_mxfp4(hf_id: str) -> bool:
    cfg = transformers.AutoConfig.from_pretrained(hf_id)
    qc = getattr(cfg, "quantization_config", None)
    if qc is None:
        return False
    qm = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
    return str(qm).lower() == "mxfp4"


def load_hf_model(hf_id: str, dev: str):
    """Load an HF causal LM ready for jlens on ``dev`` (dequantizing mxfp4 -> bf16)."""
    dtype = torch.bfloat16 if str(dev).startswith("cuda") else torch.float32
    if _is_mxfp4(hf_id):
        # dequantize=True turns this back into a plain bf16 model; do NOT use device_map
        # (it leaves some dequantized MoE expert weights on CPU -> device-mismatch in the
        # expert matmul). Load, then .to(dev) so every tensor lands on the same device.
        from transformers import Mxfp4Config
        return transformers.AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=dtype,
            quantization_config=Mxfp4Config(dequantize=True),
        ).to(dev).eval()
    return transformers.AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=dtype,
    ).to(dev).eval()
