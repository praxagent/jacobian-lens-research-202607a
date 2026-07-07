"""Load ONLY a model's unembedding matrix — memory-minimal, no full-model load.

The CKA emergence sweep never runs the model forward; it needs just U (vocab x
d_model). Loading the whole transformer to grab U wastes gigabytes and caps how
far up the size ladder a CPU box can climb. This pulls only the embedding tensor
from the model's safetensors shards.

Handles tied embeddings (most modern decoders: lm_head == embed_tokens) and the
common tensor-name variants. Falls back to a full (fp16) model load if the
partial path can't find the tensor.
"""
from __future__ import annotations

import json

import torch

# candidate tensor names for the (un)embedding, most-specific first
_UNEMBED_NAMES = (
    "lm_head.weight",
    "model.embed_tokens.weight",
    "embed_tokens.weight",
    "transformer.wte.weight",
    "wte.weight",
    "gpt_neox.embed_in.weight",
    "model.embed_in.weight",
)


def load_unembedding(hf_id: str) -> torch.Tensor:
    """Return U as a float32 CPU tensor (vocab, d_model)."""
    try:
        return _load_partial(hf_id)
    except Exception as e:  # pragma: no cover - fallback path
        print(f"  (partial unembed load failed: {type(e).__name__}: {str(e)[:80]}; "
              f"falling back to full fp16 model)")
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16)
        return m.get_output_embeddings().weight.detach().float()


def _load_partial(hf_id: str) -> torch.Tensor:
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    # 1) sharded checkpoint: read the index, find which shard holds the tensor
    try:
        idx_path = hf_hub_download(hf_id, filename="model.safetensors.index.json")
        weight_map = json.load(open(idx_path))["weight_map"]
        name = _pick(weight_map.keys())
        shard = weight_map[name]
        path = hf_hub_download(hf_id, filename=shard)
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.get_tensor(name).float()
    except Exception:
        pass

    # 2) single-file checkpoint
    path = hf_hub_download(hf_id, filename="model.safetensors")
    with safe_open(path, framework="pt", device="cpu") as f:
        name = _pick(f.keys())
        return f.get_tensor(name).float()


def _pick(keys) -> str:
    keys = set(keys)
    for n in _UNEMBED_NAMES:
        if n in keys:
            return n
    # last resort: anything ending in embed_tokens.weight / wte.weight
    for k in keys:
        if k.endswith(("embed_tokens.weight", "wte.weight", "embed_in.weight")):
            return k
    raise KeyError(f"no unembedding tensor among {len(keys)} tensors")
