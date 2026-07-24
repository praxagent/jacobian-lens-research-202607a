"""Fetch the assets the readout-ablation needs: neuronpedia Jacobian lenses + each model's
embedding shard (tied -> unembedding U), for the flat gemma-2 smalls, the STRUCTURED
same-family control gemma-2-27b, and a cross-family control qwen3-4b.

Extracts only embed_tokens.weight to a local fp16 .npy (no full model). RAM-frugal: reads
one shard, pulls one tensor. HF_TOKEN in env for the gated gemma repos.
"""
from __future__ import annotations
import json, sys
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors import safe_open

OUT = Path(__file__).resolve().parent / "decompose_out"
OUT.mkdir(exist_ok=True)
LENS_REPO = "neuronpedia/jacobian-lens"
TGT = "model.embed_tokens.weight"

MODELS = {
    "gemma-2-2b":  "google/gemma-2-2b",
    "gemma-2-9b":  "google/gemma-2-9b",
    "gemma-2-27b": "google/gemma-2-27b",
    "qwen3-4b":    "Qwen/Qwen3-4B",
    "gemma-3-1b":  "google/gemma-3-1b-pt",
}


def fetch_lens(slug):
    """Resolve the lens filename from the repo listing (it is NOT always the slug: e.g.
    qwen3-4b's file is Qwen3-4B_jacobian_lens.pt), then download it."""
    dst = OUT / f"{slug}_np_lens.pt"
    if dst.exists():
        print(f"[{slug}] lens cached", flush=True); return
    from huggingface_hub import list_repo_files
    cands = [f for f in list_repo_files(LENS_REPO)
             if f.startswith(slug + "/") and f.endswith("_jacobian_lens.pt")]
    if not cands:
        raise FileNotFoundError(f"no lens .pt under {slug}/ in {LENS_REPO}")
    p = hf_hub_download(LENS_REPO, sorted(cands)[0])
    Path(dst).symlink_to(p)
    print(f"[{slug}] lens -> {p}", flush=True)


def fetch_embed(slug, repo):
    dst = OUT / f"{slug}_embed.npy"
    if dst.exists():
        print(f"[{slug}] embed cached {dst.stat().st_size/1e9:.2f}GB", flush=True); return
    try:
        idx = hf_hub_download(repo, "model.safetensors.index.json")
        shard = json.load(open(idx))["weight_map"][TGT]
    except Exception:
        shard = "model.safetensors"          # single-shard checkpoints
    sp = hf_hub_download(repo, shard)
    # framework="pt": numpy cannot represent bfloat16 (Qwen ships bf16; Gemma shards are f32)
    import torch
    with safe_open(sp, framework="pt") as f:
        key = TGT if TGT in f.keys() else next(k for k in f.keys() if k.endswith("embed_tokens.weight"))
        W = f.get_tensor(key)
    np.save(dst, W.to(torch.float16).numpy())
    print(f"[{slug}] embed {W.shape} -> {dst} ({dst.stat().st_size/1e9:.2f}GB)", flush=True)


if __name__ == "__main__":
    only = sys.argv[1:] or list(MODELS)
    for slug in only:
        fetch_lens(slug)
        fetch_embed(slug, MODELS[slug])
    print("FETCH_ASSETS_DONE", flush=True)
