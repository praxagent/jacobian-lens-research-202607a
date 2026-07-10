"""Consumer-path verification of the published 397B lens (release battery item 3).

Simulates an independent user: download the lens from the public HF repo
(praxagent/jacobian-lens-qwen3.5-397b-a17b), verify its sha256 against the pod-original
receipt, and recompute the band statistic from the public copy — same probe pipeline as
fit_at_scale.py (seed 0, n_probe 4096, U from the base model's lm_head) — then compare
to the shipped band.json (mid_sep +0.3434).

Memory-minimal for a small CPU box: only the lm_head shard of the 397B is downloaded
(never the full 807 GB), and layer geometries are stored fp16 / CKA computed fp32 —
the exact storage/compute split the 34-model A/B measured at max |Δ mid_sep| = 0.00000.

Run:  .venv/bin/python consumer_check_397b.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "jacobian_lens"))
from cka_layers import band_stats  # noqa: E402
sys.path.insert(0, str(HERE.parents[1]))
from common.cka import linear_cka  # noqa: E402

LENS_REPO = "praxagent/jacobian-lens-qwen3.5-397b-a17b"
LENS_FILE = "jlens/wikitext/qwen35_397b.pt"
BASE_MODEL = "Qwen/Qwen3.5-397B-A17B"
POD_SHA256 = "668c3bf17305b0d52495cb7ba589a1c1173301b1d13c3c6ad84e58245dc99e97"
REF_BAND = HERE.parents[1] / "artifacts" / "lenses-397b" / "qwen35_397b_dm.band.json"
SEED, N_PROBE = 0, 4096


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_unembedding_partial(hf_id: str) -> torch.Tensor:
    """Download only the shard holding lm_head.weight, return it (bf16/fp16 CPU)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    idx = json.load(open(hf_hub_download(hf_id, "model.safetensors.index.json")))
    weight_map = idx["weight_map"]
    for name in ("lm_head.weight", "model.embed_tokens.weight",
                 "model.language_model.embed_tokens.weight"):
        if name in weight_map:
            break
    else:
        raise KeyError(f"no unembedding tensor in weight_map ({len(weight_map)} keys)")
    shard = hf_hub_download(hf_id, weight_map[name])
    print(f"unembedding tensor: {name}  (shard {weight_map[name]})")
    with safe_open(shard, framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def main() -> None:
    from huggingface_hub import hf_hub_download

    print(f"=== consumer path: {LENS_REPO}/{LENS_FILE} ===")
    lens_path = hf_hub_download(LENS_REPO, LENS_FILE)
    got = sha256(lens_path)
    print(f"sha256(download) = {got}")
    print(f"sha256(pod orig) = {POD_SHA256}")
    assert got == POD_SHA256, "HASH MISMATCH — public copy differs from pod original!"
    print("hash: MATCH")

    # U first, freed before the 2 GB lens dict loads (7 GB box)
    U = load_unembedding_partial(BASE_MODEL)
    vocab, d_model = U.shape
    print(f"U: {vocab} x {d_model} ({U.dtype})")
    rng = np.random.default_rng(SEED)
    probe = rng.choice(vocab, size=min(N_PROBE, vocab), replace=False)
    Up = U[probe].float()  # same indexing/order as fit_at_scale.py
    del U

    d = torch.load(lens_path, map_location="cpu", weights_only=False)
    layers = d["source_layers"]
    print(f"lens: {len(layers)} source layers, d_model {d['d_model']}, "
          f"n_prompts {d['n_prompts']}")

    geom: dict[int, np.ndarray] = {}
    for l in layers:
        J = d["J"][l].float()
        geom[l] = (Up @ J).to(torch.float16).numpy()
        del d["J"][l]  # free as we go — 7 GB box
    del d

    L = len(layers)
    M = np.eye(L)
    for i in range(L):
        gi = geom[layers[i]].astype(np.float32)
        for j in range(i + 1, L):
            M[i, j] = M[j, i] = linear_cka(gi, geom[layers[j]].astype(np.float32))
    s = band_stats(M, layers)

    ref = json.load(open(REF_BAND))
    print(f"\nmid_sep (recomputed from public copy) = {s['mid_sep']:+.6f}")
    print(f"mid_sep (shipped band.json)            = {ref['mid_sep']:+.6f}")
    print(f"|delta| = {abs(s['mid_sep'] - ref['mid_sep']):.6f}")
    verdict = "PASS" if abs(s["mid_sep"] - ref["mid_sep"]) < 5e-3 else "FAIL"
    print(f"CONSUMER_CHECK_{verdict}")

    out = HERE.parents[1] / "artifacts" / "lenses-397b" / "consumer_check_397b.json"
    json.dump({"lens_repo": LENS_REPO, "lens_file": LENS_FILE, "sha256": got,
               "sha256_matches_pod": True, "mid_sep_recomputed": s["mid_sep"],
               "mid_sep_shipped": ref["mid_sep"], **s}, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
