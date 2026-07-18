"""Layer x layer CKA heatmap for the published 397B lens (the Anthropic-style figure).

The release shipped the DERIVED band statistic (mid_sep) but never persisted the full
CKA matrix behind it. This recomputes the matrix on CPU via the EXACT consumer path of
consumer_check_397b.py (public HF lens hash-verified against the pod original, base
model's lm_head shard, numpy seed-0 n_probe=4096 probe, fp16 storage / fp32 CKA — the
pipeline verified at |delta mid_sep| = 2e-8), saves it as .npz, and plots the heatmap.
Correctness gate: recomputed mid_sep must match the shipped band.json to 1e-4 or the
run fails — a heatmap whose matrix disagrees with the published statistic never ships.

--null adds a Frobenius-matched random-J control matrix (the confound check: how much
layer x layer structure comes from the shared unembedding alone).

Run (CPU, hours-scale):  .venv/bin/python cka_heatmap_397b.py --null
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(HERE))
from consumer_check_397b import (  # noqa: E402
    BASE_MODEL, LENS_REPO, LENS_FILE, POD_SHA256, load_unembedding_partial, sha256)

REF_BAND = HERE.parents[1] / "artifacts" / "lenses-397b" / "qwen35_397b_dm.band.json"
OUT = HERE.parents[1] / "artifacts" / "lenses-397b"
N_PROBE, SEED = 4096, 0


def cka_matrix(geom, layers):
    L = len(layers)
    M = np.eye(L)
    for i in range(L):
        gi = geom[layers[i]].astype(np.float32)
        for j in range(i + 1, L):
            M[i, j] = M[j, i] = linear_cka(gi, geom[layers[j]].astype(np.float32))
        print(f"  row {i + 1}/{L}", flush=True)
    return M


def plot(M, layers, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    fig.patch.set_facecolor("#F7F4F0")
    im = ax.imshow(M, vmin=0, vmax=1, cmap="magma", origin="lower",
                   extent=(layers[0], layers[-1], layers[0], layers[-1]))
    ax.set_xlabel("source layer", fontsize=9)
    ax.set_ylabel("source layer", fontsize=9)
    ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("linear CKA of J-lens token geometry (D_l = U @ J_l)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(Path(path).with_suffix(".svg"), format="svg", metadata={"Date": None})
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", action="store_true", help="also compute random-J control")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    lens_path = hf_hub_download(LENS_REPO, LENS_FILE)
    assert sha256(lens_path) == POD_SHA256, "public lens hash mismatch"

    U = load_unembedding_partial(BASE_MODEL)
    vocab, d_model = U.shape
    rng = np.random.default_rng(SEED)
    probe = rng.choice(vocab, size=min(N_PROBE, vocab), replace=False)
    Up = U[probe].float()          # same indexing/order as consumer_check/fit_at_scale
    del U

    d = torch.load(lens_path, map_location="cpu", weights_only=False)
    layers = d["source_layers"]
    print(f"lens: {len(layers)} source layers, d_model {d['d_model']}", flush=True)
    geom, geom_null = {}, {}
    gnull = torch.Generator().manual_seed(SEED)
    for l in layers:
        J = d["J"][l].float()
        geom[l] = (Up @ J).to(torch.float16).numpy()
        if a.null:
            R = torch.randn(J.shape, generator=gnull)
            geom_null[l] = (Up @ (R * (J.norm() / R.norm()))).to(torch.float16).numpy()
        del d["J"][l]
    del d

    print("[real matrix]", flush=True)
    M = cka_matrix(geom, layers)
    s = band_stats(M, layers)
    ref = json.load(open(REF_BAND))
    delta = abs(s["mid_sep"] - ref["mid_sep"])
    print(f"mid_sep recomputed {s['mid_sep']:+.6f} vs shipped {ref['mid_sep']:+.6f} "
          f"(|delta| {delta:.2e})", flush=True)
    assert delta < 1e-4, "matrix disagrees with the published band statistic — DO NOT SHIP"

    out = {"layers": np.array(layers), "cka": M,
           "mid_sep": np.float64(s["mid_sep"]), "n_probe": np.int64(N_PROBE),
           "seed": np.int64(SEED)}
    plot(M, layers, OUT / "cka_397b.png",
         f"Qwen3.5-397B-A17B J-lens: layer x layer CKA (n=24 lens, "
         f"mid_sep {s['mid_sep']:+.3f})")
    if a.null:
        print("[null matrix]", flush=True)
        Mn = cka_matrix(geom_null, layers)
        sn = band_stats(Mn, layers)
        out["cka_null"] = Mn
        out["mid_sep_null"] = np.float64(sn["mid_sep"])
        print(f"null mid_sep {sn['mid_sep']:+.6f}", flush=True)
        plot(Mn, layers, OUT / "cka_397b_null.png",
             f"Random-J control (mid_sep {sn['mid_sep']:+.3f}): structure-free")
    np.savez_compressed(OUT / "cka_397b.npz", **out)
    print(f"CKA_HEATMAP_DONE -> {OUT}/cka_397b.npz + png/svg", flush=True)


if __name__ == "__main__":
    main()
