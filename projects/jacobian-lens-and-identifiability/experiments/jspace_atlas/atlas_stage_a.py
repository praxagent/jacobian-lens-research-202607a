"""J-space atlas Stage A — per-model quantities (CPU, resumable).

Per model: within-model layer x layer CKA (own-vocab seed-0 probe), per-layer
participation ratio real + scale-matched-random null, fitted 3-segmentation, and the
shared-vocab geometry basis (U rows for the resolved shared strings) for Stage B.
One npz + receipt per slug in atlas_out/; existing outputs are skipped (resume).

  .venv/bin/python atlas_stage_a.py --slug gpt2-small          # one model
  .venv/bin/python atlas_stage_a.py --all                      # the whole zoo
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
JL = HERE.parent / "jacobian_lens"
sys.path.insert(0, str(JL))
sys.path.insert(0, str(HERE.parents[1]))
from common.cka import linear_cka          # noqa: E402
from cka_layers import resolve, band_stats, REPO  # noqa: E402
from unembed import load_unembedding       # noqa: E402
from shared_vocab import resolve_ids       # noqa: E402

OUT = HERE / "atlas_out"
SHARED = json.load(open(JL / "shared_tokens.json"))
N_PROBE, SEED = 4096, 0
OWN_397B = HERE.parents[1] / "artifacts/lenses-397b"


def load_lens(slug):
    """(jacobians {int: fp16 tensor}, layers, hf_id)."""
    if slug == "qwen35-397b-own":
        import glob
        lens = sorted(glob.glob(str(Path.home() /
            ".cache/huggingface/hub/models--praxagent-org--jacobian-lens-qwen3.5-397b-a17b"
            "/snapshots/*/jlens/wikitext/qwen35_397b.pt")))
        if lens:
            path = lens[-1]
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download("praxagent-org/jacobian-lens-qwen3.5-397b-a17b",
                                   "jlens/wikitext/qwen35_397b.pt")
        d = torch.load(path, map_location="cpu", weights_only=False)
        jac = {int(l): J.to(torch.float16) for l, J in d["J"].items()}
        return jac, sorted(jac), "Qwen/Qwen3.5-397B-A17B"
    from huggingface_hub import hf_hub_download
    hf_id, lens_file = resolve(slug)
    ckpt = torch.load(hf_hub_download(REPO, filename=lens_file),
                      map_location="cpu", weights_only=True)
    jac = {int(l): J for l, J in ckpt["J"].items()}
    return jac, sorted(jac), hf_id


def load_U(slug, hf_id):
    if slug == "qwen35-397b-own":
        sys.path.insert(0, str(HERE.parent / "fit_our_own"))
        from consumer_check_397b import load_unembedding_partial
        return load_unembedding_partial(hf_id).float()
    return load_unembedding(hf_id).float()


def pr_of(D):
    s = torch.linalg.svdvals(D.float())
    s2 = s ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def fitted_seg(M):
    """(b1, b2, fitted_sep): 3 contiguous segments maximizing mean within-CKA."""
    L = M.shape[0]
    S = M.cumsum(0).cumsum(1)

    def block_sum(a, b):  # sum of M[a:b, a:b]
        t = S[b - 1, b - 1]
        if a > 0:
            t = t - S[a - 1, b - 1] - S[b - 1, a - 1] + S[a - 1, a - 1]
        return t

    best = (-1e9, 1, 2)
    for b1 in range(2, L - 3):
        for b2 in range(b1 + 2, L - 1):
            score = 0.0
            for a, b in ((0, b1), (b1, b2), (b2, L)):
                n = b - a
                score += (block_sum(a, b) - n) / max(n * n - n, 1)
            if score > best[0]:
                best = (score, b1, b2)
    _, b1, b2 = best
    mask = np.ones((L, L), bool)
    np.fill_diagonal(mask, False)
    seg_id = np.zeros(L, int)
    seg_id[b1:b2] = 1
    seg_id[b2:] = 2
    same = seg_id[:, None] == seg_id[None, :]
    between_mean = float(M[mask & ~same].mean())
    within_mean = float(M[mask & same].mean())
    return b1, b2, within_mean - between_mean


def run_slug(slug):
    out_npz = OUT / f"{slug}.npz"
    if out_npz.exists():
        print(f"[{slug}] exists, skip", flush=True)
        return
    t0 = time.time()
    jac, layers, hf_id = load_lens(slug)
    d_model = jac[layers[0]].shape[0]
    U = load_U(slug, hf_id)
    vocab = U.shape[0]
    rng = np.random.default_rng(SEED)
    probe = rng.choice(vocab, size=min(N_PROBE, vocab), replace=False)
    Up = U[probe].float()
    ids_map = resolve_ids(hf_id, SHARED["strings"])
    shared_strings = [s for s in SHARED["strings"] if s in ids_map]
    U_shared = U[[ids_map[s] for s in shared_strings]].float()
    del U

    gen = torch.Generator().manual_seed(SEED)
    geom, pr, pr_null = {}, [], []
    for l in layers:
        J = jac[l].float()
        D = Up @ J
        geom[l] = D.to(torch.float16)
        pr.append(pr_of(D))
        R = torch.randn(J.shape, generator=gen) * J.std()
        pr_null.append(pr_of(Up @ R))
        del J, D, R
    L = len(layers)
    M = np.eye(L)
    for i in range(L):
        gi = geom[layers[i]].float().numpy()
        for j in range(i + 1, L):
            M[i, j] = M[j, i] = linear_cka(gi, geom[layers[j]].float().numpy())
    b1, b2, fsep = fitted_seg(M)
    bs = band_stats(M, layers)
    np.savez_compressed(out_npz, layers=np.array(layers), cka=M,
                        pr=np.array(pr), pr_null=np.array(pr_null),
                        d_model=np.int64(d_model),
                        seg=np.array([b1, b2]), fitted_sep=np.float64(fsep),
                        mid_sep=np.float64(bs["mid_sep"]),
                        U_shared=U_shared.to(torch.float16).numpy())
    (OUT / f"{slug}.strings.json").write_text(json.dumps(shared_strings))
    row = {"slug": slug, "hf_id": hf_id, "d_model": d_model, "n_layers": L,
           "mid_sep": round(bs["mid_sep"], 4), "fitted_sep": round(fsep, 4),
           "seg_b1": int(b1), "seg_b2": int(b2),
           "pr_over_d_median": round(float(np.median(pr) / d_model), 5),
           "n_shared": len(shared_strings), "seconds": round(time.time() - t0, 1)}
    with open(OUT / "summary.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)
    print(f"[{slug}] L={L} d={d_model} mid_sep={bs['mid_sep']:+.3f} "
          f"fitted_sep={fsep:+.3f} seg=({b1},{b2}) "
          f"prd={np.median(pr)/d_model:.4f} ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.all:
        slugs = [r["slug"] for r in csv.DictReader(open(JL / "emergence.csv"))]
        slugs.append("qwen35-397b-own")
        for s in slugs:
            try:
                run_slug(s)
            except Exception as e:
                print(f"[{s}] FAILED: {type(e).__name__}: {e}", flush=True)
        print("STAGE_A_DONE", flush=True)
    else:
        run_slug(a.slug)


if __name__ == "__main__":
    main()
