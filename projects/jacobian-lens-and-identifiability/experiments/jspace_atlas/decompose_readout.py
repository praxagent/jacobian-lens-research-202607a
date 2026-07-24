"""#1 Readout decomposition: is Gemma's flat lens a global pass-through, or specific
to the unembedding (readout) subspace?

The lens readout geometry is D_l = U_probe @ J_l (probe tokens x d_model). Its layer x
layer CKA is what the atlas measures, and for gemma-2 it is flat (band-sep ~+0.005).
Algebraically, CKA(U J_i, U J_j) compares the Jacobians *weighted by the unembedding
second moment* M = U^T U (anisotropic, concentrated on output-relevant directions):

    CKA(U J_i, U J_j) = ||J_j^T M J_i||_F^2 / (||J_i^T M J_i||_F ||J_j^T M J_j||_F)

Replace the real unembedding probe U with a RANDOM Gaussian probe R (so M -> ~isotropic,
R^T R ~ n I) and you get the *isotropic* Jacobian comparison: does J_l rotate with depth
in GENERIC directions? Same construction, same shapes, only the probe covariance differs.

  - isotropic band-sep >> readout band-sep  ->  the flatness is SPECIFIC to the readout
    subspace: the Jacobians rotate generically, but are held near-invariant exactly where
    the unembedding covariance concentrates (theory: a readout-collapse mechanism).
  - isotropic band-sep ~ readout band-sep (both flat) -> the first-order map is globally
    pass-through; the readout is not special.

Control: qwen3-4b (structured readout) should be structured isotropically too.

CPU-only, uses cached lens .pt files (no GPU, no model download). Gram-trick CKA:
CKA(X,Y) = <K_X,K_Y>_F / (||K_X||_F ||K_Y||_F) with K = X_c X_c^T (n x n), X column-centered.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "decompose_out"

MODELS = {
    "gemma-2-9b": {"pt": "/home/ubuntu/hf-lenses-tier1/gemma2_9b_wiki.pt", "readout": 0.0047},
    "qwen3-4b":   {"pt": "/home/ubuntu/hf-lenses-tier1/qwen4b_wiki.pt",    "readout": 0.0383},
}
N_PROBE = 2048   # random probe rows / subsample; > enough for a stable Gram, fast on CPU
SEED = 0


def band_sep(M: np.ndarray) -> float:
    """Fixed index-thirds band separation, matching activation_cka.py / the atlas."""
    L = M.shape[0]
    th = np.array_split(np.arange(L), 3)
    def blk(a_, b_):
        v = [M[i, j] for i in a_ for j in b_ if i != j]
        return float(np.mean(v)) if v else 1.0
    e, mid, la = th
    return blk(mid, mid) - 0.5 * (blk(e, mid) + blk(mid, la))


def cka_from_grams(Ks: list[np.ndarray]) -> np.ndarray:
    """Layer x layer linear CKA from precomputed centered Gram matrices."""
    L = len(Ks)
    norm = [np.linalg.norm(K, "fro") for K in Ks]
    M = np.eye(L)
    for i in range(L):
        for j in range(i + 1, L):
            v = float(np.sum(Ks[i] * Ks[j]) / (norm[i] * norm[j])) if norm[i] * norm[j] > 0 else 0.0
            M[i, j] = M[j, i] = v
    return M


def gram(G: np.ndarray) -> np.ndarray:
    """Column-center then n x n Gram (matches linear_cka's column-centering)."""
    Gc = G - G.mean(0, keepdims=True)
    return Gc @ Gc.T


def participation_ratio(G: np.ndarray) -> float:
    """(sum s^2)^2 / sum s^4 on the centered geometry's singular values (effective dim)."""
    Gc = G - G.mean(0, keepdims=True)
    s = np.linalg.svd(Gc, compute_uv=False)
    s2 = s ** 2
    return float((s2.sum() ** 2) / (np.square(s2).sum() + 1e-12))


def run_model(slug: str, cfg: dict) -> dict:
    d = torch.load(cfg["pt"], map_location="cpu", weights_only=False)
    Jd = d["J"]; layers = sorted(Jd.keys()); dim = d["d_model"]
    rng = np.random.default_rng(SEED)
    R = rng.standard_normal((N_PROBE, dim)).astype(np.float32)        # random probe (isotropic)
    ridx = rng.choice(dim, size=min(N_PROBE, dim), replace=False)     # raw-J row subsample

    Krand, Kraw, prs, colnorms = [], [], [], []
    for l in layers:
        J = Jd[l].float().numpy()                                     # d x d
        Grand = R @ J                                                 # (N_PROBE x d) random-probe geometry
        Krand.append(gram(Grand)); prs.append(participation_ratio(Grand))
        Kraw.append(gram(J[ridx, :]))                                 # raw-J: output-dim rows as samples
        colnorms.append(np.linalg.norm(J, axis=0))                    # per-input-dim response magnitude
    Mrand = cka_from_grams(Krand)
    Mraw = cka_from_grams(Kraw)

    # massive-dim diagnostic: are a few input dims dominant AND consistent across layers?
    C = np.stack(colnorms)                                            # (L x d)
    share_top16 = float(np.mean(np.sort(C, 1)[:, -16:].sum(1) / C.sum(1)))
    # across-layer consistency of the column-norm profile (mean pairwise corr)
    Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-9)
    corr = (Cn @ Cn.T) / C.shape[1]
    consistency = float(corr[np.triu_indices_from(corr, 1)].mean())

    return {
        "slug": slug, "n_layers": len(layers), "d_model": dim, "n_probe": N_PROBE,
        "readout_band_sep_Mweighted": cfg["readout"],
        "random_probe_band_sep_isotropic": round(band_sep(Mrand), 4),
        "raw_J_band_sep_isotropic": round(band_sep(Mraw), 4),
        "random_probe_min_offdiag": round(float(Mrand[~np.eye(len(layers), dtype=bool)].min()), 4),
        "random_probe_median_offdiag": round(float(np.median(Mrand[~np.eye(len(layers), dtype=bool)])), 4),
        "mean_participation_ratio": round(float(np.mean(prs)), 2),
        "colnorm_top16_share": round(share_top16, 4),
        "colnorm_across_layer_consistency": round(consistency, 4),
        "_maps": {"random": Mrand, "raw": Mraw},
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    res = {}
    for slug, cfg in MODELS.items():
        r = run_model(slug, cfg)
        maps = r.pop("_maps")
        np.savez_compressed(OUT / f"{slug}_decomp.npz", random=maps["random"], raw=maps["raw"])
        res[slug] = r
        print(f"[{slug}] readout(M-wt)={r['readout_band_sep_Mweighted']:+.4f}  "
              f"random-probe(iso)={r['random_probe_band_sep_isotropic']:+.4f}  "
              f"raw-J(iso)={r['raw_J_band_sep_isotropic']:+.4f}  "
              f"PR={r['mean_participation_ratio']}  top16share={r['colnorm_top16_share']:.3f}  "
              f"colconsist={r['colnorm_across_layer_consistency']:.3f}")
    (OUT / "decompose_results.json").write_text(json.dumps(res, indent=1))
    # headline
    g = res["gemma-2-9b"]; q = res["qwen3-4b"]
    print("\n=== READOUT-SPECIFICITY ===")
    for slug, r in res.items():
        ratio = r["random_probe_band_sep_isotropic"] / max(r["readout_band_sep_Mweighted"], 1e-4)
        print(f"  {slug:12s} isotropic/readout band-sep ratio = {ratio:5.1f}x  "
              f"({'structured isotropically, flat readout -> READOUT-SPECIFIC' if ratio>3 else 'both similar -> global'})")


if __name__ == "__main__":
    main()
