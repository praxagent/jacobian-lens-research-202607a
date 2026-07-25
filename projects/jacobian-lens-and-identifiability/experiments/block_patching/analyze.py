"""Frozen analysis for the block-patching pre-registration.

E(i,j) ~ per-distance dummies + beta * crosses(i,j).  beta is the boundary effect with layer
distance fully absorbed. Significance comes from a random-3-segmentation null with the SAME
block-size multiset (1,000 draws), which also controls for any smooth positional trend.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ATLAS = HERE.parents[1] / "experiments/jspace_atlas"
sys.path.insert(0, str(ATLAS))


def boundaries(slug):
    from atlas_stage_a import fitted_seg
    M = np.load(ATLAS / f"atlas_out/shared_maps/{slug}.npz", allow_pickle=True)["cka"]
    b1, b2, sep = fitted_seg(M)
    return int(b1), int(b2), float(sep), M.shape[0]


def lens_source_layers(slug, n_lens):
    """Model-layer indices the lens covers. The lens has no Jacobian for the final layer, so
    a model with L blocks yields an L-1 row CKA map; the block boundaries are in LENS index
    space and must be mapped back before they can be applied to a patching matrix over MODEL
    layers. Falls back to the contiguous prefix if the lens cannot be fetched."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        import torch
        fs = [f for f in list_repo_files("neuronpedia/jacobian-lens")
              if f.startswith(slug + "/") and f.endswith(".pt")]
        sl = torch.load(hf_hub_download("neuronpedia/jacobian-lens", sorted(fs)[0]),
                        map_location="cpu", weights_only=False)["source_layers"]
        assert len(sl) == n_lens, f"lens rows {len(sl)} != cka rows {n_lens}"
        return [int(x) for x in sl]
    except Exception as e:
        print(f"  (lens source_layers unavailable: {type(e).__name__}; assuming prefix 0..{n_lens-1})")
        return list(range(n_lens))


def beta_for(E, blk):
    """OLS of E on per-distance dummies + crosses; returns beta (coef on crosses)."""
    L = E.shape[0]
    rows, y = [], []
    for i in range(L):
        for j in range(L):
            if i == j or not np.isfinite(E[i, j]):
                continue
            rows.append((abs(i - j), 1.0 if blk[i] != blk[j] else 0.0)); y.append(E[i, j])
    y = np.array(y); dists = sorted({r[0] for r in rows})
    X = np.zeros((len(rows), len(dists) + 1))
    for k, (d, c) in enumerate(rows):
        X[k, dists.index(d)] = 1.0; X[k, -1] = c
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[-1]), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--nperm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    r = json.load(open(a.result)); E_full = np.array(r["E"], float); slug = r["slug"]
    b1, b2, sep, L = boundaries(slug)
    # map lens index space -> model layer space, then restrict the patching matrix to the
    # layers the lens actually covers (the final model layer has no Jacobian)
    src = lens_source_layers(slug, L)
    assert max(src) < r["n_layers"], f"lens layer {max(src)} beyond model ({r['n_layers']})"
    E = E_full[np.ix_(src, src)]
    print(f"  lens covers model layers {src[0]}..{src[-1]} of {r['n_layers']}; "
          f"patching matrix restricted to {E.shape[0]}x{E.shape[1]}")
    sizes = [b1, b2 - b1, L - b2]

    blk = np.array([0 if i < b1 else (1 if i < b2 else 2) for i in range(L)])
    beta, n = beta_for(E, blk)

    # random-3-segmentation null with the SAME block-size multiset
    rng = np.random.default_rng(a.seed)
    null = []
    for _ in range(a.nperm):
        perm = list(rng.permutation(sizes)); cuts = np.cumsum(perm)[:2]
        rb = np.array([0 if i < cuts[0] else (1 if i < cuts[1] else 2) for i in range(L)])
        # random contiguous relabeling: shift the cut positions too
        shift = rng.integers(0, L)
        rb = np.roll(rb, shift)
        null.append(beta_for(E, rb)[0])
    null = np.array(null)
    p = float((np.abs(null) >= abs(beta)).mean())

    sp = r["self_patch_median"]
    gate = sp >= 0.9
    print(f"model {slug}  L={L}  fitted boundaries {b1},{b2}  block sizes {sizes}  fitted_sep={sep:+.3f}")
    print(f"SANITY self-patch median E(i,i) = {sp:.3f}  -> gate {'PASS' if gate else 'FAIL (verdict VOID)'}")
    print(f"n pairs used = {n}")
    print(f"beta (cross-boundary effect, distance absorbed) = {beta:+.4f}")
    print(f"random-boundary null: mean {null.mean():+.4f}  sd {null.std():.4f}  "
          f"2-sided p = {p:.4f}  (nperm={a.nperm})")
    if not gate:
        verdict = "VOID (sanity gate failed)"
    elif beta < 0 and p < 0.05:
        verdict = "P1 SUPPORTED in this model (cross-boundary patches transfer worse)"
    else:
        verdict = "P1 NOT SUPPORTED in this model (no boundary discontinuity beyond the null)"
    print("VERDICT:", verdict)
    out = Path(a.result).with_name(Path(a.result).stem + "_analysis.json")
    out.write_text(json.dumps({"slug": slug, "b1": b1, "b2": b2, "block_sizes": sizes,
                               "fitted_sep": sep, "beta": beta, "p_perm": p,
                               "null_mean": float(null.mean()), "null_sd": float(null.std()),
                               "n_pairs": n, "self_patch_median": sp, "gate_pass": bool(gate),
                               "verdict": verdict, "nperm": a.nperm}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
