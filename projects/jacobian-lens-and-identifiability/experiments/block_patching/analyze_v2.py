"""Frozen v2 analysis (PREREG_V2.md).

    D(i,j) ~ dummies(|i-j|) + dummies(mean_position) + beta * crosses(i,j)

The mean-position dummies are the fix for v1's confound. Significance from a random
3-segmentation null with the same block-size multiset. P1 predicts beta > 0.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ATLAS = HERE.parents[1] / "experiments/jspace_atlas"
sys.path.insert(0, str(ATLAS))
from analyze import boundaries, lens_source_layers          # noqa: E402  (shared, tested)


def beta_for(D, blk, use_position=True):
    """OLS with distance dummies, optional mean-position dummies, and a crosses indicator."""
    L = D.shape[0]
    dist, pos, cross, y = [], [], [], []
    for i in range(L):
        for j in range(L):
            if i == j or not np.isfinite(D[i, j]):
                continue
            dist.append(abs(i - j)); pos.append(int(round((i + j) / 2)))
            cross.append(1.0 if blk[i] != blk[j] else 0.0); y.append(D[i, j])
    y = np.array(y)
    du, pu = sorted(set(dist)), sorted(set(pos))
    ncol = len(du) + (len(pu) - 1 if use_position else 0) + 1
    X = np.zeros((len(y), ncol))
    for k in range(len(y)):
        X[k, du.index(dist[k])] = 1.0
        if use_position and pos[k] != pu[0]:          # drop one level to avoid collinearity
            X[k, len(du) + pu.index(pos[k]) - 1] = 1.0
        X[k, -1] = cross[k]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[-1]), len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True); ap.add_argument("--nperm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    r = json.load(open(a.result)); Df = np.array(r["D"], float); slug = r["slug"]
    b1, b2, sep, L = boundaries(slug)
    src = lens_source_layers(slug, L)
    D = Df[np.ix_(src, src)]
    sizes = [b1, b2 - b1, L - b2]
    blk = np.array([0 if i < b1 else (1 if i < b2 else 2) for i in range(L)])

    beta, n = beta_for(D, blk)
    beta_nopos, _ = beta_for(D, blk, use_position=False)

    rng = np.random.default_rng(a.seed)
    null = []
    for _ in range(a.nperm):
        perm = list(rng.permutation(sizes)); cuts = np.cumsum(perm)[:2]
        rb = np.array([0 if i < cuts[0] else (1 if i < cuts[1] else 2) for i in range(L)])
        null.append(beta_for(np.roll(np.roll(D, 0, 0), 0, 1), np.roll(rb, rng.integers(0, L)))[0])
    null = np.array(null)
    p = float((np.abs(null) >= abs(beta)).mean())

    gates = {"sanity": r["diag_max_abs"] < 1e-6,
             "calibration": 0.05 <= r["median_D_far"] <= 5.0}
    print(f"model {slug}  L={L}  blocks {sizes}  balance {min(sizes)/max(sizes):.2f}  fitted_sep={sep:+.3f}")
    print(f"GATES  sanity max|D(i,i)|={r['diag_max_abs']:.1e} {'PASS' if gates['sanity'] else 'FAIL'}  |  "
          f"calibration median D={r['median_D_far']:.3f} nats {'PASS' if gates['calibration'] else 'FAIL'}")
    print(f"n pairs = {n}")
    print(f"beta (crosses; distance AND position absorbed) = {beta:+.4f}")
    print(f"  [ref] without position dummies                = {beta_nopos:+.4f}")
    print(f"random-boundary null: mean {null.mean():+.4f} sd {null.std():.4f}  2-sided p = {p:.4f}")
    if not all(gates.values()):
        verdict = "VOID (gate failed)"
    elif beta > 0 and p < 0.05:
        verdict = "CONFIRMED: boundaries mark real format changes"
    elif beta < 0 and p < 0.05:
        verdict = "REVERSED: crossing a boundary damages LESS (report as such)"
    else:
        verdict = "NULL: blocks are geometric, not computational"
    print("VERDICT:", verdict)
    Path(a.result).with_name(Path(a.result).stem + "_analysis.json").write_text(json.dumps(
        {"slug": slug, "blocks": sizes, "balance": min(sizes)/max(sizes), "fitted_sep": sep,
         "beta": beta, "beta_no_position": beta_nopos, "p_perm": p,
         "null_mean": float(null.mean()), "null_sd": float(null.std()), "n_pairs": n,
         "gates": gates, "verdict": verdict, "nperm": a.nperm}, indent=1))
    print("wrote analysis json")


if __name__ == "__main__":
    main()
