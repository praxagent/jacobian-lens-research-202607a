"""Frozen v2.1 analysis (PREREG_V2_1.md): one-sided test of beta_b1 > 0 at the early/middle
boundary, per model, with the same random-3-segmentation null."""
import json, sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from analyze import boundaries, lens_source_layers
from analyze_C import design, per_boundary

def run(path, slug, nperm=400, seed=0):
    r = json.load(open(path)); Df = np.array(r["D"], float)
    b1, b2, sep, L = boundaries(slug); src = lens_source_layers(slug, L)
    D = Df[np.ix_(src, src)]
    obs, null, n = per_boundary(D, b1, b2, L, nperm, seed)
    p1_one = float((null[:, 0] >= obs[0]).mean())      # one-sided, direction frozen a priori
    p1_two = float((np.abs(null[:, 0]) >= abs(obs[0])).mean())
    gates = {"sanity": r["diag_max_abs"] < 1e-6, "calibration": 0.05 <= r["median_D_far"] <= 5.0}
    print(f"{slug}: blocks {[b1,b2-b1,L-b2]} balance {min(b1,b2-b1,L-b2)/max(b1,b2-b1,L-b2):.2f}")
    print(f"  gates sanity={'PASS' if gates['sanity'] else 'FAIL'} calibration={'PASS' if gates['calibration'] else 'FAIL'} (median D {r['median_D_far']:.3f})")
    print(f"  beta_b1 (early<->middle) = {obs[0]:+.4f}  null sd {null[:,0].std():.4f}  p_one={p1_one:.4f}  p_two={p1_two:.4f}")
    print(f"  [secondary] beta_b2 = {obs[1]:+.4f}   beta_both = {obs[2]:+.4f}")
    return {"slug": slug, "beta_b1": float(obs[0]), "beta_b2": float(obs[1]),
            "beta_both": float(obs[2]), "p_one_sided": p1_one, "p_two_sided": p1_two,
            "null_sd": float(null[:,0].std()), "gates": gates, "n_pairs": n}

if __name__ == "__main__":
    res = run(sys.argv[1], sys.argv[2])
    Path(sys.argv[1]).with_name(Path(sys.argv[1]).stem + "_v21analysis.json").write_text(json.dumps(res, indent=1))
    print("wrote analysis")
