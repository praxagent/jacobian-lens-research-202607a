"""Multi-seed robustness sweep for the linear identifiability core.

Addresses the single-seed fragility flagged in adversarial review: a single seed-0
point estimate is not a result. This runs K seeds per cell (sparse/dense x L1 on/off),
reports mean +/- 95% CI, and tests the load-bearing INTERACTION — does the L1 penalty
raise recovery MORE when the true mixing is structurally sparse than when it is dense?
That interaction (not the raw 0.741) is the identifiability claim. CPU, free.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import synth               # noqa: E402
from train import run_linear           # noqa: E402
from scipy import stats as S           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--l1", type=float, default=0.05)
    ap.add_argument("--out", default=str(Path(__file__).parent / "seed_sweep_linear.json"))
    a = ap.parse_args()

    cells = {"sparse_L1": [], "sparse_none": [], "dense_L1": [], "dense_none": []}
    keymap = {("sparse", a.l1): "sparse_L1", ("sparse", 0.0): "sparse_none",
              ("dense", a.l1): "dense_L1", ("dense", 0.0): "dense_none"}
    for seed in range(a.seeds):
        torch.manual_seed(seed)
        for support in ("sparse", "dense"):
            for l1 in (a.l1, 0.0):
                d = synth.make_linear(a.samples, a.n, support, "gaussian", seed)
                cells[keymap[(support, l1)]].append(float(run_linear(d["x"], d["s"], l1, "cpu")))
        print(f"seed {seed} done", flush=True)

    def ci(v):
        v = np.array(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v))
        return float(m), float(m - 1.96 * se), float(m + 1.96 * se), float(v.std(ddof=1))

    res = {}
    for k, v in cells.items():
        m, lo, hi, sd = ci(v); res[k] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "std": sd, "n": len(v), "values": v}

    sL = np.array(cells["sparse_L1"]); sN = np.array(cells["sparse_none"])
    dL = np.array(cells["dense_L1"]); dN = np.array(cells["dense_none"])
    gain_sparse = sL - sN; gain_dense = dL - dN
    interaction = gain_sparse - gain_dense          # >0 => L1 helps more WITH structure
    w = S.wilcoxon(gain_sparse, gain_dense)
    sparse_is_max = np.mean([sL[i] >= max(sN[i], dL[i], dN[i]) for i in range(len(sL))])
    res["interaction"] = {
        "gain_sparse_mean": float(gain_sparse.mean()), "gain_dense_mean": float(gain_dense.mean()),
        "interaction_mean": float(interaction.mean()),
        "interaction_frac_positive": float((interaction > 0).mean()),
        "sparse_L1_is_max_frac": float(sparse_is_max),
        "wilcoxon_p": float(w.pvalue),
    }
    res["config"] = {"seeds": a.seeds, "n": a.n, "samples": a.samples, "l1": a.l1,
                     "chance": float(1.0 / np.sqrt(a.n))}
    json.dump(res, open(a.out, "w"), indent=2)

    print("\n=== MULTI-SEED SUMMARY (K=%d, n=%d) ===" % (a.seeds, a.n))
    for k in ("sparse_L1", "sparse_none", "dense_L1", "dense_none"):
        r = res[k]; print(f"  {k:12s} {r['mean']:.3f}  95%CI [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  (sd {r['std']:.3f})")
    print(f"  chance       {res['config']['chance']:.3f}")
    it = res["interaction"]
    print(f"  L1 gain: sparse {it['gain_sparse_mean']:+.3f} vs dense {it['gain_dense_mean']:+.3f}  |  interaction {it['interaction_mean']:+.3f}")
    print(f"  interaction>0 in {it['interaction_frac_positive']*100:.0f}% of seeds; "
          f"sparse+L1 is the max cell in {it['sparse_L1_is_max_frac']*100:.0f}% of seeds; "
          f"Wilcoxon p={it['wilcoxon_p']:.4f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
