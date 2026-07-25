"""Frozen analysis: per-model bootstrap over PROMPTS (2000 resamples) of median-over-layers A_l.
P1 requires the one-sided CI to exclude 0 in every model."""
import json, glob, os, numpy as np

def boot(d, n=2000, seed=0):
    pl = d["per_layer"]; layers = sorted(pl, key=int)
    A = np.array([pl[l]["kl_aligned"] for l in layers])            # (L, P)
    R = np.array([pl[l]["kl_random"] for l in layers]).mean(1)     # (L, P) mean over 8 draws
    Lc = np.array([pl[l]["kl_local"] for l in layers])
    P = A.shape[1]; rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, P, P)
        a = A[:, idx].mean(1); r = R[:, idx].mean(1)
        stats.append(np.median(np.log(a / np.maximum(r, 1e-30))))
    stats = np.array(stats)
    obs = np.median(np.log(A.mean(1) / np.maximum(R.mean(1), 1e-30)))
    C = np.median(A.mean(1) / np.maximum(Lc.mean(1), 1e-30))
    return obs, np.percentile(stats, [2.5, 97.5]), C, A, R, Lc

print(f"{'model':14s} {'median A':>9s} {'95% CI':>20s} {'ratio':>7s} {'median C':>9s}  P1")
res = {}
for f in sorted(glob.glob("out/*.json")):
    if "smoke" in f: continue
    d = json.load(open(f)); obs, ci, C, A, R, Lc = boot(d)
    ok = ci[0] > 0
    res[d["slug"]] = {"median_A": float(obs), "ci95": [float(ci[0]), float(ci[1])],
                      "ratio": float(np.exp(obs)), "median_C": float(C),
                      "gates": d["gates"], "p1_supported": bool(ok)}
    print(f"{d['slug']:14s} {obs:+9.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] {np.exp(obs):7.2f}x {C:9.4f}  {'YES' if ok else 'NO'}")

# interrogate gemma's C ~ 1: is aligned literally the same as local?
d = json.load(open("out/gemma-3-270m.json")); pl = d["per_layer"]
print("\ngemma-3-270m: per-prompt correlation between aligned and local KL, by layer")
for l in list(sorted(pl, key=int))[:6]:
    a = np.array(pl[l]["kl_aligned"]); lo = np.array(pl[l]["kl_local"])
    print(f"  layer {l:>3s}: corr={np.corrcoef(a,lo)[0,1]:.4f}  mean ratio={np.mean(a/np.maximum(lo,1e-30)):.4f}  "
          f"aligned={a.mean():.3e} local={lo.mean():.3e}")
json.dump(res, open("out/analysis.json","w"), indent=1)
print("\nALL P1 SUPPORTED:", all(v["p1_supported"] for v in res.values()))
