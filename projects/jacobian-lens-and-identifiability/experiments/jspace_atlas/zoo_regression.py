"""#2 Zoo regression: which architectural / training features predict lens flatness?

Target: readout band-sep (shared-vocab mid_sep) for all 36 atlas lenses. Features encoded
by hand from tech reports (with confidence flags). The cleanest evidence is the WITHIN-MODEL
base->instruct delta (same weights +/- instruct tuning), which removes all cross-model
confounds. We also report group contrasts and a standardized OLS.

Reliable features: instruct (slug suffix), soft_cap (gemma-2 only), family, params, depth.
Lower-confidence: distilled (gemma-2 2b/9b yes, gemma-2-27b no, gemma-3 yes; others unknown).
CPU-only, reads atlas_out/shared_maps/*.npz.
"""
from __future__ import annotations
import json, glob, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SM = HERE / "atlas_out/shared_maps"
OUT = HERE / "decompose_out"

# params in billions (approx), soft_cap (gemma-2 logit soft-capping), distilled (confidence-flagged)
PARAMS = {
    "gemma-2-2b": 2.6, "gemma-2-9b": 9.2, "gemma-2-27b": 27,
    "gemma-3-270m": 0.27, "gemma-3-1b": 1.0, "gemma-3-4b": 4.3, "gemma-3-12b": 12, "gemma-3-27b": 27,
    "gemma-4-e2b": 2.0, "gemma-4-e4b": 4.0,
    "qwen2.5-7b": 7.6, "qwen3-1.7b": 1.7, "qwen3-4b": 4.0, "qwen3-8b": 8.2, "qwen3-14b": 14.8,
    "qwen3.5-0.8b": 0.8, "qwen3.5-2b": 2.0, "qwen3.5-4b": 4.0, "qwen3.5-9b": 9.0,
    "qwen3.5-27b": 27, "qwen35-397b": 397,
    "llama3.1-8b": 8, "llama3.3-70b": 70, "olmo-3-1025-7b": 7, "olmo-3-1125-32b": 32,
    "gpt2-small": 0.124, "pythia-70m-deduped": 0.07, "gpt-oss-20b": 21,
}
# distilled: 1 known-yes, 0 known-no, None unknown (excluded from that feature's stats)
DISTILLED = {
    "gemma-2-2b": 1, "gemma-2-9b": 1, "gemma-2-27b": 0,     # 27b trained from scratch (teacher)
    "gemma-3-270m": 1, "gemma-3-1b": 1, "gemma-3-4b": 1, "gemma-3-12b": 1, "gemma-3-27b": 1,
    "gemma-4-e2b": 1, "gemma-4-e4b": 1,
}


def base_slug(slug: str) -> str:
    for suf in ("-it", "-pt", "-own"):
        if slug.endswith(suf):
            return slug[: -len(suf)]
    return slug


def family(slug):
    for f in ("gemma-4", "gemma-3", "gemma-2", "qwen3.5", "qwen3", "qwen2.5", "llama", "olmo", "gpt2", "pythia", "gpt-oss"):
        if slug.startswith(f):
            return f
    return "other"


def feat(slug):
    core = base_slug(slug)
    return {
        "slug": slug, "family": family(slug),
        "instruct": 1 if slug.endswith("-it") else 0,
        "soft_cap": 1 if slug.startswith("gemma-2") else 0,     # only gemma-2 has logit soft-capping
        "distilled": DISTILLED.get(core),
        "params": PARAMS.get(core),
        "log_params": np.log10(PARAMS.get(core)) if PARAMS.get(core) else None,
    }


def spearman(x, y):
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    xr -= xr.mean(); yr -= yr.mean()
    d = np.sqrt((xr**2).sum() * (yr**2).sum())
    return float((xr*yr).sum()/d) if d > 0 else 0.0


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for f in sorted(glob.glob(str(SM/"*.npz"))):
        slug = os.path.basename(f)[:-4]
        z = np.load(f, allow_pickle=True)
        r = feat(slug); r["mid_sep"] = float(z["mid_sep"]); r["depth"] = int(z["cka"].shape[0])
        rows.append(r)

    y = np.array([r["mid_sep"] for r in rows])
    print(f"{len(rows)} models. mean band-sep {y.mean():+.4f}\n")

    # --- (a) within-model base -> instruct deltas (cleanest test) ---
    by = {r["slug"]: r for r in rows}
    pairs = []
    for r in rows:
        if r["instruct"]:
            b = r["slug"][:-3]
            if b in by:
                pairs.append((b, by[b]["mid_sep"], r["mid_sep"]))
    print("=== (a) base -> instruct readout band-sep (same weights +/- instruct) ===")
    deltas = []
    for b, mb, mi in sorted(pairs):
        deltas.append(mi - mb)
        print(f"  {b:16s} base {mb:+.4f} -> it {mi:+.4f}   delta {mi-mb:+.4f}  {'FLATTER' if mi<mb else 'more structured'}")
    deltas = np.array(deltas)
    print(f"  => {int((deltas<0).sum())}/{len(deltas)} instruct pairs FLATTER; mean delta {deltas.mean():+.4f}\n")

    # --- (b) group contrasts ---
    def mean_of(pred):
        v = [r["mid_sep"] for r in rows if pred(r)]
        return (np.mean(v), len(v)) if v else (float("nan"), 0)
    print("=== (b) group contrasts (mean band-sep) ===")
    g2small = lambda r: r["family"] == "gemma-2" and r["params"] and r["params"] < 15
    print(f"  gemma-2 small (2b/9b, +/-it): {mean_of(g2small)[0]:+.4f}  (n={mean_of(g2small)[1]})")
    print(f"  gemma-2-27b (soft-cap, NOT distilled): {mean_of(lambda r: base_slug(r['slug'])=='gemma-2-27b')[0]:+.4f}")
    print(f"  gemma-3 base (distilled, no soft-cap, no instruct): {mean_of(lambda r: r['family']=='gemma-3' and not r['instruct'])[0]:+.4f}  (n={mean_of(lambda r: r['family']=='gemma-3' and not r['instruct'])[1]})")
    print(f"  instruct (all families): {mean_of(lambda r: r['instruct'])[0]:+.4f}  (n={mean_of(lambda r: r['instruct'])[1]})")
    print(f"  base (all families):     {mean_of(lambda r: not r['instruct'])[0]:+.4f}  (n={mean_of(lambda r: not r['instruct'])[1]})\n")

    # --- (c) univariate Spearman ---
    print("=== (c) Spearman(feature, band-sep) across all 36 ===")
    corrs = {}
    for name, getter in [("instruct", lambda r: r["instruct"]),
                         ("soft_cap", lambda r: r["soft_cap"]),
                         ("log_params", lambda r: r["log_params"]),
                         ("depth", lambda r: r["depth"])]:
        xs = np.array([getter(r) for r in rows], dtype=float)
        rho = spearman(xs, y); corrs[name] = round(rho, 3)
        print(f"  {name:12s} rho = {rho:+.3f}")
    # distilled only over models with a known label
    dl = [(r["distilled"], r["mid_sep"]) for r in rows if r["distilled"] is not None]
    if dl:
        xd = np.array([a for a, _ in dl], float); yd = np.array([b for _, b in dl])
        corrs["distilled_knownonly"] = round(spearman(xd, yd), 3)
        print(f"  distilled(known n={len(dl)}) rho = {spearman(xd, yd):+.3f}")

    # --- (d) standardized OLS on reliable features ---
    print("\n=== (d) standardized OLS: band-sep ~ instruct + soft_cap + log_params + depth ===")
    X = np.column_stack([
        [r["instruct"] for r in rows],
        [r["soft_cap"] for r in rows],
        [r["log_params"] for r in rows],
        [r["depth"] for r in rows],
    ]).astype(float)
    names = ["instruct", "soft_cap", "log_params", "depth"]
    yv = y.copy()
    keep = ~np.isnan(X).any(1)                      # drop rows with any missing feature
    if (~keep).any():
        print(f"  (dropped {int((~keep).sum())} rows with missing features from OLS)")
    X, yv = X[keep], yv[keep]
    Xz = (X - X.mean(0)) / X.std(0)
    Xd = np.column_stack([np.ones(len(yv)), Xz])
    beta, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
    yhat = Xd @ beta
    r2 = 1 - ((yv - yhat)**2).sum() / ((yv - yv.mean())**2).sum()
    coefs = {"intercept": round(float(beta[0]), 4)}
    for n, b in zip(names, beta[1:]):
        coefs[n] = round(float(b), 4)
        print(f"  {n:12s} std-beta {b:+.4f}")
    print(f"  R^2 = {r2:.3f}")

    out = {"n": len(rows),
           "base_to_instruct": {"n_pairs": len(deltas), "n_flatter": int((deltas<0).sum()),
                                 "mean_delta": round(float(deltas.mean()), 4),
                                 "pairs": [{"model": b, "base": mb, "it": mi} for b, mb, mi in sorted(pairs)]},
           "spearman": corrs,
           "ols_std_betas": coefs, "ols_r2": round(float(r2), 3),
           "rows": [{k: r[k] for k in ("slug","family","instruct","soft_cap","distilled","params","depth","mid_sep")} for r in rows]}
    (OUT / "zoo_regression.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT/'zoo_regression.json'}")


if __name__ == "__main__":
    main()
