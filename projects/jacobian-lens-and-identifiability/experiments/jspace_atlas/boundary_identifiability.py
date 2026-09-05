"""How well identified are the fitted three-segmentation boundaries of a layer x layer CKA map?

The atlas fits two boundaries (b1, b2) by maximising the mean off-diagonal within-block CKA summed
over the three blocks (`atlas_stage_a.fitted_seg`). When that objective is nearly flat, many
segmentations are almost as good as the optimum and "the boundary" is a near-arbitrary pick among
them. Any downstream test that uses the boundary as a variable (Test B, ignition depth, corpus
boundary shift) inherits that arbitrariness.

Measure (frozen in bands_vs_computation/PREREG_B3_IDENT.md, 2026-09-05): score every legal
segmentation; the NEAR-OPTIMAL SET is every segmentation within `tol` of the (best - worst) range
below the best; the SPREAD of a boundary is (max - min) of that boundary over the near-optimal set
divided by L. A boundary pair is IDENTIFIED when both spreads are <= 0.25 at tol = 0.05.

    boundary_identifiability.py            compute for all shared + own atlas maps, write JSON/CSV
    boundary_identifiability.py --fig      also build the note figure + receipt
    boundary_identifiability.py --verify   rebuild the figure and assert byte identity

numpy-only so it can be imported by the Test B analyzer.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "atlas_out"
SHARED = OUT / "shared_maps"
POST = Path("/home/ubuntu/PRAX/pre-blog/blog-source/content/posts/2026/07/jlens-cka-397b")
STEM = "boundary-identifiability"
TOL, TOL_SENS = 0.05, (0.15, 0.35)
SPREAD_MAX = 0.25


def segmentation_scores(M):
    """{(b1, b2): score} for every legal 3-segmentation, same objective as fitted_seg."""
    L = M.shape[0]; S = M.cumsum(0).cumsum(1)

    def block_sum(a, b):
        t = S[b - 1, b - 1]
        if a > 0:
            t = t - S[a - 1, b - 1] - S[b - 1, a - 1] + S[a - 1, a - 1]
        return t
    out = {}
    for b1 in range(2, L - 3):
        for b2 in range(b1 + 2, L - 1):
            s = 0.0
            for a, b in ((0, b1), (b1, b2), (b2, L)):
                n = b - a
                s += (block_sum(a, b) - n) / max(n * n - n, 1)
            out[(b1, b2)] = s
    return out


def near_optimal_spread(M, tol=TOL):
    """(best_seg, n_near, spread_b1, spread_b2, objective_range) or None if L is too small."""
    sc = segmentation_scores(M)
    if not sc:
        return None
    L = M.shape[0]
    keys = list(sc.keys()); v = np.array([sc[k] for k in keys])
    best, worst = v.max(), v.min(); rng = best - worst
    near = [keys[i] for i in range(len(keys)) if v[i] >= best - tol * rng]
    b1 = [k[0] for k in near]; b2 = [k[1] for k in near]
    return (keys[int(v.argmax())], len(near), (max(b1) - min(b1)) / L, (max(b2) - min(b2)) / L,
            float(rng))


def identified(M, tol=TOL, spread_max=SPREAD_MAX):
    r = near_optimal_spread(M, tol)
    return bool(r is not None and r[2] <= spread_max and r[3] <= spread_max)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compute():
    rows = {}
    for f in sorted(SHARED.glob("*.npz")):
        slug = f.stem
        M = np.load(f)["cka"]; L = int(M.shape[0])
        rec = {"n_layers": L}
        for label, path in (("shared", f), ("own", OUT / f"{slug}.npz")):
            if not path.exists():
                continue
            Mx = np.load(path)["cka"]
            r = near_optimal_spread(Mx, TOL)
            if r is None:
                rec[label] = {"too_small": True}; continue
            seg, n, s1, s2, rng = r
            d = {"fitted_seg": [int(seg[0]), int(seg[1])], "objective_range": rng,
                 "n_near_optimal": n, "spread_b1": s1, "spread_b2": s2,
                 "identified": bool(s1 <= SPREAD_MAX and s2 <= SPREAD_MAX)}
            for t in TOL_SENS:
                rs = near_optimal_spread(Mx, t)
                d[f"spread_b1_tol{t}"] = rs[2]; d[f"spread_b2_tol{t}"] = rs[3]
                d[f"identified_tol{t}"] = bool(rs[2] <= SPREAD_MAX and rs[3] <= SPREAD_MAX)
            rec[label] = d
        rows[slug] = rec
    out = {"tol": TOL, "tol_sensitivity": list(TOL_SENS), "spread_max": SPREAD_MAX,
           "rule": "identified iff both near-optimal boundary spreads <= spread_max at tol",
           "prereg": "bands_vs_computation/PREREG_B3_IDENT.md", "models": rows}
    (OUT / "boundary_identifiability.json").write_text(json.dumps(out, indent=1))
    with open(OUT / "boundary_identifiability.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slug", "n_layers", "probe", "b1", "b2", "objective_range", "n_near_optimal",
                    "spread_b1", "spread_b2", "identified"])
        for slug, rec in rows.items():
            for probe in ("shared", "own"):
                d = rec.get(probe)
                if not d or d.get("too_small"):
                    continue
                w.writerow([slug, rec["n_layers"], probe, *d["fitted_seg"],
                            f"{d['objective_range']:.4f}", d["n_near_optimal"],
                            f"{d['spread_b1']:.3f}", f"{d['spread_b2']:.3f}", int(d["identified"])])
    n_sh = [r for r in rows.values() if "shared" in r and not r["shared"].get("too_small")]
    n_id = sum(1 for r in n_sh if r["shared"]["identified"])
    print(f"shared probe: {n_id}/{len(n_sh)} models identified at tol {TOL}, spread <= {SPREAD_MAX}")
    for slug, rec in sorted(rows.items(), key=lambda kv: -max(kv[1].get("shared", {}).get("spread_b1", 0),
                                                              kv[1].get("shared", {}).get("spread_b2", 0))):
        d = rec.get("shared", {})
        if d.get("too_small") or not d:
            continue
        print(f"  {slug:18s} L={rec['n_layers']:3d} seg={tuple(d['fitted_seg'])!s:>9s} "
              f"spread b1={d['spread_b1']:.2f} b2={d['spread_b2']:.2f} "
              f"{'IDENTIFIED' if d['identified'] else 'not identified'}")
    return out


def build_fig(out, verify=False):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Inter", "Arial", "DejaVu Sans"],
                         "figure.facecolor": "#F7F4F0", "axes.facecolor": "#F7F4F0",
                         "savefig.facecolor": "#F7F4F0", "text.color": "#2C2924",
                         "axes.edgecolor": "#A89B8C", "xtick.color": "#5A544C",
                         "ytick.color": "#5A544C", "axes.labelcolor": "#2C2924",
                         "svg.hashsalt": "prax-ident"})
    rows = [(s, r["shared"]) for s, r in out["models"].items()
            if "shared" in r and not r["shared"].get("too_small")]
    rows.sort(key=lambda kv: max(kv[1]["spread_b1"], kv[1]["spread_b2"]))
    names = [s for s, _ in rows]
    s1 = [d["spread_b1"] for _, d in rows]; s2 = [d["spread_b2"] for _, d in rows]
    ident = [d["identified"] for _, d in rows]
    fig, ax = plt.subplots(figsize=(8.6, 0.27 * len(rows) + 1.6))
    y = np.arange(len(rows)); h = 0.38
    ax.barh(y + h / 2, s1, h, color="#4B6787", label="early boundary (b1)")
    ax.barh(y - h / 2, s2, h, color="#B5544B", label="late boundary (b2)")
    ax.axvline(out["spread_max"], color="#5A544C", lw=1.0, ls=":")
    ax.text(out["spread_max"] + 0.01, len(rows) - 0.4, f"identified if both <= {out['spread_max']}",
            fontsize=7.6, color="#5A544C", va="top")
    ax.set_yticks(y); ax.set_yticklabels([n + ("" if i else "  *") for n, i in zip(names, ident)],
                                         fontsize=7.6)
    ax.set_xlabel(f"spread of near-optimal boundary positions (fraction of depth), "
                  f"segmentations within {int(out['tol']*100)}% of the objective range", fontsize=8.4)
    n_id = sum(ident)
    ax.set_title(f"Fitted boundaries are identified for {n_id} of {len(rows)} lenses on the shared probe "
                 f"(* = not identified)", fontsize=10.2, fontweight="bold", loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_xlim(0, max(max(s1), max(s2)) * 1.08)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    svg = POST / f"{STEM}.svg"
    old = svg.read_bytes() if (verify and svg.exists()) else None
    fig.savefig(svg, format="svg", metadata={"Date": None})
    fig.savefig(POST / f"{STEM}.png", dpi=170); plt.close(fig)
    if verify and old is not None and old != svg.read_bytes():
        svg.write_bytes(old); raise SystemExit("VERIFY FAILED: svg drifted")
    worst = names[-1]
    alt = (f"Horizontal bar chart, one row per lens, {len(rows)} lenses sorted from best to worst "
           f"identified. Each row has two bars, the spread of the early and of the late boundary "
           f"over the near-optimal segmentations, as a fraction of depth. A dotted line at "
           f"{out['spread_max']} marks the identification threshold. {n_id} of {len(rows)} lenses "
           f"have both bars inside the line; {len(rows) - n_id} do not, the worst being {worst} "
           f"with spreads {rows[-1][1]['spread_b1']:.2f} and {rows[-1][1]['spread_b2']:.2f}. "
           f"The 397B lens (qwen35-397b-own) has spreads "
           f"{out['models']['qwen35-397b-own']['shared']['spread_b1']:.2f} and "
           f"{out['models']['qwen35-397b-own']['shared']['spread_b2']:.2f}.")
    (POST / f"{STEM}.receipt.json").write_text(json.dumps({
        "figure_id": STEM,
        "title": "Identifiability of the fitted three-segmentation boundaries, shared probe",
        "alt_text": alt, "description": alt,
        "data_source": [{"receipt": "jspace_atlas/atlas_out/boundary_identifiability.json",
                         "sha256": sha(OUT / "boundary_identifiability.json")}],
        "provenance": {"generator": "jspace_atlas/boundary_identifiability.py",
                       "svg_sha256": sha(svg), "prereg": out["prereg"]},
        "interval_semantics": "descriptive; spreads are deterministic functions of each cached map",
        "plotted_values": {s: {"spread_b1": d["spread_b1"], "spread_b2": d["spread_b2"],
                               "identified": d["identified"], "fitted_seg": d["fitted_seg"]}
                           for s, d in rows},
        "accessibility": {"color_only_channel": False, "text_equivalent": "plotted_values"}},
        indent=1))
    print(("VERIFY OK " if verify else "built ") + STEM + f"  identified {n_id}/{len(rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", action="store_true"); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        out = json.loads((OUT / "boundary_identifiability.json").read_text())
        build_fig(out, verify=True)
    else:
        out = compute()
        if a.fig:
            build_fig(out)
