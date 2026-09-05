"""Per-model view of Test B (shared-probe re-run): observed lens/activation boundary agreement against the
pre-registered size-permutation null and the uniform-segmentation sensitivity null, with identifiability marks.
Reads results_B2.json, results_B3_ident.json, outB/*.json. --verify asserts byte identity."""
from __future__ import annotations
import argparse, glob, hashlib, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import analyze_B2 as B2  # noqa: E402
POST = Path("/home/ubuntu/PRAX/pre-blog/blog-source/content/posts/2026/07/jlens-cka-397b")
STEM = "testB-per-model"
plt.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif", "font.sans-serif": ["Inter", "Arial", "DejaVu Sans"],
                     "figure.facecolor": "#F7F4F0", "axes.facecolor": "#F7F4F0", "savefig.facecolor": "#F7F4F0",
                     "text.color": "#2C2924", "axes.edgecolor": "#A89B8C", "xtick.color": "#5A544C",
                     "ytick.color": "#5A544C", "axes.labelcolor": "#2C2924", "svg.hashsalt": "prax-testB"})
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def uniform_null(L, act, n=1000, seed=0):
    rng = np.random.default_rng(seed); legal = [(b1, b2) for b1 in range(2, L - 3) for b2 in range(b1 + 2, L - 1)]
    if not legal: return np.array([abs(1 - act[0]) + abs(2 - act[1])] * n, float)
    idx = rng.integers(0, len(legal), n); return np.array([abs(legal[i][0] - act[0]) + abs(legal[i][1] - act[1]) for i in idx], float)
def build(verify=False):
    R = json.loads((HERE / "results_B2.json").read_text()); I = json.loads((HERE / "results_B3_ident.json").read_text())
    rows = []
    for slug, m in R["models"].items():
        raw = m.get("raw");
        if not raw: continue
        L = m["n_layers"]; lb = m["lens_boundaries"]; ab = raw["boundaries"]
        nd = B2.null_agreement(lb[0], lb[1], L, ab, B2.NPERM); nu = uniform_null(L, ab)
        idn = I["models"].get(slug, {}).get("tol0.05", {})
        rows.append({"slug": slug, "L": L, "obs": raw["agreement"] / L, "null_sizes_median": float(np.median(nd)) / L,
                     "null_uniform_q05": float(np.percentile(nu, 5)) / L, "null_uniform_q50": float(np.median(nu)) / L, "null_uniform_q95": float(np.percentile(nu, 95)) / L,
                     "excluded": bool(raw["excluded_new_gate"]), "lens_identified": idn.get("lens_identified"), "act_identified": idn.get("raw", {}).get("act_identified")})
    rows.sort(key=lambda r: r["obs"])
    fig, ax = plt.subplots(figsize=(8.8, 0.36 * len(rows) + 1.8))
    y = np.arange(len(rows))
    for k, r in enumerate(rows):
        ax.plot([r["null_uniform_q05"], r["null_uniform_q95"]], [k, k], color="#D9D2C8", lw=6, solid_capstyle="round", zorder=1)
        ax.plot([r["null_uniform_q50"]], [k], marker="|", ms=12, color="#8A8378", zorder=2)
        ax.plot([r["null_sizes_median"]], [k], marker="D", ms=5, color="#B0603A", zorder=3)
        ax.plot([r["obs"]], [k], marker="o", ms=7, color="#2C2924" if not r["excluded"] else "#A89B8C", zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([f"{r['slug']}{'  (excluded)' if r['excluded'] else ''}{'' if (r['lens_identified'] and r['act_identified']) else '  *'}" for r in rows], fontsize=8)
    ax.set_xlabel("|lens boundaries - activation boundaries| / depth  (lower = closer agreement)", fontsize=8.6)
    ax.plot([], [], marker="o", ls="", color="#2C2924", label="observed"); ax.plot([], [], marker="D", ls="", color="#B0603A", label="pre-registered null median (size permutation)")
    ax.plot([], [], color="#D9D2C8", lw=6, label="uniform-segmentation null, 5th to 95th percentile"); ax.legend(fontsize=7.6, frameon=False, loc="lower right")
    n_beat = sum(1 for r in rows if not r["excluded"] and r["obs"] < r["null_sizes_median"]); n_use = sum(1 for r in rows if not r["excluded"])
    v = R["verdicts"]["raw_new"]
    ax.set_title(f"Test B per model (raw activation map, shared-probe lens boundaries): {n_beat} of {n_use} usable models beat their own null median\n"
                 f"pooled agreement {v['obs_mean_agreement_norm']:.3f} vs null median {v['pooled_null_median']:.3f}, p = {v['p_one_sided']:.2f}; * = a boundary pair not identified",
                 fontsize=9, fontweight="bold", loc="left")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    svg = POST / f"{STEM}.svg"; old = svg.read_bytes() if (verify and svg.exists()) else None
    fig.savefig(svg, format="svg", metadata={"Date": None}); fig.savefig(POST / f"{STEM}.png", dpi=170); plt.close(fig)
    if verify and old is not None and old != svg.read_bytes():
        svg.write_bytes(old); sys.exit("VERIFY FAILED: svg drifted")
    alt = (f"Dot plot with one row per model ({len(rows)} rows). Each row shows the observed distance between the lens boundaries and the "
           f"activation boundaries as a black dot, the pre-registered null median as an orange diamond, and the 5th to 95th percentile of a uniform "
           f"segmentation null as a grey bar. {n_beat} of {n_use} usable models have the dot left of the diamond. Excluded models are grey; rows "
           f"marked with an asterisk have at least one boundary pair that is not identified. Pooled agreement {v['obs_mean_agreement_norm']:.3f} against a null median of {v['pooled_null_median']:.3f}, p = {v['p_one_sided']:.2f}.")
    (POST / f"{STEM}.receipt.json").write_text(json.dumps({
        "figure_id": STEM, "title": "Test B per model, shared probe", "alt_text": alt, "description": alt,
        "data_source": [{"receipt": "bands_vs_computation/results_B2.json", "sha256": sha(HERE / "results_B2.json")}, {"receipt": "bands_vs_computation/results_B3_ident.json", "sha256": sha(HERE / "results_B3_ident.json")}],
        "provenance": {"generator": "bands_vs_computation/build_testB_fig.py", "svg_sha256": sha(svg), "prereg": "bands_vs_computation/PREREG_B2.md"},
        "interval_semantics": "nulls are permutation distributions (1,000 draws each); no confidence intervals on observed values",
        "plotted_values": rows, "accessibility": {"color_only_channel": False, "text_equivalent": "plotted_values"}}, indent=1))
    print(("VERIFY OK " if verify else "built ") + STEM + f"  {n_beat}/{n_use} beat")
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); build(ap.parse_args().verify)
