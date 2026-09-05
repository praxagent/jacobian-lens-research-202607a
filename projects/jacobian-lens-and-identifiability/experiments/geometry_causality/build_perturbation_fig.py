"""The equal-norm perturbation result with its controls on the same axes: per layer, the next-token KL
produced by the lens direction, the mean of the equal-norm random directions, and the input-specific
comparator, from the float32 receipts at the comparator-calibrated dose. --verify asserts byte identity."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
POST = Path("/home/ubuntu/PRAX/pre-blog/blog-source/content/posts/2026/07/jlens-cka-397b")
STEM = "perturbation-arms"
FILES = {"gpt2-small": "out/gpt2-small_C32.json", "gemma-3-270m": "out/gemma_ladder.json", "qwen3.5-0.8b": "out/qwen3.5-0.8b_C32.json"}
CI_LABEL = {"gpt2-small": "gpt2-small C32 (local-calibrated dose)", "gemma-3-270m": "gemma-3-270m ladder (extended, float32)",
            "qwen3.5-0.8b": "qwen3.5-0.8b C32 (local-calibrated dose, 2026-09-05 pod re-run)"}
plt.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif", "font.sans-serif": ["Inter", "Arial", "DejaVu Sans"],
                     "figure.facecolor": "#F7F4F0", "axes.facecolor": "#F7F4F0", "savefig.facecolor": "#F7F4F0",
                     "text.color": "#2C2924", "axes.edgecolor": "#A89B8C", "xtick.color": "#5A544C",
                     "ytick.color": "#5A544C", "axes.labelcolor": "#2C2924", "svg.hashsalt": "prax-perturb"})
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arms(d):
    pl = d["per_layer"]; layers = sorted(pl, key=int)
    A = np.array([np.mean(pl[l]["kl_aligned"]) for l in layers]); Lc = np.array([np.mean(pl[l]["kl_local"]) for l in layers])
    R = np.array([np.mean(np.asarray(pl[l]["kl_random"], float)) for l in layers])
    return [int(l) for l in layers], A, R, Lc
def build(verify=False):
    ci = json.loads((HERE / "out/analysis_fp32_ci.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), constrained_layout=True)
    plotted = {}
    for ax, (slug, f) in zip(axes, FILES.items()):
        d = json.loads((HERE / f).read_text()); layers, A, R, Lc = arms(d)
        ax.plot(layers, Lc, marker="s", ms=3.5, lw=1.1, color="#6F8D5E", label="input-specific comparator")
        ax.plot(layers, A, marker="o", ms=4, lw=1.4, color="#4B6787", label="lens direction")
        ax.plot(layers, R, marker="^", ms=3.5, lw=1.1, color="#B5544B", label="equal-norm random (mean of 8)")
        ax.set_yscale("log"); ax.set_xlabel("layer", fontsize=8.5); ax.set_ylabel("next-token KL (nats)", fontsize=8.5)
        c = ci[CI_LABEL[slug]]
        ax.set_title(f"{slug}\nlens / random {c['ratio']:.1f}x [{c['ratio_ci95'][0]:.1f}, {c['ratio_ci95'][1]:.1f}]   lens / comparator {c['median_C']:.3f}",
                     fontsize=8.6, fontweight="bold", loc="left")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        plotted[slug] = {"layers": layers, "kl_lens": A.tolist(), "kl_random_mean": R.tolist(), "kl_comparator": Lc.tolist(),
                         "ratio": c["ratio"], "ratio_ci95": c["ratio_ci95"], "C": c["median_C"], "dtype": c["dtype"], "n_prompts": c["n_prompts"], "receipt": f}
    axes[0].legend(fontsize=7.6, frameon=False, loc="lower right")
    fig.suptitle("Three arms at identical norm and layer, float32, comparator-calibrated dose, 200 held-out prompts", fontsize=10.4, fontweight="bold", x=0.01, ha="left")
    svg = POST / f"{STEM}.svg"; old = svg.read_bytes() if (verify and svg.exists()) else None
    fig.savefig(svg, format="svg", metadata={"Date": None}); fig.savefig(POST / f"{STEM}.png", dpi=160); plt.close(fig)
    if verify and old is not None and old != svg.read_bytes():
        svg.write_bytes(old); sys.exit("VERIFY FAILED: svg drifted")
    alt = ("Three panels, one per model, next-token KL on a log axis against layer for three perturbation arms at identical norm: "
           "the lens direction (blue circles), the mean of eight equal-norm random directions (red triangles) and the input-specific "
           "comparator (green squares). In every panel the lens sits well above random and well below the comparator at every layer. " +
           " ".join(f"{s}: lens over random {v['ratio']:.1f}x with 95% interval [{v['ratio_ci95'][0]:.1f}, {v['ratio_ci95'][1]:.1f}], lens over comparator {v['C']:.3f}." for s, v in plotted.items()))
    (POST / f"{STEM}.receipt.json").write_text(json.dumps({
        "figure_id": STEM, "title": "Equal-norm perturbation: the three arms per layer", "alt_text": alt, "description": alt,
        "data_source": [{"receipt": f"geometry_causality/{f}", "sha256": sha(HERE / f)} for f in FILES.values()] + [{"receipt": "geometry_causality/out/analysis_fp32_ci.json", "sha256": sha(HERE / "out/analysis_fp32_ci.json")}],
        "provenance": {"generator": "geometry_causality/build_perturbation_fig.py", "svg_sha256": sha(svg), "prereg": "geometry_causality/PREREG.md"},
        "interval_semantics": "ratios: 2,000-resample bootstrap over prompts of the median-over-layers log ratio; curves: per-layer means over prompts",
        "plotted_values": plotted, "accessibility": {"color_only_channel": False, "text_equivalent": "plotted_values"}}, indent=1))
    print(("VERIFY OK " if verify else "built ") + STEM)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); build(ap.parse_args().verify)
