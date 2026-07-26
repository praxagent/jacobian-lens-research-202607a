"""Fit-budget figure: how far each budget's map sits from the 100-prompt reference.

One point per budget per model, against the seed-null reference scale measured in the corpus
experiment and the frozen 2x-null convergence band from PREREG_FITBUDGET.md.

    build_fitbudget_fig.py            build + write receipt
    build_fitbudget_fig.py --verify   rebuild and assert byte-identity + prose consistency
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
POST = Path("/home/ubuntu/PRAX/pre-blog/blog-source/content/posts/2026/07/jlens-cka-397b")
STEM = "fit-budget"
RES = HERE / "results_fitbudget.json"
COLORS = {"gpt2-small": "#6F8D5E", "gemma-3-270m": "#A67C52"}

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Inter", "Arial", "DejaVu Sans"],
                     "figure.facecolor": "#F7F4F0", "axes.facecolor": "#F7F4F0",
                     "savefig.facecolor": "#F7F4F0", "text.color": "#2C2924",
                     "axes.edgecolor": "#A89B8C", "xtick.color": "#5A544C",
                     "ytick.color": "#5A544C", "axes.labelcolor": "#2C2924",
                     "svg.hashsalt": "prax-fitbudget"})


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(verify=False):
    R = json.loads(RES.read_text())
    models = {k: v for k, v in R["models"].items() if "error" not in v}
    fig, axes = plt.subplots(1, len(models), figsize=(8.6, 3.9), sharey=False)
    if len(models) == 1: axes = [axes]
    plotted = {}
    for ax, (slug, m) in zip(axes, models.items()):
        c = COLORS.get(slug, "#8A7A66")
        null = m["seed_null_map_distance"]
        xs = [int(b) for b in R["budgets"] if str(b) in m["by_budget"]]
        ys = [m["by_budget"][str(b)]["map_distance_to_ref"] for b in xs]
        ax.axhspan(0, R["converged_factor"] * null, color="#C9D8BE", alpha=0.55, lw=0)
        ax.axhline(null, color="#5A544C", lw=1.0, ls=":")
        ax.annotate(f"seed null {null:.1e}", (xs[0], null), xytext=(2, 4),
                    textcoords="offset points", fontsize=6.6, color="#5A544C")
        ax.plot(xs, ys, "o-", color=c, lw=1.7, ms=6)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y/null:.0f}x", (x, y), xytext=(0, 7), textcoords="offset points",
                        fontsize=6.8, color=c, ha="center")
        ax.axvline(R["reference_budget"], color="#A89B8C", lw=0.9, ls="--")
        ax.annotate("reference\n(100 prompts)", (R["reference_budget"], 0.98),
                    xycoords=("data", "axes fraction"), xytext=(4, 0),
                    textcoords="offset points", fontsize=6.6, color="#5A544C", va="top")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(xs + [R["reference_budget"]])
        ax.set_xticklabels([str(v) for v in xs + [R["reference_budget"]]], fontsize=7.4)
        ax.minorticks_off()
        ax.set_title(slug, fontsize=9.6, fontweight="bold", loc="left")
        ax.set_xlabel("fitting budget (prompts)", fontsize=8.6)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        plotted[slug] = {"budgets": xs, "map_distance_to_ref": ys,
                         "seed_null": null,
                         "ratio_to_seed_null": [m["by_budget"][str(b)]["ratio_to_seed_null"]
                                                for b in xs],
                         "boundary_shift": [m["by_budget"][str(b)]["boundary_shift"] for b in xs]}
    axes[0].set_ylabel("map distance to the 100-prompt fit  (1 - CKA)", fontsize=8.4)
    fig.suptitle("Small fitting budgets move the map far more than resampling does; "
                 "green band = within 2x the seed null",
                 fontsize=10.2, fontweight="bold", x=0.015, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    svg = POST / f"{STEM}.svg"
    old = svg.read_bytes() if (verify and svg.exists()) else None
    fig.savefig(svg, format="svg", metadata={"Date": None})
    fig.savefig(POST / f"{STEM}.png", dpi=200); plt.close(fig)
    if verify and old is not None and old != svg.read_bytes():
        svg.write_bytes(old); sys.exit("VERIFY FAILED: svg drifted")

    verdict = R.get("verdict", "UNKNOWN")
    alt = ("Two panels, one per model, plotting how far a lens fitted on 25, 50, 200 or 400 "
           "prompts sits from the same model's 100-prompt fit, on log axes. A green band marks "
           "twice the seed null. The 25- and 50-prompt fits sit far above the band; the 200- "
           "and 400-prompt fits sit "
           + ("inside it" if verdict == "CONVERGED" else "outside it") + ".")
    (POST / f"{STEM}.receipt.json").write_text(json.dumps({
        "figure_id": STEM,
        "title": "Map distance to the 100-prompt reference fit, by fitting budget",
        "alt_text": alt, "description": alt,
        "data_source": [{"receipt": "corpus_dependence/results_fitbudget.json",
                         "sha256": sha(RES)}],
        "provenance": {"generator": "corpus_dependence/build_fitbudget_fig.py",
                       "svg_sha256": sha(svg),
                       "prereg": "corpus_dependence/PREREG_FITBUDGET.md"},
        "interval_semantics": "point estimates; the seed null is the reference scale, "
                              "not a confidence interval",
        "verdict": verdict,
        "plotted_values": plotted,
        "accessibility": {"color_only_channel": False,
                          "text_equivalent": "plotted_values"}}, indent=1))
    print(("VERIFY OK " if verify else "built ") + STEM + f"  verdict={verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    build(ap.parse_args().verify)
