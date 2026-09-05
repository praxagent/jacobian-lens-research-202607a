"""Fit-budget figure: budget effects against the seed null AND against the corpus effect.

The point of the figure is the contrast in magnitude. Refitting the same model on the same
corpus at 25 to 400 prompts moves the map by about as much as simply resampling the corpus does;
refitting it on code moves the map by two orders of magnitude more. Log y, one panel per model,
so the two effects are visible on one axis.

    build_fitbudget_fig.py            build + write receipt
    build_fitbudget_fig.py --verify   rebuild and assert byte-identity
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
CORPUS = HERE / "results.json"
C_BUDGET, C_NULL, C_CORPUS = "#6F8D5E", "#8A8378", "#B0603A"

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Inter", "Arial", "DejaVu Sans"],
                     "figure.facecolor": "#F7F4F0", "axes.facecolor": "#F7F4F0",
                     "savefig.facecolor": "#F7F4F0", "text.color": "#2C2924",
                     "axes.edgecolor": "#A89B8C", "xtick.color": "#5A544C",
                     "ytick.color": "#5A544C", "axes.labelcolor": "#2C2924",
                     "svg.hashsalt": "prax-fitbudget"})


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(verify=False):
    R = json.loads(RES.read_text()); CO = json.loads(CORPUS.read_text())
    models = {k: v for k, v in R["models"].items() if "error" not in v}
    fig, axes = plt.subplots(1, len(models), figsize=(8.8, 4.3))
    if len(models) == 1: axes = [axes]
    plotted = {}

    for ax, (slug, m) in zip(axes, models.items()):
        null = m["seed_null_map_distance"]
        corpus_d = CO[slug]["corpus"]["map_distance"]
        budgets = [int(b) for b in R["budgets"] if str(b) in m["by_budget"]]
        dists = [m["by_budget"][str(b)]["map_distance_to_ref"] for b in budgets]

        # x positions: one slot per budget, then the two reference bars on the right
        xs = list(range(len(budgets)))
        x_null, x_corp = len(budgets) + 0.6, len(budgets) + 1.6
        ax.bar(xs, dists, width=0.62, color=C_BUDGET)
        ax.bar([x_null], [null], width=0.62, color=C_NULL)
        ax.bar([x_corp], [corpus_d], width=0.62, color=C_CORPUS)
        ax.axhline(2 * null, color="#5A544C", lw=1.0, ls=":")
        # right-aligned: the region above the line is empty there, while the leftmost bars
        # sit right against it and would collide with their own value labels
        ax.annotate("2x seed null (the frozen convergence bar)", (x_corp + 0.35, 2 * null),
                    xytext=(0, 4), textcoords="offset points", fontsize=6.6,
                    color="#5A544C", ha="right")
        for x, v in list(zip(xs, dists)) + [(x_null, null), (x_corp, corpus_d)]:
            ax.annotate(f"{v:.1e}", (x, v), xytext=(0, 3), textcoords="offset points",
                        fontsize=6.5, color="#2C2924", ha="center")
        ax.annotate(f"{corpus_d/null:.0f}x the null", (x_corp, corpus_d), xytext=(0, 15),
                    textcoords="offset points", fontsize=7.4, color=C_CORPUS,
                    ha="center", fontweight="bold")

        ax.set_yscale("log")
        ax.set_xticks(xs + [x_null, x_corp])
        ax.set_xticklabels([f"n={b}" for b in budgets] + ["resample\n(seed null)", "code\ncorpus"],
                           fontsize=7.2)
        ax.set_ylim(min(dists + [null]) / 3, corpus_d * 6)
        ax.set_title(slug, fontsize=9.8, fontweight="bold", loc="left")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        plotted[slug] = {"budgets": budgets, "map_distance_to_ref": dists,
                         "seed_null": null, "corpus_map_distance": corpus_d,
                         "ratio_to_seed_null": [m["by_budget"][str(b)]["ratio_to_seed_null"]
                                                for b in budgets],
                         "boundary_shift_vs_reference":
                             [m["by_budget"][str(b)]["boundary_shift"] for b in budgets],
                         "corpus_boundary_shift": CO[slug]["corpus"]["boundary_shift"]}

    axes[0].set_ylabel("map distance to the 100-prompt fit  (1 - CKA, log scale)", fontsize=8.2)
    allb = sorted({b for v in plotted.values() for b in v["budgets"]})
    # Title and alt text are DERIVED from the data: which budgets clear the frozen 2x bar.
    over = {k: [b for b, r in zip(v["budgets"], v["ratio_to_seed_null"]) if r > 2.0]
            for k, v in plotted.items()}
    first_ok = {}
    for k, v in plotted.items():
        ok = [b for b, r in zip(v["budgets"], v["ratio_to_seed_null"]) if r <= 2.0]
        first_ok[k] = min(ok) if ok else None
    conv_from = max(b for b in first_ok.values() if b is not None) if any(first_ok.values()) else None
    fig.suptitle((f"Budget is inside the seed null from {conv_from} prompts up; the corpus is not"
                  if conv_from else "Budget never enters the seed null; the corpus is far outside it"),
                 fontsize=11, fontweight="bold", x=0.015, ha="left")
    fig.text(0.015, 0.90, f"Green: same corpus, {allb[0]} to {allb[-1]} prompts. Grey: the same "
             "corpus resampled. Orange: the same budget on code.",
             fontsize=7.8, color="#5A544C", ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    svg = POST / f"{STEM}.svg"
    old = svg.read_bytes() if (verify and svg.exists()) else None
    fig.savefig(svg, format="svg", metadata={"Date": None})
    fig.savefig(POST / f"{STEM}.png", dpi=200); plt.close(fig)
    if verify and old is not None and old != svg.read_bytes():
        svg.write_bytes(old); sys.exit("VERIFY FAILED: svg drifted")

    blist = ", ".join(str(b) for b in allb[:-1]) + f" and {allb[-1]}"
    over_phrase = "; ".join(
        (f"for {k} the bars at {', '.join(str(b) for b in bs)} prompts sit above the line"
         if bs else f"for {k} every bar sits at or below the line") for k, bs in over.items())
    alt = ("Two panels, gpt2-small and gemma-3-270m, each a log-scale bar chart of how far a "
           f"refitted lens sits from that model's 100-prompt reference fit. {len(allb)} green "
           f"bars for budgets of {blist} prompts are compared with a dotted line marking twice "
           f"the seed null: {over_phrase}. A grey bar shows simply resampling the corpus. A single "
           "orange bar for the same model fitted on code sits far above all of them, at "
           + " and ".join(f"{v['corpus_map_distance']/v['seed_null']:.0f} times the null for {k}"
                          for k, v in plotted.items()) + ".")
    (POST / f"{STEM}.receipt.json").write_text(json.dumps({
        "figure_id": STEM,
        "title": "Fitting budget versus fitting corpus, measured against the same seed null",
        "alt_text": alt, "description": alt,
        "data_source": [{"receipt": "corpus_dependence/results_fitbudget.json",
                         "sha256": sha(RES)},
                        {"receipt": "corpus_dependence/results.json", "sha256": sha(CORPUS)}],
        "provenance": {"generator": "corpus_dependence/build_fitbudget_fig.py",
                       "svg_sha256": sha(svg),
                       "prereg": "corpus_dependence/PREREG_FITBUDGET.md"},
        "interval_semantics": "point estimates; the seed null is the reference scale, "
                              "not a confidence interval",
        "verdict": R.get("verdict", "UNKNOWN"),
        "low_budget_caveat": R.get("low_budget_caveat"),
        "statistic": CO.get("_statistic", "unrecorded"),
        "plotted_values": plotted,
        "accessibility": {"color_only_channel": False,
                          "text_equivalent": "plotted_values"}}, indent=1))
    print(("VERIFY OK " if verify else "built ") + STEM + f"  verdict={R.get('verdict')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    build(ap.parse_args().verify)
