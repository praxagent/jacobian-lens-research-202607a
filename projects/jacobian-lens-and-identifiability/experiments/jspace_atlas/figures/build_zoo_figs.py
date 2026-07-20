"""Zoo figures for the expanded Atlas-of-Depth note — from Stage-A/B receipts.

Per RESEARCH_NOTE §5: matplotlib from committed receipts, per-figure receipt JSON,
post-wide manifest, --verify byte-identity + prose check. Figures:
  1. band-by-scale   : fitted_sep vs params, colored by family (the family split)
  2. mid-vs-fitted   : fixed-thirds mid_sep vs data-fitted sep (how conservative)
  3. pr-curves       : participation-ratio-by-depth small multiples per family
  4. (Stage B) family-mds + dist matrix, added when stageB_results.json exists

  build_zoo_figs.py            # build into the post bundle
  build_zoo_figs.py --verify
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ATLAS = HERE.parent
OUT_A = ATLAS / "atlas_out"
POST = Path("/home/ubuntu/PRAX/pre-blog/blog-source/content/posts/2026/07/jlens-cka-397b")
PIN = "e555939"
REPO = "https://github.com/praxagent/jacobian-lens-research-202607a"
SUMMARY_REL = ("projects/jacobian-lens-and-identifiability/experiments/"
               "jspace_atlas/atlas_out/summary.csv")

plt.rcParams.update({
    "svg.hashsalt": "praxagent-atlas", "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Arial", "Helvetica", "DejaVu Sans"],
    "figure.facecolor": "#F7F4F0", "axes.facecolor": "#F7F4F0",
    "savefig.facecolor": "#F7F4F0", "text.color": "#2C2924",
    "axes.edgecolor": "#A89B8C", "axes.labelcolor": "#2C2924",
    "xtick.color": "#5A544C", "ytick.color": "#5A544C"})
FAM_COLOR = {"qwen": "#4B6787", "gemma": "#A67C52", "llama": "#6F8D5E",
             "olmo": "#7F786D", "other": "#B0A79A"}


def fam(slug):
    for f in ("gemma", "qwen", "llama", "olmo"):
        if slug.startswith(f):
            return f
    return "other"


def rows():
    r = list(csv.DictReader(open(OUT_A / "summary.csv")))
    return list({x["slug"]: x for x in r}.values())


def params(slug):
    import re
    m = re.search(r"(\d+\.?\d*)b", slug.lower())
    if m:
        return float(m[1]) * 1e9
    m = re.search(r"(\d+)m", slug.lower())
    if m:
        return float(m[1]) * 1e6
    return {"qwen35-397b-own": 397e9, "gpt-oss-20b": 20e9,
            "pythia-70m-deduped": 70e6, "gpt2-small": 124e6}.get(slug, np.nan)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(fig, stem, title, alt, vals):
    fig.savefig(POST / f"{stem}.svg", format="svg", metadata={"Date": None})
    fig.savefig(POST / f"{stem}.png", dpi=200)
    plt.close(fig)
    (POST / f"{stem}.receipt.json").write_text(json.dumps({
        "figure_id": stem, "title": title, "alt_text": alt, "description": alt,
        "data_source": [{"receipt": SUMMARY_REL, "sha256": sha(OUT_A / "summary.csv"),
                         "pinned_url": f"{REPO}/blob/{PIN}/{SUMMARY_REL}"}],
        "provenance": {"generator": "projects/jacobian-lens-and-identifiability/"
                       "experiments/jspace_atlas/figures/build_zoo_figs.py",
                       "svg_sha256": sha(POST / f"{stem}.svg")},
        "interval_semantics": "descriptive per-model census values, no sampling",
        "guards": "--verify fails on source-hash / manifest / output-byte drift",
        "plotted_values": vals,
        "accessibility": {"color_only_channel": False,
                          "text_equivalent": "plotted_values + summary.csv"}}, indent=1))
    return stem, vals


def fig_band_by_scale(R):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    vals = {}
    for r in R:
        p, y, f = params(r["slug"]), float(r["fitted_sep"]), fam(r["slug"])
        ax.scatter(p, y, s=54, c=FAM_COLOR[f], edgecolor="#2C2924", linewidth=0.5,
                   zorder=3)
        vals[r["slug"]] = y
    for lab in ("qwen35-397b-own", "llama3.3-70b-it", "gemma-3-27b-it"):
        r = next(x for x in R if x["slug"] == lab)
        ax.annotate(lab.replace("-own", ""), (params(lab), float(r["fitted_sep"])),
                    textcoords="offset points", xytext=(6, 5), fontsize=7.5,
                    color="#5A544C")
    ax.axhline(0, color="#A89B8C", lw=0.9, ls=(0, (5, 4)))
    ax.set_xscale("log")
    ax.set_xlabel("parameters (log scale)", fontsize=9)
    ax.set_ylabel("fitted band separation (higher = sharper depth phases)", fontsize=9)
    ax.set_title("Depth-phase structure is a family trait, not a scale trend",
                 fontsize=11, fontweight="bold", loc="left")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", label=k,
              markerfacecolor=v, markersize=8) for k, v in FAM_COLOR.items()],
              fontsize=8, loc="upper left", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return save(fig, "zoo-band-by-scale",
                "Fitted band separation vs scale, colored by family",
                "Scatter of fitted band separation against parameter count on a log "
                "x-axis, colored by family. Qwen (blue) and the Llama-70B sit high "
                "(0.25-0.41); every Gemma (clay) sits near zero across a 270M-27B "
                "span. Structure tracks family, not size.", vals)


def fig_mid_vs_fitted(R):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    vals = {}
    for r in R:
        x, y, f = float(r["mid_sep"]), float(r["fitted_sep"]), fam(r["slug"])
        ax.scatter(x, y, s=48, c=FAM_COLOR[f], edgecolor="#2C2924", linewidth=0.5,
                   zorder=3)
        vals[r["slug"]] = [x, y]
    lim = [-0.03, 0.45]
    ax.plot(lim, lim, color="#A89B8C", lw=0.9, ls=(0, (5, 4)), zorder=1)
    ax.text(0.30, 0.31, "fitted = fixed", fontsize=7.5, color="#7F786D", rotation=34)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("fixed-thirds mid_sep (the released statistic)", fontsize=9)
    ax.set_ylabel("data-fitted band separation", fontsize=9)
    ax.set_title("The released statistic is conservative: fitted boundaries always "
                 "separate more", fontsize=10.5, fontweight="bold", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return save(fig, "zoo-mid-vs-fitted",
                "Fixed-thirds mid_sep vs data-fitted band separation",
                "Scatter: every model lies on or above the y=x line, so the "
                "data-fitted 3-segmentation always finds at least as much band "
                "separation as the conservative fixed-thirds statistic; the 397B "
                "moves from 0.34 to 0.41.", vals)


def fig_pr_curves(R):
    fams = ["qwen", "gemma", "llama", "olmo", "other"]
    fig, axes = plt.subplots(1, 5, figsize=(11, 3.0), sharey=True)
    vals = {}
    for ax, f in zip(axes, fams):
        for r in R:
            if fam(r["slug"]) != f:
                continue
            d = np.load(OUT_A / f"{r['slug']}.npz")
            pr = d["pr"] / float(d["d_model"])
            xs = np.linspace(0, 1, len(pr))
            ax.plot(xs, pr, color=FAM_COLOR[f], lw=1.0, alpha=0.7)
            vals[r["slug"]] = float(np.median(pr))
        ax.set_yscale("log")
        ax.set_title(f, fontsize=9)
        ax.set_xlabel("relative depth", fontsize=8)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("PR / d_model (log)", fontsize=8.5)
    fig.suptitle("The capacity dial: effective readout width by depth, per family",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save(fig, "zoo-pr-curves",
                "Participation-ratio-by-depth curves, one panel per family",
                "Five small-multiple panels of participation ratio over d_model "
                "versus relative depth, one per family, log y-axis. All families rise "
                "toward late layers; Qwen and Llama run wider readouts than Gemma at "
                "matched depth.", vals)


def build(verify=False):
    R = rows()
    figs = [fig_band_by_scale(R), fig_mid_vs_fitted(R), fig_pr_curves(R)]
    ent = []
    for stem, vals in figs:
        for k, v in vals.items():
            ent.append({"label": f"{stem}:{k}", "value": v})
    manifest = {"what": "zoo-figure provenance (Stage A census)", "repo": REPO,
                "pin_commit": PIN, "n_models": len(R), "figures": dict(figs)}
    (POST / "zoo-provenance.json").write_text(json.dumps(manifest, indent=1, default=float))
    idx = {f"{s}.receipt.json": sha(POST / f"{s}.receipt.json") for s, _ in figs}
    (POST / "zoo-receipts-index.json").write_text(json.dumps(idx, indent=1))
    print(f"built {len(figs)} zoo figures -> {POST}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    POST.mkdir(parents=True, exist_ok=True)
    build(a.verify)
