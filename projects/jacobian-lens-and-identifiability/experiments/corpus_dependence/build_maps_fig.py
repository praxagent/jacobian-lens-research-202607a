"""WikiText-vs-code map pair per model: what 'the map moves 84x to 888x the seed null' looks like.
Reads maps/<slug>.npz written by analyze.py (linear CKA, shared probe). --verify asserts byte identity."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
POST = Path("/home/ubuntu/PRAX/pre-blog/blog-source/content/posts/2026/07/jlens-cka-397b")
STEM = "corpus-maps"
ORDER = ["gpt2-small", "gemma-3-270m", "qwen3.5-0.8b", "llama3.1-8b"]
ORDER = [m for m in ORDER if m in json.load(open(Path(__file__).resolve().parent / "results.json"))]  # only models with results
plt.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif", "font.sans-serif": ["Inter", "Arial", "DejaVu Sans"],
                     "figure.facecolor": "#F7F4F0", "axes.facecolor": "#F7F4F0", "savefig.facecolor": "#F7F4F0",
                     "text.color": "#2C2924", "axes.edgecolor": "#A89B8C", "xtick.color": "#5A544C",
                     "ytick.color": "#5A544C", "axes.labelcolor": "#2C2924", "svg.hashsalt": "prax-corpus-maps"})
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(verify=False):
    R = json.loads((HERE / "results.json").read_text())
    fig, axes = plt.subplots(len(ORDER), 3, figsize=(10.2, 3.3 * len(ORDER)), constrained_layout=True)
    plotted = {}
    for r, slug in enumerate(ORDER):
        z = np.load(HERE / "maps" / f"{slug}.npz"); A, B = z["wiki_a"], z["code"]; D = B - A
        lim = float(np.abs(D).max()) or 1e-6
        axes[r, 0].imshow(A, origin="lower", cmap="magma", vmin=0, vmax=1)
        axes[r, 1].imshow(B, origin="lower", cmap="magma", vmin=0, vmax=1)
        im = axes[r, 2].imshow(D, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        d = R[slug]["corpus"]["map_distance"]; n = R[slug]["seed_null"]["map_distance"]
        ba, bc = R[slug]["boundaries"]["wiki_a"], R[slug]["boundaries"]["code"]
        axes[r, 0].set_title(f"{slug}\nWikiText fit, boundaries {tuple(ba)}", fontsize=8.6, fontweight="bold", loc="left")
        axes[r, 1].set_title(f"\ncode fit, boundaries {tuple(bc)}", fontsize=8.6, fontweight="bold", loc="left")
        axes[r, 2].set_title(f"code minus WikiText\nmap distance {d:.3f} ({d / n:.0f}x the seed null)", fontsize=8.6, fontweight="bold", loc="left")
        for ax, bb in ((axes[r, 0], ba), (axes[r, 1], bc)):
            for b in bb:
                ax.axhline(b - 0.5, color="white", lw=0.8, ls="--", alpha=0.8); ax.axvline(b - 0.5, color="white", lw=0.8, ls="--", alpha=0.8)
        for ax in axes[r]:
            ax.set_xlabel("source layer", fontsize=8); ax.set_ylabel("source layer", fontsize=8)
        fig.colorbar(im, ax=axes[r, 2], fraction=0.05, pad=0.02, shrink=0.9)
        plotted[slug] = {"map_distance_corpus": d, "map_distance_seed": n, "boundaries_wiki_a": ba, "boundaries_code": bc,
                         "max_abs_cell_change": lim, "offdiag_range_wiki_a": [float(A[~np.eye(len(A), dtype=bool)].min()), float(A[~np.eye(len(A), dtype=bool)].max())],
                         "offdiag_range_code": [float(B[~np.eye(len(B), dtype=bool)].min()), float(B[~np.eye(len(B), dtype=bool)].max())]}
    fig.suptitle("Same model, same recipe, different fitting corpus: the map changes, the fitted boundaries barely do", fontsize=10.6, fontweight="bold", x=0.01, ha="left")
    svg = POST / f"{STEM}.svg"; old = svg.read_bytes() if (verify and svg.exists()) else None
    fig.savefig(svg, format="svg", metadata={"Date": None}); fig.savefig(POST / f"{STEM}.png", dpi=150); plt.close(fig)
    if verify and old is not None and old != svg.read_bytes():
        svg.write_bytes(old); sys.exit("VERIFY FAILED: svg drifted")
    alt = (f"{len(ORDER)} rows of three heatmaps, one row per model ({', '.join(ORDER)}). Left: the layer-by-layer "
           "CKA map of the lens fitted on WikiText with its fitted boundaries dashed; middle: the same for the lens fitted on code; "
           "right: the cell-by-cell difference on a red-blue scale. " + " ".join(
               f"{s}: map distance {v['map_distance_corpus']:.3f} ({v['map_distance_corpus'] / v['map_distance_seed']:.0f} times the seed null), "
               f"boundaries {tuple(v['boundaries_wiki_a'])} on WikiText and {tuple(v['boundaries_code'])} on code, largest single-cell change {v['max_abs_cell_change']:.2f}."
               for s, v in plotted.items()))
    (POST / f"{STEM}.receipt.json").write_text(json.dumps({
        "figure_id": STEM, "title": "WikiText versus code lens maps, three models", "alt_text": alt, "description": alt,
        "data_source": [{"receipt": f"corpus_dependence/maps/{s}.npz", "sha256": sha(HERE / "maps" / f"{s}.npz")} for s in ORDER] + [{"receipt": "corpus_dependence/results.json", "sha256": sha(HERE / "results.json")}],
        "provenance": {"generator": "corpus_dependence/build_maps_fig.py", "svg_sha256": sha(svg)},
        "interval_semantics": "descriptive; one fit per arm", "plotted_values": plotted,
        "accessibility": {"color_only_channel": False, "text_equivalent": "plotted_values"}}, indent=1))
    print(("VERIFY OK " if verify else "built ") + STEM)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); build(ap.parse_args().verify)
