"""Corpus-dependence figure: seed null versus corpus effect.

Left: fitted boundary positions on each model's depth axis, for two WikiText seeds and code.
The two seeds coincide exactly; code moves.
Right: map distance on a log axis, seed null against corpus, per model.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, json, hashlib, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
POST = Path("/home/ubuntu/PRAX/pre-blog/blog-source/content/posts/2026/07/jlens-cka-397b")
STEM = "corpus-dependence"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Inter","Arial","DejaVu Sans"],
 "figure.facecolor":"#F7F4F0","axes.facecolor":"#F7F4F0","savefig.facecolor":"#F7F4F0",
 "text.color":"#2C2924","axes.edgecolor":"#A89B8C","xtick.color":"#5A544C","ytick.color":"#5A544C",
 "axes.labelcolor":"#2C2924","svg.hashsalt":"prax-corpus"})
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

WIKI, CODE = "#4B6787", "#B5544B"
ORDER = ["gpt2-small", "gemma-3-270m", "qwen3.5-0.8b"]


def build():
    d = json.load(open(HERE / "results.json"))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})

    for row, slug in enumerate(ORDER):
        r = d[slug]; L = r["n_layers"]; y = len(ORDER) - 1 - row
        ax.plot([0, L - 1], [y, y], color="#D9D2C8", lw=7, solid_capstyle="round", zorder=1)
        wa, wb, co = r["boundaries"]["wiki_a"], r["boundaries"]["wiki_b"], r["boundaries"]["code"]
        for b in wa:
            ax.plot([b], [y + 0.13], marker="v", ms=11, color=WIKI, zorder=3)
        for b in wb:   # second seed, drawn slightly offset so coincidence is visible
            ax.plot([b], [y + 0.24], marker="v", ms=8, color=WIKI, alpha=0.55, zorder=3)
        for b in co:
            ax.plot([b], [y - 0.16], marker="^", ms=11, color=CODE, zorder=3)
        shift = r["corpus"]["boundary_shift"]
        ax.text(L + 0.4, y, f"{slug}\n{L} layers, shift {shift}", fontsize=8, va="center", color="#5A544C")
    ax.set_yticks([]); ax.set_xlabel("layer index", fontsize=9.5)
    ax.set_xlim(-0.5, 30); ax.set_ylim(-0.6, len(ORDER) - 0.2)
    ax.set_title("Fitted block boundaries: two WikiText seeds coincide, code does not",
                 fontsize=10.6, fontweight="bold", loc="left")
    ax.plot([], [], marker="v", ls="", color=WIKI, label="WikiText (two seeds)")
    ax.plot([], [], marker="^", ls="", color=CODE, label="code")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)

    x = np.arange(len(ORDER)); w = 0.36
    seed = [d[s]["seed_null"]["map_distance"] for s in ORDER]
    corp = [d[s]["corpus"]["map_distance"] for s in ORDER]
    ax2.bar(x - w/2, seed, w, color="#B0A79A", label="different seed, same corpus")
    ax2.bar(x + w/2, corp, w, color=CODE, label="different corpus")
    for i, (a_, b_) in enumerate(zip(seed, corp)):
        ax2.text(i + w/2, b_ * 1.15, f"{b_/a_:.0f}x", ha="center", fontsize=9,
                 fontweight="bold", color=CODE)
    ax2.set_yscale("log"); ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace("-", "-\n", 1) for s in ORDER], fontsize=8.4)
    ax2.set_ylabel("map distance  (1 - CKA between maps)", fontsize=9)
    ax2.set_title("Corpus moves the map ~100x more", fontsize=10.6,
                  fontweight="bold", loc="left")
    ax2.legend(fontsize=8, frameon=False, loc="lower left")
    ax2.set_ylim(top=max(corp)*3.5)
    for s in ("top", "right"): ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(POST / f"{STEM}.svg", format="svg", metadata={"Date": None})
    fig.savefig(POST / f"{STEM}.png", dpi=170); plt.close(fig)

    alt = ("Two panels. Left: depth axes for three models with fitted block boundaries marked. "
           "The two WikiText seeds place boundaries on identical layers in every model; the "
           "code-fitted boundaries sit elsewhere, moving 10 layers in gemma-3-270m and 15 in "
           "qwen3.5-0.8b, and not at all in gpt2-small. Right: log-scale bars of map distance, "
           "showing the corpus effect exceeding the seed null by 292x, 198x and 87x.")
    (POST / f"{STEM}.receipt.json").write_text(json.dumps({
        "figure_id": STEM,
        "title": "Corpus dependence of fitted J-lens depth boundaries",
        "alt_text": alt, "description": alt,
        "data_source": [{"receipt": "corpus_dependence/results.json", "sha256": sha(HERE / "results.json")}],
        "provenance": {"generator": "corpus_dependence/build_fig.py",
                       "svg_sha256": sha(POST / f"{STEM}.svg")},
        "interval_semantics": "descriptive; seed null is a single same-corpus refit per model",
        "plotted_values": {s: {"boundaries": d[s]["boundaries"], "band_sep": d[s]["band_sep"],
                               "seed_map_distance": d[s]["seed_null"]["map_distance"],
                               "corpus_map_distance": d[s]["corpus"]["map_distance"]} for s in ORDER},
        "accessibility": {"color_only_channel": False, "text_equivalent": "plotted_values"},
    }, indent=1))
    for s in ORDER:
        print(f"  {s:14s} seed={d[s]['seed_null']['map_distance']:.4f} "
              f"corpus={d[s]['corpus']['map_distance']:.4f} shift={d[s]['corpus']['boundary_shift']}")
    print("wrote", POST / f"{STEM}.svg")


def verify():
    rec = json.loads((POST / f"{STEM}.receipt.json").read_text())
    got, want = sha(POST / f"{STEM}.svg"), rec["provenance"]["svg_sha256"]
    print("VERIFY", "OK" if got == want else "MISMATCH")
    if got != want: raise SystemExit("svg mismatch")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    verify() if a.verify else build()
