"""Activation-vs-lens dissociation figure for the Gemma-mechanism finding.

For each of gemma-2-9b, gemma-2-2b (flat lens) and qwen3-4b (structured control),
put the RAW-ACTIVATION layer x layer CKA next to the SHARED-VOCABULARY LENS CKA.
Same statistic (linear CKA, fixed-thirds band separation), two different objects:
the residual stream itself vs the lens's readout geometry. The Gemma rows show
structured activations next to a flat lens; the qwen row is structured in both.

Usage:
    python build_actcka_figs.py            # render + write receipt
    python build_actcka_figs.py --verify   # check the SVG matches the receipt sha
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, json, hashlib, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACT = HERE.parent / "actcka_out"
LENS = HERE.parent / "atlas_out" / "shared_maps"
POST = Path("/home/ubuntu/PRAX/pre-blog/blog-source/content/posts/2026/07/jlens-cka-397b")
STEM = "gemma-activation-vs-lens"

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Inter","Arial","DejaVu Sans"],
 "figure.facecolor":"#F7F4F0","axes.facecolor":"#F7F4F0","savefig.facecolor":"#F7F4F0",
 "text.color":"#2C2924","axes.edgecolor":"#A89B8C","xtick.color":"#5A544C","ytick.color":"#5A544C",
 "axes.labelcolor":"#2C2924","svg.hashsalt":"prax-actcka"})

# rows: (title, activation npz, lens npz)
ROWS = [
    ("gemma-2-9b",  ACT/"gemma2_9b.npz", LENS/"gemma-2-9b.npz"),
    ("gemma-2-2b",  ACT/"gemma2_2b.npz", LENS/"gemma-2-2b.npz"),
    ("qwen3-4b",    ACT/"qwen3_4b.npz",  LENS/"qwen3-4b.npz"),
]

def load(p):
    return np.load(p, allow_pickle=True)["cka"].astype(float)

def stats(M):
    """Fixed index-thirds band separation + min off-diagonal, matching activation_cka.py."""
    L = M.shape[0]
    th = np.array_split(np.arange(L), 3)
    def blk(a_, b_):
        v = [M[i, j] for i in a_ for j in b_ if i != j]
        return float(np.mean(v)) if v else 1.0
    e, mid, la = th
    mid_sep = blk(mid, mid) - 0.5*(blk(e, mid) + blk(mid, la))
    off = M[~np.eye(L, dtype=bool)]
    return {"mid_sep": mid_sep, "min": float(off.min()), "median": float(np.median(off))}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build():
    fig, axes = plt.subplots(3, 2, figsize=(7.6, 11.3))
    plotted = {}
    for r, (name, actp, lensp) in enumerate(ROWS):
        for c, (kind, path) in enumerate([("raw activations", actp), ("lens readout geometry", lensp)]):
            M = load(path); s = stats(M); ax = axes[r, c]
            im = ax.imshow(M, cmap="magma", vmin=0.0, vmax=1.0, origin="upper", aspect="equal")
            ax.set_title(f"{name} — {kind}", fontsize=9.6, fontweight="bold", loc="left", pad=6)
            ax.text(0.5, -0.14, f"band sep {s['mid_sep']:+.3f}   min CKA {s['min']:.3f}",
                    transform=ax.transAxes, ha="center", fontsize=8.2, color="#5A544C")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor("#A89B8C")
            plotted[f"{name}|{kind}"] = s
    fig.suptitle("Gemma's flat lens sits on a structured residual stream",
                 fontsize=13.2, fontweight="bold", x=0.5, y=0.975)
    fig.text(0.5, 0.945, "Left column: CKA of the raw hidden states.   Right column: CKA of the shared-vocabulary lens readout.",
             ha="center", fontsize=8.6, color="#5A544C")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.915, bottom=0.115, hspace=0.30, wspace=0.06)
    cax = fig.add_axes([0.30, 0.048, 0.40, 0.012])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("layer x layer linear CKA  (0 = orthogonal, 1 = identical)", fontsize=8.2)
    cbar.ax.tick_params(labelsize=7.3)
    POST.mkdir(parents=True, exist_ok=True)
    fig.savefig(POST/f"{STEM}.svg", format="svg", metadata={"Date": None})
    fig.savefig(POST/f"{STEM}.png", dpi=170)
    plt.close(fig)

    alt = ("Six CKA heatmaps in three rows. Each row is one model: left panel is the raw-activation "
           "layer-by-layer CKA, right panel is the shared-vocabulary lens CKA. gemma-2-9b and gemma-2-2b "
           "show clear block structure on the left (band separation about +0.11 and +0.08, minimum CKA "
           "near 0.07) but a near-uniform bright map on the right (band separation near +0.006). qwen3-4b, "
           "the control, is structured in both panels (activation band separation +0.264).")
    (POST/f"{STEM}.receipt.json").write_text(json.dumps({
        "figure_id": STEM,
        "title": "Gemma activation CKA vs lens CKA (activation-vs-lens dissociation)",
        "alt_text": alt, "description": alt,
        "data_source": [{"receipt": f"jspace_atlas/actcka_out/{p.name}", "sha256": sha(p)}
                        for _, a, l in ROWS for p in (a, l)],
        "provenance": {"generator": "jspace_atlas/figures/build_actcka_figs.py",
                       "svg_sha256": sha(POST/f"{STEM}.svg")},
        "interval_semantics": "descriptive statistics on one CKA map each; no resampling",
        "plotted_values": plotted,
        "accessibility": {"color_only_channel": False, "text_equivalent": "plotted_values"},
    }, indent=1))
    print("wrote", POST/f"{STEM}.svg")
    for k, v in plotted.items():
        print(f"  {k:34s} mid_sep={v['mid_sep']:+.4f} min={v['min']:.4f} median={v['median']:.4f}")


def verify():
    rec = json.loads((POST/f"{STEM}.receipt.json").read_text())
    got = sha(POST/f"{STEM}.svg")
    want = rec["provenance"]["svg_sha256"]
    print("VERIFY", "OK" if got == want else "MISMATCH", f"svg_sha256={got}")
    if got != want:
        raise SystemExit("svg does not match receipt; re-run without --verify")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    verify() if a.verify else build()
