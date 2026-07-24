"""Figure for the closing result: readout concentration explains the residual flatness.

Left: band separation as a function of how many top readout eigendirections are ablated.
Right: readout participation ratio vs measured band separation (monotone relation).
Reads decompose_out/readout_ablation_r2048.json.  --verify checks the svg sha.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, json, hashlib, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEC = HERE.parent / "decompose_out"
POST = Path("/home/ubuntu/PRAX/pre-blog/blog-source/content/posts/2026/07/jlens-cka-397b")
STEM = "readout-concentration"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Inter","Arial","DejaVu Sans"],
 "figure.facecolor":"#F7F4F0","axes.facecolor":"#F7F4F0","savefig.facecolor":"#F7F4F0",
 "text.color":"#2C2924","axes.edgecolor":"#A89B8C","xtick.color":"#5A544C","ytick.color":"#5A544C",
 "axes.labelcolor":"#2C2924","svg.hashsalt":"prax-ablation"})
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

ORDER = ["gemma-2-2b", "gemma-2-9b", "gemma-2-27b", "gemma-3-1b", "qwen3-4b"]
COL = {"gemma-2-2b":"#C4A176", "gemma-2-9b":"#A67C52", "gemma-2-27b":"#6B4E2E",
       "gemma-3-1b":"#8FA37A", "qwen3-4b":"#4B6787"}


def build():
    d = json.load(open(DEC/"readout_ablation_r2048.json"))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.5), gridspec_kw={"width_ratios":[1.35,1]})

    for s in ORDER:
        r = d[s]; bk = {int(k): v for k, v in r["band_sep_by_k"].items()}
        ks = sorted(bk); ys = [bk[k] for k in ks]
        xs = [max(k, 0.5) for k in ks]          # 0 plotted at 0.5 on the log axis
        lab = f"{s} ({'flat' if r['flat_lens'] else 'structured'})"
        ax.plot(xs, ys, "-o", color=COL[s], lw=1.9, ms=4.5, label=lab)
    ax.set_xscale("log"); ax.set_xticks([0.5,1,2,4,8,16,32,64,128])
    ax.set_xticklabels(["0","1","2","4","8","16","32","64","128"], fontsize=8)
    ax.set_xlabel("top readout eigendirections ablated (k)", fontsize=9.5)
    ax.set_ylabel("band separation", fontsize=9.5)
    ax.set_title("Ablating top readout directions restores depth structure",
                 fontsize=10.2, fontweight="bold", loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.axhline(0, color="#A89B8C", lw=0.8, ls=":")
    for sp in ("top","right"): ax.spines[sp].set_visible(False)

    # concentration vs how much the top directions MASK (the monotone relation);
    # concentration does NOT track measured flatness across families, only the masking does.
    prs = [d[s]["readout_participation_ratio"]/d[s]["d"] for s in ORDER]   # PR/d: PR is bounded by d
    gain = [{int(k):v for k,v in d[s]["band_sep_by_k"].items()}[16]
            - {int(k):v for k,v in d[s]["band_sep_by_k"].items()}[0] for s in ORDER]
    o = np.argsort(prs)
    ax2.plot([prs[i] for i in o], [gain[i] for i in o], "-", color="#B0A79A", lw=1.0, zorder=1)
    OFF = {"gemma-2-2b": (8, -12), "gemma-2-9b": (8, 8), "gemma-2-27b": (9, 3),
           "gemma-3-1b": (-20, -16), "qwen3-4b": (-16, 12)}
    for s, p, y in zip(ORDER, prs, gain):
        ax2.scatter([p], [y], s=90, color=COL[s], zorder=3)
        ax2.annotate(s, (p, y), xytext=OFF[s], textcoords="offset points", fontsize=8, color=COL[s])
    ax2.set_xlabel("readout participation ratio / d  (higher = less concentrated)", fontsize=9)
    ax2.set_ylabel("band separation masked by top 16 directions", fontsize=9)
    ax2.set_title("More concentrated readout masks more",
                  fontsize=10.2, fontweight="bold", loc="left")
    ax2.set_xlim(0.02, 0.55); ax2.set_ylim(-0.03, 0.185)
    ax2.axhline(0, color="#A89B8C", lw=0.8, ls=":")
    for sp in ("top","right"): ax2.spines[sp].set_visible(False)

    fig.tight_layout()
    fig.savefig(POST/f"{STEM}.svg", format="svg", metadata={"Date": None})
    fig.savefig(POST/f"{STEM}.png", dpi=170); plt.close(fig)

    alt = ("Two panels. Left: band separation versus number of ablated top readout directions, "
           "for four models. The two flat gemma-2 models rise steeply from about 0.01 to about "
           "0.16 by k=16; gemma-2-27b rises from 0.13 to 0.21; qwen3-4b is nearly flat, 0.039 to "
           "0.056. Right: readout participation ratio against band separation masked by the top 16 directions, "
           "with gemma models at low participation ratio (266, 293) and qwen highest (1191).")
    (POST/f"{STEM}.receipt.json").write_text(json.dumps({
        "figure_id": STEM,
        "title": "Readout concentration and the ablation recovery of depth structure",
        "alt_text": alt, "description": alt,
        "data_source": [{"receipt": "jspace_atlas/decompose_out/readout_ablation_r2048.json",
                         "sha256": sha(DEC/"readout_ablation_r2048.json")}],
        "provenance": {"generator": "jspace_atlas/figures/build_ablation_fig.py",
                       "svg_sha256": sha(POST/f"{STEM}.svg")},
        "interval_semantics": "descriptive statistics on one lens per model; no resampling",
        "plotted_values": {s: {"participation_ratio": d[s]["readout_participation_ratio"],
                               "band_sep_by_k": d[s]["band_sep_by_k"],
                               "atlas_mid_sep": d[s]["atlas_mid_sep"],
                               "rank_energy_share": d[s]["rank_energy_share"]} for s in ORDER},
        "accessibility": {"color_only_channel": False, "text_equivalent": "plotted_values"},
    }, indent=1))
    for s in ORDER:
        bk = {int(k): v for k, v in d[s]["band_sep_by_k"].items()}
        print(f"  {s:13s} PR={d[s]['readout_participation_ratio']:7.1f} k0={bk[0]:+.4f} "
              f"k16={bk[16]:+.4f} gain={bk[16]-bk[0]:+.4f}")
    print("wrote", POST/f"{STEM}.svg")


def verify():
    rec = json.loads((POST/f"{STEM}.receipt.json").read_text())
    got, want = sha(POST/f"{STEM}.svg"), rec["provenance"]["svg_sha256"]
    print("VERIFY", "OK" if got == want else "MISMATCH", got)
    if got != want: raise SystemExit("svg mismatch")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    verify() if a.verify else build()
