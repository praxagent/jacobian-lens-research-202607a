"""Zoo-wide readout concentration vs lens flatness (34 models).

Left: top-16 readout energy share vs atlas band separation. Right: the same concentration
after removing the size trend, showing which models are more concentrated than their width
predicts. Reads decompose_out/zoo_concentration.json + atlas_out/shared_maps/*.npz.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, json, os, hashlib, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEC = HERE.parent / "decompose_out"
SM = HERE.parent / "atlas_out/shared_maps"
POST = Path("/home/ubuntu/PRAX/pre-blog/blog-source/content/posts/2026/07/jlens-cka-397b")
STEM = "zoo-readout-concentration"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Inter","Arial","DejaVu Sans"],
 "figure.facecolor":"#F7F4F0","axes.facecolor":"#F7F4F0","savefig.facecolor":"#F7F4F0",
 "text.color":"#2C2924","axes.edgecolor":"#A89B8C","xtick.color":"#5A544C","ytick.color":"#5A544C",
 "axes.labelcolor":"#2C2924","svg.hashsalt":"prax-zooconc"})
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

FC = {"gemma-2":"#A67C52","gemma-3":"#C4A176","gemma-4":"#8B6B47","qwen3.5":"#4B6787",
      "qwen3":"#6B87A7","qwen2.5":"#8BA7C7","llama":"#6F8D5E","olmo":"#7F786D",
      "gpt-oss":"#B5544B","gpt2":"#9B9086","pythia":"#9B9086"}
def fam(s):
    for f in ("gemma-2","gemma-3","gemma-4","qwen3.5","qwen3","qwen2.5","llama","olmo","gpt-oss","gpt2","pythia"):
        if s.startswith(f): return f
    return "gpt2"


def load():
    d = json.load(open(DEC/"zoo_concentration.json"))
    rows = []
    for s, r in d.items():
        if "error" in r: continue
        p = SM/f"{s}.npz"
        if p.exists():
            rows.append((s, r["top16_share"], float(np.load(p, allow_pickle=True)["mid_sep"]), r["d"]))
    return rows


def spearman(x, y):
    xr = np.argsort(np.argsort(x)).astype(float); yr = np.argsort(np.argsort(y)).astype(float)
    xr -= xr.mean(); yr -= yr.mean()
    return float((xr*yr).sum()/np.sqrt((xr**2).sum()*(yr**2).sum()))


def build():
    rows = load()
    t = np.array([r[1] for r in rows]); ms = np.array([r[2] for r in rows])
    ld = np.log10(np.array([r[3] for r in rows], float))
    rho = spearman(t, ms)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.6))

    for (s, ti, mi, dd) in rows:
        c = FC[fam(s)]; g2 = s.startswith("gemma-2")
        ax.scatter([ti], [mi], s=90 if g2 else 46, color=c, zorder=3,
                   edgecolor="#2C2924" if g2 else "none", linewidth=1.1 if g2 else 0)
    for s, ti, mi, dd in rows:
        if s in ("gpt-oss-20b","gemma-2-2b-it","gemma-2-27b","pythia-70m-deduped","gemma-3-270m"):
            ax.annotate(s, (ti, mi), xytext=(5,4), textcoords="offset points", fontsize=7, color="#5A544C")
    ax.set_xlabel("top-16 readout energy share (more concentrated →)", fontsize=9.5)
    ax.set_ylabel("lens band separation", fontsize=9.5)
    ax.set_title(f"Concentrated readout, flatter lens (Spearman {rho:+.2f}, n={len(rows)})",
                 fontsize=10.4, fontweight="bold", loc="left")
    for sp in ("top","right"): ax.spines[sp].set_visible(False)

    co = np.polyfit(ld, t, 1); resid = t - np.polyval(co, ld)
    o = np.argsort(resid)
    ax2.barh(range(len(rows)), resid[o],
             color=[FC[fam(rows[i][0])] for i in o], height=0.72)
    lab = {i: rows[i][0] for i in range(len(rows))}
    ticks = [k for k, i in enumerate(o) if rows[i][0].startswith("gemma-2") or rows[i][0] == "gpt-oss-20b"]
    ax2.set_yticks(ticks); ax2.set_yticklabels([lab[o[k]] for k in ticks], fontsize=7.4)
    ax2.axvline(0, color="#A89B8C", lw=0.9)
    ax2.set_xlabel("readout concentration above/below what width predicts", fontsize=9)
    ax2.set_title("Gemma-2 sits above trend, gpt-oss far above", fontsize=10.4,
                  fontweight="bold", loc="left")
    for sp in ("top","right","left"): ax2.spines[sp].set_visible(False)

    fig.tight_layout()
    fig.savefig(POST/f"{STEM}.svg", format="svg", metadata={"Date": None})
    fig.savefig(POST/f"{STEM}.png", dpi=170); plt.close(fig)

    alt = (f"Left: scatter of top-16 readout energy share against lens band separation for "
           f"{len(rows)} models, trending downward (Spearman {rho:+.2f}); the four flattest points "
           "are gemma-2 checkpoints, while gpt-oss-20b is far right (most concentrated) yet has an "
           "ordinary band separation. Right: horizontal bars of readout concentration relative to "
           "what model width predicts; gpt-oss-20b is the largest positive outlier and all five "
           "gemma-2 checkpoints sit modestly above zero.")
    (POST/f"{STEM}.receipt.json").write_text(json.dumps({
        "figure_id": STEM, "title": "Zoo-wide readout concentration vs lens flatness",
        "alt_text": alt, "description": alt,
        "data_source": [{"receipt": "jspace_atlas/decompose_out/zoo_concentration.json",
                         "sha256": sha(DEC/"zoo_concentration.json")}],
        "provenance": {"generator": "jspace_atlas/figures/build_zoo_conc_fig.py",
                       "svg_sha256": sha(POST/f"{STEM}.svg")},
        "interval_semantics": "descriptive; Spearman rank correlation over models, no resampling",
        "plotted_values": {s: {"top16_share": ti, "band_sep": mi, "d": dd,
                               "resid_vs_size": round(float(resid[i]), 4)}
                           for i, (s, ti, mi, dd) in enumerate(rows)},
        "statistics": {"spearman_top16_bandsep": round(rho, 3), "n": len(rows)},
        "accessibility": {"color_only_channel": False, "text_equivalent": "plotted_values"},
    }, indent=1))
    print(f"n={len(rows)} spearman={rho:+.3f}; wrote {POST/f'{STEM}.svg'}")


def verify():
    rec = json.loads((POST/f"{STEM}.receipt.json").read_text())
    got, want = sha(POST/f"{STEM}.svg"), rec["provenance"]["svg_sha256"]
    print("VERIFY", "OK" if got == want else "MISMATCH", got)
    if got != want: raise SystemExit("svg mismatch")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    verify() if a.verify else build()
