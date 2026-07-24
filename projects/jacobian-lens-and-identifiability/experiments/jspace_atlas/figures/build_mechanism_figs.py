"""Figures for the Gemma-mechanism deep dive: (1) readout-specificity bars (isotropic vs
readout band-sep for gemma-2-9b vs qwen3-4b), (2) instruct-flattens slope lines (8 pairs).
Reads decompose_out/{decompose_results,zoo_regression}.json.  --verify checks svg shas.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, json, hashlib, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEC = HERE.parent / "decompose_out"
POST = Path("/home/ubuntu/PRAX/pre-blog/blog-source/content/posts/2026/07/jlens-cka-397b")
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Inter","Arial","DejaVu Sans"],
 "figure.facecolor":"#F7F4F0","axes.facecolor":"#F7F4F0","savefig.facecolor":"#F7F4F0",
 "text.color":"#2C2924","axes.edgecolor":"#A89B8C","xtick.color":"#5A544C","ytick.color":"#5A544C",
 "axes.labelcolor":"#2C2924","svg.hashsalt":"prax-mech"})
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(stem, title, alt, src, vals):
    (POST/f"{stem}.receipt.json").write_text(json.dumps({"figure_id":stem,"title":title,
     "alt_text":alt,"description":alt,"data_source":src,
     "provenance":{"generator":"jspace_atlas/figures/build_mechanism_figs.py","svg_sha256":sha(POST/f"{stem}.svg")},
     "interval_semantics":"descriptive band-separation statistics; no resampling",
     "plotted_values":vals,"accessibility":{"color_only_channel":False,"text_equivalent":"plotted_values"}},indent=1))

BROWN, BLUE, GREEN, MUTE = "#A67C52", "#4B6787", "#6F8D5E", "#B0A79A"


def fig_readout_specificity():
    d = json.load(open(DEC/"decompose_results.json"))
    models = ["gemma-2-9b", "qwen3-4b"]
    iso = [d[m]["random_probe_band_sep_isotropic"] for m in models]
    rdo = [d[m]["readout_band_sep_Mweighted"] for m in models]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(models)); w = 0.36
    ax.bar(x - w/2, iso, w, color=MUTE, label="isotropic (random probe): does J rotate at all?")
    ax.bar(x + w/2, rdo, w, color=BROWN, label="readout (real unembedding): does the output projection rotate?")
    for i, m in enumerate(models):
        ratio = iso[i]/max(rdo[i], 1e-4)
        ax.text(i, max(iso[i], rdo[i]) + 0.008, f"{int(ratio)}x gap" if ratio > 3 else f"{ratio:.1f}x",
                ha="center", fontsize=10, fontweight="bold", color=BROWN if ratio > 3 else "#5A544C")
        ax.text(i - w/2, iso[i] + 0.003, f"{iso[i]:+.3f}", ha="center", fontsize=8, color="#5A544C")
        ax.text(i + w/2, rdo[i] + 0.003, f"{rdo[i]:+.3f}", ha="center", fontsize=8, color="#5A544C")
    ax.set_xticks(x); ax.set_xticklabels(["gemma-2-9b\n(flat lens)", "qwen3-4b\n(control)"], fontsize=9.5)
    ax.set_ylabel("layer x layer band separation", fontsize=9.5)
    ax.set_ylim(0, 0.30)
    ax.set_title("Gemma's Jacobian rotates; only its readout projection stays fixed",
                 fontsize=11.0, fontweight="bold", loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(POST/"readout-specificity.svg", format="svg", metadata={"Date":None})
    fig.savefig(POST/"readout-specificity.png", dpi=170); plt.close(fig)
    rec("readout-specificity", "Isotropic vs readout band separation (readout-specificity of Gemma flatness)",
        "Grouped bar chart. gemma-2-9b: isotropic band separation +0.242 (tall) vs readout +0.005 (near zero), labelled 51x gap. qwen3-4b: isotropic +0.069 vs readout +0.038, labelled 1.8x.",
        [{"receipt":"jspace_atlas/decompose_out/decompose_results.json","sha256":sha(DEC/"decompose_results.json")}],
        {m:{"isotropic":iso[i],"readout":rdo[i]} for i,m in enumerate(models)})
    print("wrote readout-specificity.svg", {m:(iso[i],rdo[i]) for i,m in enumerate(models)})


def fig_instruct_flattens():
    z = json.load(open(DEC/"zoo_regression.json"))
    pairs = z["base_to_instruct"]["pairs"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    def col(m): return BROWN if m.startswith("gemma") else GREEN if m.startswith("llama") else BLUE
    for p in pairs:
        ax.plot([0, 1], [p["base"], p["it"]], "-o", color=col(p["model"]), lw=1.6, ms=5, alpha=0.9)
    # de-overlap the right-hand labels: sort by it-value, enforce a minimum vertical gap
    lab = sorted(pairs, key=lambda p: p["it"])
    py = None
    for p in lab:
        y = p["it"] if py is None else max(p["it"], py + 0.011)
        ax.annotate(p["model"], (1.02, p["it"]), xytext=(1.04, y), fontsize=7.2, color=col(p["model"]),
                    va="center", arrowprops=dict(arrowstyle="-", color=col(p["model"]), lw=0.5, alpha=0.5))
        py = y
    ax.set_xlim(-0.08, 1.42); ax.set_xticks([0, 1]); ax.set_xticklabels(["base", "instruct"], fontsize=10)
    ax.set_ylabel("readout band separation (shared vocab)", fontsize=9.5)
    md = z["base_to_instruct"]["mean_delta"]; nf = z["base_to_instruct"]["n_flatter"]; npr = z["base_to_instruct"]["n_pairs"]
    ax.set_title(f"Instruct tuning flattens the readout in every family ({nf}/{npr} pairs)",
                 fontsize=11.2, fontweight="bold", loc="left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(POST/"instruct-flattens.svg", format="svg", metadata={"Date":None})
    fig.savefig(POST/"instruct-flattens.png", dpi=170); plt.close(fig)
    rec("instruct-flattens", "Base-to-instruct readout band separation (instruct flattens the readout)",
        "Slope chart. Eight base-to-instruct pairs across gemma-2, gemma-3 and llama; every line slopes downward from its base value to a lower instruct value.",
        [{"receipt":"jspace_atlas/decompose_out/zoo_regression.json","sha256":sha(DEC/"zoo_regression.json")}],
        {p["model"]:{"base":p["base"],"it":p["it"]} for p in pairs})
    print("wrote instruct-flattens.svg", f"{nf}/{npr} flatter, mean {md}")


def verify():
    for stem in ("readout-specificity", "instruct-flattens"):
        rec_j = json.loads((POST/f"{stem}.receipt.json").read_text())
        got = sha(POST/f"{stem}.svg"); want = rec_j["provenance"]["svg_sha256"]
        print("VERIFY", stem, "OK" if got == want else "MISMATCH")
        if got != want: raise SystemExit(f"{stem} svg mismatch")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify: verify()
    else:
        fig_readout_specificity(); fig_instruct_flattens()
