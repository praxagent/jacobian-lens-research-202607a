import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv, json, hashlib
from pathlib import Path
OUT=Path("projects/jacobian-lens-and-identifiability/experiments/jspace_atlas/atlas_out")
POST=Path("/home/ubuntu/PRAX/pre-blog/blog-source/content/posts/2026/07/jlens-cka-397b")
FAM={"qwen":"#4B6787","gemma":"#A67C52","llama":"#6F8D5E","olmo":"#7F786D","other":"#8A8078"}
def fam(s):
    for f in ("gemma","qwen","llama","olmo"):
        if s.startswith(f): return f
    return "other"
R=list({r["slug"]:r for r in csv.DictReader(open(OUT/"summary.csv"))}.values())
R=sorted(R,key=lambda r:float(r["fitted_sep"]),reverse=True)
n=len(R); cols=6; rows=(n+cols-1)//cols
fig,axes=plt.subplots(rows,cols,figsize=(13,13.6)); fig.patch.set_facecolor("#F7F4F0")
for ax in axes.flat: ax.axis("off")
for k,r in enumerate(R):
    ax=axes.flat[k]; d=np.load(OUT/f"{r['slug']}.npz"); M=d["cka"]
    ax.imshow(M,vmin=0,vmax=1,cmap="magma",origin="lower"); ax.axis("on")
    ax.set_xticks([]); ax.set_yticks([])
    name=r["slug"].replace("-own","").replace("-deduped","")
    ax.set_title(f"{name}",fontsize=8.2,color=FAM[fam(r['slug'])],fontweight="bold",pad=2)
    ax.text(0.5,-0.09,f"sep {float(r['fitted_sep']):+.2f}",transform=ax.transAxes,
            ha="center",fontsize=7,color="#5A544C")
    for sp in ax.spines.values(): sp.set_edgecolor(FAM[fam(r['slug'])]); sp.set_linewidth(1.3)
fig.suptitle("Layer x layer CKA maps, all 36 lenses (sorted by fitted band separation)",
             fontsize=13,fontweight="bold",x=0.02,ha="left",color="#2C2924")
fig.text(0.02,0.005,"same magma scale 0 to 1 as the 397B hero map; title color = family; "
         "bright diagonal blocks = depth phases. Qwen and Llama band; Gemma stays smooth.",
         fontsize=8.5,color="#5A544C")
fig.tight_layout(rect=(0,0.02,1,0.965))
fig.savefig(POST/"zoo-contact-sheet.svg",format="svg",metadata={"Date":None})
fig.savefig(POST/"zoo-contact-sheet.png",dpi=150); plt.close(fig)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
(POST/"zoo-contact-sheet.receipt.json").write_text(json.dumps({
 "figure_id":"zoo-contact-sheet","title":"All 36 layer-by-layer CKA maps",
 "description":"6x6 grid of layer-by-layer CKA heatmaps for all 36 fitted lenses, sorted by fitted band separation, magma 0-1, family-colored titles.",
 "alt_text":"Grid of 36 CKA heatmaps sorted by band strength: Qwen and Llama lenses (top) show bright blocky diagonals; Gemma lenses (bottom) are smooth near-uniform maps.",
 "data_source":[{"receipt":f"atlas_out/{r['slug']}.npz"} for r in R],
 "provenance":{"generator":"jspace_atlas contact-sheet","svg_sha256":sha(POST/"zoo-contact-sheet.svg")},
 "interval_semantics":"descriptive fixed-census matrices, no sampling",
 "accessibility":{"color_only_channel":False,"text_equivalent":"per-model fitted_sep in summary.csv + labels"}},indent=1))
print("contact sheet:",n,"maps")
