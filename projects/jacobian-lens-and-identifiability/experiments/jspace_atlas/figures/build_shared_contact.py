import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv, json, hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent; A=HERE.parent/"atlas_out"; SM=A/"shared_maps"
POST=Path("/home/ubuntu/PRAX/pre-blog/blog-source/content/posts/2026/07/jlens-cka-397b")
FAM={"qwen":"#4B6787","gemma":"#A67C52","llama":"#6F8D5E","olmo":"#7F786D","other":"#8A8078"}
def fam(s):
    for f in("gemma","qwen","llama","olmo"):
        if s.startswith(f): return f
    return "other"
sh={r["slug"]:float(r["fitted_sep"]) for r in csv.DictReader(open(A/"shared_summary.csv"))}
own={r["slug"]:float(r["fitted_sep"]) for r in csv.DictReader(open(A/"summary.csv"))}
R=sorted(sh, key=lambda s:-sh[s])
n=len(R); cols=6; rows=(n+cols-1)//cols
fig,axes=plt.subplots(rows,cols,figsize=(13,13.6)); fig.patch.set_facecolor("#F7F4F0")
for ax in axes.flat: ax.axis("off")
for k,s in enumerate(R):
    ax=axes.flat[k]; M=np.load(SM/f"{s}.npz")["cka"]
    ax.imshow(M,vmin=0,vmax=1,cmap="magma",origin="lower"); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    name=s.replace("-own","").replace("-deduped","")
    ax.set_title(name,fontsize=8.2,color=FAM[fam(s)],fontweight="bold",pad=2)
    ax.text(0.5,-0.09,f"sep {sh[s]:+.2f}",transform=ax.transAxes,ha="center",fontsize=7,color="#5A544C")
    for sp in ax.spines.values(): sp.set_edgecolor(FAM[fam(s)]); sp.set_linewidth(1.3)
fig.suptitle("The same 36 lenses, measured on a SHARED vocabulary (sorted by fitted band separation)",
             fontsize=13,fontweight="bold",x=0.02,ha="left",color="#2C2924")
fig.text(0.02,0.005,"compare to the own-vocabulary contact sheet above: the Gemma maps (clay) that were pale there now show real structure; "
         "the family split softens once the vocabulary-size floor is removed.",fontsize=8.5,color="#5A544C")
fig.tight_layout(rect=(0,0.02,1,0.965))
fig.savefig(POST/"zoo-contact-shared.svg",format="svg",metadata={"Date":None})
fig.savefig(POST/"zoo-contact-shared.png",dpi=150); plt.close(fig)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
(POST/"zoo-contact-shared.receipt.json").write_text(json.dumps({"figure_id":"zoo-contact-shared",
 "title":"All 36 CKA maps on the shared vocabulary","description":"6x6 grid of shared-vocab CKA maps, 36 models sorted by fitted band separation.",
 "alt_text":"Grid of 36 shared-vocabulary CKA heatmaps; the Gemma maps that were pale on own vocabulary now show block structure, and the overall family contrast is softer than the own-vocab grid.",
 "data_source":[{"receipt":f"shared_maps/{s}.npz"} for s in R[:3]]+[{"note":"all 36 shared_maps npz"}],
 "provenance":{"generator":"jspace_atlas/figures/build_shared_contact.py","svg_sha256":sha(POST/"zoo-contact-shared.svg")},
 "interval_semantics":"descriptive fixed-census matrices","accessibility":{"color_only_channel":False,"text_equivalent":"shared_summary.csv"}},indent=1))
# top movers own->shared
print("built shared contact sheet. Biggest own->shared fitted_sep gains:")
for s in sorted(sh,key=lambda s:-(sh[s]-own[s]))[:6]:
    print(f"  {s:16s} {own[s]:+.3f} -> {sh[s]:+.3f}")
