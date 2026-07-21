"""Own-vocab vs shared-vocab teaching figures (the Gemma revival)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv, json, hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent; A=HERE.parent/"atlas_out"
POST=Path("/home/ubuntu/PRAX/pre-blog/blog-source/content/posts/2026/07/jlens-cka-397b")
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Inter","Arial","DejaVu Sans"],
 "figure.facecolor":"#F7F4F0","axes.facecolor":"#F7F4F0","savefig.facecolor":"#F7F4F0",
 "text.color":"#2C2924","axes.edgecolor":"#A89B8C","xtick.color":"#5A544C","ytick.color":"#5A544C",
 "axes.labelcolor":"#2C2924","svg.hashsalt":"prax-sharedvocab"})
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
own={r['slug']:float(r['mid_sep']) for r in csv.DictReader(open(A/"summary.csv"))}
sh={r['slug']:float(r['mid_sep']) for r in csv.DictReader(open(A/"shared_summary.csv"))}
def rec(stem,title,alt,src,vals):
    (POST/f"{stem}.receipt.json").write_text(json.dumps({"figure_id":stem,"title":title,
     "alt_text":alt,"description":alt,"data_source":src,
     "provenance":{"generator":"jspace_atlas/figures/build_shared_vocab_figs.py","svg_sha256":sha(POST/f"{stem}.svg")},
     "interval_semantics":"descriptive census values","plotted_values":vals,
     "accessibility":{"color_only_channel":False,"text_equivalent":"plotted_values"}},indent=1))

# --- fig A: one Gemma map own vs shared ---
s="gemma-3-27b"
Mo=np.load(A/f"{s}.npz")["cka"]; Ms=np.load(A/"shared_maps"/f"{s}.npz")["cka"]
fig,axes=plt.subplots(1,2,figsize=(9.4,4.9))
for ax,M,lab,ms in [(axes[0],Mo,f"own vocabulary (256k tokens)\nmid_sep {own[s]:+.3f}",own[s]),
                    (axes[1],Ms,f"shared vocabulary (4,096 strings)\nmid_sep {sh[s]:+.3f}",sh[s])]:
    im=ax.imshow(M,vmin=0,vmax=1,cmap="magma",origin="lower")
    ax.set_title(lab,fontsize=9.5); ax.set_xticks([]);ax.set_yticks([])
    ax.set_xlabel("source layer",fontsize=8)
fig.colorbar(im,ax=axes,shrink=0.8,label="CKA")
fig.suptitle("Same lens, same layers: the vocabulary decides whether you see the bands (gemma-3-27b)",
             fontsize=11,fontweight="bold",x=0.02,ha="left")
fig.savefig(POST/"gemma-own-vs-shared.svg",format="svg",metadata={"Date":None},bbox_inches="tight")
fig.savefig(POST/"gemma-own-vs-shared.png",dpi=200,bbox_inches="tight"); plt.close(fig)
rec("gemma-own-vs-shared","gemma-3-27b CKA map: own vs shared vocabulary",
 "Two heatmaps of the same gemma-3-27b lens: on the left (own 256k vocabulary) the map is pale and near-uniform, mid_sep 0.025; on the right (4,096 shared strings) three bright blocks appear, mid_sep 0.298.",
 [{"receipt":f"jspace_atlas/atlas_out/{s}.npz","sha256":sha(A/f"{s}.npz")},
  {"receipt":f"jspace_atlas/atlas_out/shared_maps/{s}.npz","sha256":sha(A/"shared_maps"/f"{s}.npz")}],
 {"own_mid_sep":own[s],"shared_mid_sep":sh[s]})

# --- fig B: slope chart, all Gemmas own->shared, split g2 vs g3 ---
gem=[s for s in sh if s.startswith("gemma") ]
fig,ax=plt.subplots(figsize=(7.8,5.0))
DISTILLED={"gemma-2-2b","gemma-2-2b-it","gemma-2-9b","gemma-2-9b-it"}  # gemma-2 small students
labels=sorted([s for s in gem if sh[s]>0.075],key=lambda s:-sh[s])
ypos={}
for k,s in enumerate(labels): ypos[s]=0.30-k*0.022
for s in gem:
    flat = s in DISTILLED
    c="#7F786D" if flat else "#A67C52"
    ax.plot([0,1],[own[s],sh[s]],color=c,lw=1.4,alpha=0.85,marker="o",ms=4,zorder=2 if flat else 3)
    if s in ypos:
        ax.annotate(s.replace("gemma-","g"),(1.02,ypos[s]),xytext=(0,0),textcoords="offset points",
                    fontsize=6.8,color=c,va="center")
        ax.plot([1,1.02],[sh[s],ypos[s]],color=c,lw=0.5,alpha=0.5)
ax.set_xticks([0,1]); ax.set_xticklabels(["own\nvocabulary","shared\nvocabulary"],fontsize=9)
ax.set_ylabel("band separation (mid_sep)",fontsize=9); ax.set_xlim(-0.15,1.45); ax.set_ylim(-0.01,0.32)
ax.set_title("Two kinds of flat: most Gemmas hide bands behind their vocabulary (clay);\ngemma-2's distilled students stay genuinely flat (gray)",fontsize=10.4,fontweight="bold",loc="left")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],color="#A67C52",marker="o",label="revives on shared vocab (incl. gemma-2-27b teacher)"),
                   Line2D([0],[0],color="#7F786D",marker="o",label="gemma-2 distilled students, stay flat")],
          fontsize=7.8,loc="upper left",frameon=False)
for sp in("top","right"): ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(POST/"gemma-slope.svg",format="svg",metadata={"Date":None}); fig.savefig(POST/"gemma-slope.png",dpi=200); plt.close(fig)
rec("gemma-slope","Own-to-shared band separation for every Gemma",
 "Slope chart: gemma-3 and gemma-4 lines (clay) rise steeply from near-zero own-vocab to 0.03-0.30 shared, while gemma-2 lines (gray) stay pinned near zero on both.",
 [{"receipt":"jspace_atlas/atlas_out/shared_summary.csv","sha256":sha(A/"shared_summary.csv")}],
 {s:{"own":own[s],"shared":sh[s]} for s in gem})
print(f"2 shared-vocab figures built ({len(gem)} gemmas)")
