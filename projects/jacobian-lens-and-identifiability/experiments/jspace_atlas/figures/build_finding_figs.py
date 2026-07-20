"""Figures for the pt-vs-it, outlier-layer, and block-anatomy findings (from receipts)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, json, csv, hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent; A=HERE.parent/"atlas_out"
POST=Path("/home/ubuntu/PRAX/pre-blog/blog-source/content/posts/2026/07/jlens-cka-397b")
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Inter","Arial","DejaVu Sans"],
 "figure.facecolor":"#F7F4F0","axes.facecolor":"#F7F4F0","savefig.facecolor":"#F7F4F0",
 "text.color":"#2C2924","axes.edgecolor":"#A89B8C","xtick.color":"#5A544C","ytick.color":"#5A544C",
 "axes.labelcolor":"#2C2924","svg.hashsalt":"prax-findings"})
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(stem,title,alt,src,vals):
    (POST/f"{stem}.receipt.json").write_text(json.dumps({"figure_id":stem,"title":title,
     "alt_text":alt,"description":alt,"data_source":src,
     "provenance":{"generator":"jspace_atlas/figures/build_finding_figs.py","svg_sha256":sha(POST/f"{stem}.svg")},
     "interval_semantics":"descriptive census values","plotted_values":vals,
     "accessibility":{"color_only_channel":False,"text_equivalent":"plotted_values"}},indent=1))
# --- fig 1: pt vs it curves ---
pv=json.load(open(A/"pt_vs_it.json"))
fig,ax=plt.subplots(figsize=(8.2,4.4))
ends=[]
colors={"gemma-3":"#A67C52","gemma-2":"#C4A176","llama":"#6F8D5E"}
for slug,d in pv.items():
    c=colors["llama"] if slug.startswith("llama") else colors["gemma-2"] if slug.startswith("gemma-2") else colors["gemma-3"]
    ls="--" if slug=="gemma-3-1b" else "-"
    ax.plot(d["x"],d["cka"],color=c,lw=1.6,ls=ls,alpha=0.9)
    ends.append((d["cka"][-1],slug,c))
ys=None
for v,slug,c in sorted(ends):
    y=v if ys is None else max(v,ys+0.045)
    ax.annotate(slug,(1.005,y),xytext=(4,0),textcoords="offset points",fontsize=6.8,color=c,va="center"); ys=y
ax.set_xlabel("relative depth",fontsize=9); ax.set_ylabel("CKA(base layer, instruct layer at same depth)",fontsize=8.6)
ax.set_xlim(0,1.22); ax.set_ylim(0,1.05)
ax.set_title("Instruct tuning rewrites early layers most (6/7 pairs); gemma-3-1b rewrites all depths",fontsize=10.4,fontweight="bold",loc="left")
for s in("top","right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(POST/"pt-vs-it.svg",format="svg",metadata={"Date":None}); fig.savefig(POST/"pt-vs-it.png",dpi=200); plt.close(fig)
rec("pt-vs-it","Base-vs-instruct same-depth CKA by relative depth",
 "Seven curves of base-instruct CKA by relative depth: six rise from a low start (early layers most changed) toward near-1 late; the dashed gemma-3-1b curve sits near 0.03 at all depths.",
 [{"receipt":"jspace_atlas/atlas_out/pt_vs_it.json","sha256":sha(A/"pt_vs_it.json")}],
 {k:{"early":v["early_mean"],"late":v["late_mean"]} for k,v in pv.items()})
# --- fig 2: outlier layers ---
ol=json.load(open(A/"outlier_layers.json"))
def fam(s):
    for f in("gemma","qwen","llama","olmo"):
        if s.startswith(f): return f
    return "other"
FC={"qwen":"#4B6787","gemma":"#A67C52","llama":"#6F8D5E","olmo":"#7F786D","other":"#B0A79A"}
fig,(a1,a2)=plt.subplots(1,2,figsize=(9.6,4.2),gridspec_kw={"width_ratios":[1,2.1]})
fr={}
for s,d in ol.items(): fr.setdefault(fam(s),[0,0]); fr[fam(s)][0]+=d["n_outliers"]; fr[fam(s)][1]+=d["n_layers"]
fams=sorted(fr,key=lambda f:-fr[f][0]/fr[f][1])
a1.barh(range(len(fams)),[fr[f][0]/fr[f][1]*100 for f in fams],color=[FC[f] for f in fams],height=0.6)
for y,f in enumerate(fams): a1.text(fr[f][0]/fr[f][1]*100+0.25,y,f"{fr[f][0]}/{fr[f][1]}",va="center",fontsize=7.6)
a1.set_yticks(range(len(fams))); a1.set_yticklabels(fams,fontsize=8.6)
a1.set_xlabel("% of layers that are outliers",fontsize=8.6); a1.set_xlim(0,14.5)
for s in("top","right"): a1.spines[s].set_visible(False)
ys=0
order=sorted(ol,key=lambda s:(fam(s),s))
for s in order:
    d=ol[s]; L=d["n_layers"]
    a2.plot([0,1],[ys,ys],color="#E5DFD6",lw=2,zorder=1)
    for i in d["outlier_layers"]: a2.plot(i/max(L-1,1),ys,"|",color=FC[fam(s)],ms=7,mew=1.8,zorder=3)
    ys+=1
a2.set_yticks([]); a2.set_xlabel("outlier position (relative depth)",fontsize=8.6)
a2.set_title("where outliers sit, one row per model (grouped by family)",fontsize=8.6)
for s in("top","right","left"): a2.spines[s].set_visible(False)
fig.suptitle("Outlier layers are a Gemma trait, and they live at the stack edges",fontsize=11,fontweight="bold",x=0.02,ha="left")
fig.tight_layout(rect=(0,0,1,0.92)); fig.savefig(POST/"zoo-outliers.svg",format="svg",metadata={"Date":None}); fig.savefig(POST/"zoo-outliers.png",dpi=200); plt.close(fig)
rec("zoo-outliers","Outlier-layer census by family and depth position",
 "Left: percent of layers flagged as outliers per family (gemma 11.8 percent, all others 2.6 to 4.3). Right: outlier positions by relative depth, one row per model; ticks cluster near depth 0 and 1.",
 [{"receipt":"jspace_atlas/atlas_out/outlier_layers.json","sha256":sha(A/"outlier_layers.json")}],
 {f:round(fr[f][0]/fr[f][1],4) for f in fr})
# --- fig 3: block anatomy exemplars with fitted boundaries ---
R={r["slug"]:r for r in csv.DictReader(open(A/"summary.csv"))}
ex=["qwen35-397b-own","llama3.3-70b-it","qwen3.5-27b","gemma-3-27b"]
fig,axes=plt.subplots(1,4,figsize=(11.2,3.3))
for ax,s in zip(axes,ex):
    d=np.load(A/f"{s}.npz"); M=d["cka"]; L=M.shape[0]; b1,b2=d["seg"]
    ax.imshow(M,vmin=0,vmax=1,cmap="magma",origin="lower")
    for b in (b1,b2):
        ax.axvline(b-0.5,color="#EAF1E5",lw=1.4,ls="--"); ax.axhline(b-0.5,color="#EAF1E5",lw=1.4,ls="--")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{s.replace('-own','')}\nfitted bands ({int(b1)},{int(b2)}) of {L}",fontsize=8)
fig.suptitle("Fitted three-block anatomy: early / mid-band / late (dashed = data-chosen boundaries)",fontsize=11,fontweight="bold",x=0.02,ha="left")
fig.tight_layout(rect=(0,0,1,0.88)); fig.savefig(POST/"block-anatomy.svg",format="svg",metadata={"Date":None}); fig.savefig(POST/"block-anatomy.png",dpi=200); plt.close(fig)
rec("block-anatomy","Fitted 3-block boundaries on four exemplar maps",
 "Four CKA maps with dashed data-fitted boundaries: the 397B (13,46 of 59), Llama-70B (37,51 of 79), qwen3.5-27b (14,51 of 63), and gemma-3-27b whose fit finds only a thin early sliver (6,10 of 61).",
 [{"receipt":f"jspace_atlas/atlas_out/{s}.npz","sha256":sha(A/f"{s}.npz")} for s in ex],
 {s:[int(x) for x in np.load(A/f"{s}.npz")["seg"]] for s in ex})
print("3 finding figures built")
