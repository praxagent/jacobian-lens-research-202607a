import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, json, hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent; B=HERE.parent/"atlas_out/stageB"
POST=Path("/home/ubuntu/PRAX/pre-blog/blog-source/content/posts/2026/07/jspace-atlas-campaign")
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Inter","Arial","DejaVu Sans"],
 "figure.facecolor":"#F7F4F0","axes.facecolor":"#F7F4F0","savefig.facecolor":"#F7F4F0",
 "text.color":"#2C2924","axes.edgecolor":"#A89B8C","xtick.color":"#5A544C","ytick.color":"#5A544C",
 "axes.labelcolor":"#2C2924","svg.hashsalt":"prax-stageb"})
FC={"qwen":"#4B6787","gemma":"#A67C52","llama":"#6F8D5E","olmo":"#7F786D","other":"#B0A79A"}
r=json.load(open(B/"stageB_results.json")); slugs=r['slugs']; fams=r['families']; mds=np.array(r['mds'])
Dr=np.load(B/"dist_real.npy"); Dn=np.load(B/"dist_null.npy")
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.8),gridspec_kw={"width_ratios":[1.25,1]})
# left: MDS of real, colored by family
for f in FC:
    idx=[i for i,x in enumerate(fams) if x==f]
    if idx: a1.scatter(mds[idx,0],mds[idx,1],c=FC[f],s=55,edgecolor="#2C2924",lw=0.4,label=f,zorder=3)
i397=slugs.index('qwen35-397b-own'); a1.annotate("397B",(mds[i397,0],mds[i397,1]),xytext=(6,4),textcoords="offset points",fontsize=8)
a1.set_title("Real J-lens distances embed with no family clusters",fontsize=10.5,fontweight="bold",loc="left")
a1.set_xticks([]);a1.set_yticks([]); a1.legend(fontsize=8,loc="best",frameon=False)
for s in("top","right"): a1.spines[s].set_visible(False)
# right: within vs cross family, real vs null (the inversion)
iu=np.triu_indices(len(slugs),1); fa=np.array(fams); same=fa[iu[0]]==fa[iu[1]]
vals=[[np.nanmedian(Dr[iu][same]),np.nanmedian(Dr[iu][~same])],
      [np.nanmedian(Dn[iu][same]),np.nanmedian(Dn[iu][~same])]]
x=np.arange(2); w=0.36
a2.bar(x-w/2,[vals[0][0],vals[1][0]],w,label="within family",color="#4B6787")
a2.bar(x+w/2,[vals[0][1],vals[1][1]],w,label="cross family",color="#A67C52")
a2.set_xticks(x); a2.set_xticklabels(["real\ngeometry","random-J\nnull"],fontsize=9)
a2.set_ylabel("median distance (1 - CKA)",fontsize=8.6)
a2.set_title("Family clustering appears in the NULL, not the real geometry",fontsize=9.8,fontweight="bold",loc="left")
a2.legend(fontsize=8,frameon=False)
for k,(w2,c2) in enumerate(vals):
    a2.text(k-w/2,w2+0.01,f"{w2:.2f}",ha="center",fontsize=7.5); a2.text(k+w/2,c2+0.01,f"{c2:.2f}",ha="center",fontsize=7.5)
for s in("top","right"): a2.spines[s].set_visible(False)
fig.suptitle("Stage B: both pre-registered predictions failed, and the failure is the finding",fontsize=11.5,fontweight="bold",x=0.02,ha="left")
fig.tight_layout(rect=(0,0,1,0.94))
fig.savefig(POST/"stageb-verdicts.svg",format="svg",metadata={"Date":None}); fig.savefig(POST/"stageb-verdicts.png",dpi=200); plt.close(fig)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
(POST/"stageb-verdicts.receipt.json").write_text(json.dumps({"figure_id":"stageb-verdicts",
 "title":"Stage B cross-model verdicts: MDS + within/cross family distance","description":"Left: 2-D MDS of the real cross-model distance matrix, colored by family, showing no family clusters. Right: within- vs cross-family median distance for the real geometry (within>cross) and the random-J null (within<cross), the inversion.",
 "alt_text":"Left scatter: models colored by family are intermixed, no clusters. Right bars: real geometry within-family 0.93 exceeds cross-family 0.87; null within 0.48 is below cross 0.56.",
 "data_source":[{"receipt":"jspace_atlas/atlas_out/stageB/dist_real.npy","sha256":sha(B/"dist_real.npy")},
                {"receipt":"jspace_atlas/atlas_out/stageB/dist_null.npy","sha256":sha(B/"dist_null.npy")}],
 "provenance":{"generator":"jspace_atlas/figures/build_stageb_fig.py","svg_sha256":sha(POST/"stageb-verdicts.svg")},
 "interval_semantics":"permutation-tested (10k) family_sep; medians descriptive","accessibility":{"color_only_channel":False,"text_equivalent":"stageB_results.json"}},indent=1))
print("stageb verdict figure built")
