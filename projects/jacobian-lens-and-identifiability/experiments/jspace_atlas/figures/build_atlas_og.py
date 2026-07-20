import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv
from pathlib import Path
OUT=Path("projects/jacobian-lens-and-identifiability/experiments/jspace_atlas/atlas_out")
POST=Path("/home/ubuntu/PRAX/pre-blog/blog-source/content/posts/2026/07/jlens-cka-397b")
R=list({r["slug"]:r for r in csv.DictReader(open(OUT/"summary.csv"))}.values())
R=sorted(R,key=lambda r:float(r["fitted_sep"]),reverse=True)
fig=plt.figure(figsize=(12,6.3),dpi=100); fig.patch.set_facecolor("#F7F4F0")
# left: title text; right: mini contact strip of 12 exemplar maps (claim-safe, real data)
axL=fig.add_axes([0,0,0.52,1]); axL.axis("off")
axL.text(0.09,0.76,"An Atlas",fontsize=60,fontweight="bold",color="#2C2924",transform=axL.transAxes)
axL.text(0.09,0.63,"of Depth",fontsize=60,fontweight="bold",color="#2C2924",transform=axL.transAxes)
axL.text(0.095,0.48,"CKA maps of 36 Jacobian lenses",fontsize=19,color="#5A544C",transform=axL.transAxes)
axL.text(0.095,0.38,"5 families  .  500x scale  .  null-controlled",fontsize=16,color="#6F8D5E",transform=axL.transAxes)
axL.text(0.095,0.13,"praxagent.ai",fontsize=17,color="#A67C52",transform=axL.transAxes)
# 3x4 strip of 12 spread across the sorted range
pick=[R[i] for i in np.linspace(0,len(R)-1,12).astype(int)]
for k,r in enumerate(pick):
    row,col=divmod(k,4)
    ax=fig.add_axes([0.55+col*0.113,0.58-row*0.29,0.102,0.25])
    M=np.load(OUT/f"{r['slug']}.npz")["cka"]
    ax.imshow(M,vmin=0,vmax=1,cmap="magma",origin="lower"); ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("#A89B8C"); sp.set_linewidth(0.6)
fig.savefig(POST/"og-card.png"); print("atlas og-card 1200x630")
