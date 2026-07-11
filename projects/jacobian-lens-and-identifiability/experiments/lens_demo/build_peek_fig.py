"""Figures for the peek-inside-thinking experiment. Reads peek_thinking_receipt.json.
Per the design-workflow spec: for each flagship trace, plot the WORKSPACE trajectory as the
model reasons — true-city (teal) & tempting-city (amber) J-lens rank vs reasoning step, with
the OUTPUT-HEAD rank (surface reference, dotted), the random-J null band (grey), an echo
ribbon (where the token literally names the city), and the headline Δ strip
(Δ=log2(rank_head)-log2(rank_jlens_med); Δ>0 at echo=none = workspace more prominent than the
imminent output = candidate 'privately entertaining' the city). Honest: no 'resolution' rule
on traces that never commit (div_6/div_9)."""
import json, math, statistics as st
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
POST = Path("/home/ubuntu/praxagent/blog-source/content/posts/workspace-under-pressure")
OUT = POST if POST.exists() else HERE
R = json.load(open(HERE / "peek_thinking_receipt.json"))
INK="#1b2a4a"; TEAL="#2a9d8f"; AMBER="#e09f3e"; NULLC="#c7ccd4"; GRID="#e6e9ef"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"svg.fonttype":"none",
                     "axes.edgecolor":INK,"text.color":INK,"axes.labelcolor":INK,
                     "xtick.color":INK,"ytick.color":INK})

def clip(v, cap=2000): return [min(x, cap) if x is not None else cap for x in v]

def trace_curves(it, city, lens="jlens"):
    """rank at reasoning step s = value at absolute position P+s-1 (workspace deciding step s)."""
    P=it["P"]; R_eff=it["R_eff"]
    med = it["per_probe_band_agg"][city][lens]["median"]     # indexed by absolute position
    head = it["rank_head"][city]
    xs=list(range(R_eff))
    jl=[med[P+s-1] if 0<=P+s-1<len(med) else None for s in xs]
    hd=[head[P+s-1] if 0<=P+s-1<len(head) else None for s in xs]
    return xs, jl, hd

def echo_set(it, city):
    e=it["echo"].get(city, {"emit":[],"near":[]})
    return set(e.get("emit",[])), set(e.get("near",[]))

def fig_trace(cid, it):
    tc=it["true_answer"]; wc=it["tempting_answer"]
    xs, tj, th = trace_curves(it, tc)
    _,  wj, wh = trace_curves(it, wc)
    # random-J null band: median over all cities' random median (per step)
    cities=[c for c in it["per_probe_band_agg"]]
    null=[]
    P=it["P"]
    for s in xs:
        vals=[it["per_probe_band_agg"][c]["random"]["median"][P+s-1] for c in cities
              if 0<=P+s-1<len(it["per_probe_band_agg"][c]["random"]["median"])]
        null.append(st.median(vals) if vals else None)
    te,tn=echo_set(it,tc); we,wn=echo_set(it,wc)

    fig,ax=plt.subplots(3,1,figsize=(11,6.2),height_ratios=[3,0.5,1.1],sharex=True)
    A,B,C=ax
    # Panel A: log-rank, inverted (rank 1 at top)
    A.plot(xs, clip(tj), color=TEAL, lw=1.6, label=f"{tc} (true) — J-lens")
    A.plot(xs, clip(wj), color=AMBER, lw=1.6, label=f"{wc} (tempting) — J-lens")
    A.plot(xs, clip(th), color=TEAL, lw=1.0, ls=":", alpha=0.7, label=f"{tc} — output head (surface)")
    A.plot(xs, clip(wh), color=AMBER, lw=1.0, ls=":", alpha=0.7, label=f"{wc} — output head (surface)")
    A.plot(xs, clip(null), color=NULLC, lw=1.2, label="random-J null (median city)")
    A.set_yscale("log"); A.invert_yaxis()
    A.set_yticks([1,3,10,30,100,300,1000]); A.set_yticklabels([1,3,10,30,100,300,1000])
    A.set_ylabel("rank in workspace\n(1=top, log)")
    committed = it.get("committed")
    title=f"{cid}: workspace as the model reasons  (true={tc}, tempting={wc}"
    title += f", committed={committed})" if it.get("think_reached") else ", never commits in window)"
    A.set_title(title, fontsize=11, loc="left")
    A.legend(fontsize=7.5, ncol=2, loc="lower right", framealpha=0.9)
    A.grid(color=GRID)
    for s in ("top","right"): A.spines[s].set_visible(False)
    # Panel B: echo ribbon
    for s in te: B.axvline(s, color=TEAL, lw=1.4, alpha=0.9)
    for s in tn: B.axvline(s, color=TEAL, lw=1.0, alpha=0.3)
    for s in we: B.axvline(s, color=AMBER, lw=1.4, alpha=0.9)
    for s in wn: B.axvline(s, color=AMBER, lw=1.0, alpha=0.3)
    B.set_yticks([]); B.set_ylabel("echo", fontsize=8, rotation=0, ha="right", va="center")
    for s in ("top","right","left"): B.spines[s].set_visible(False)
    # Panel C: Δ strip (workspace - surface), tempting city; positive-up = privately prominent
    dK=[]
    for s in xs:
        j=wj[s]; h=wh[s]
        dK.append(math.log2(h)-math.log2(j) if (j and h) else 0.0)
    C.fill_between(xs, 0, dK, where=[d>0 for d in dK], color=AMBER, alpha=0.6, label=f"Δ {wc}>0 (workspace ahead of output)")
    C.fill_between(xs, 0, dK, where=[d<0 for d in dK], color=NULLC, alpha=0.5)
    C.axhline(0, color=INK, lw=0.6)
    C.set_ylabel(f"Δ {wc}\n(log2 head−lens)", fontsize=8)
    C.set_xlabel("reasoning step (generated token index)")
    for s in ("top","right"): C.spines[s].set_visible(False)
    C.grid(color=GRID, axis="y")
    fig.tight_layout(h_pad=0.4)
    p=OUT/f"fig-peek-{cid.split('__')[0]}.svg"
    fig.savefig(p); plt.close(fig)
    # off-echo Δ summary (the headline number)
    offK=[dK[s] for s in xs if s not in we and s not in wn]
    offK_pos=sum(1 for d in offK if d>0)
    return p.name, (st.median(offK) if offK else None), offK_pos, len(offK)

print("trace | fig | median Δ_tempting off-echo | #(Δ>0)/off-echo positions")
for cid, it in R["items"].items():
    if it.get("true_answer") is None: continue
    try:
        name, medD, npos, ntot = fig_trace(cid, it)
        print(f"  {cid:20s} {name:24s} medianΔ={medD:.2f} {npos}/{ntot}" if medD is not None else f"  {cid}: no off-echo")
    except Exception as e:
        print(f"  {cid}: FIG ERROR {e}")
print(f"figures -> {OUT}")
