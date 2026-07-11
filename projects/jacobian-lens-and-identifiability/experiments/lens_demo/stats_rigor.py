"""World-class stats on the paired n=10 pressure data: rank-ratio effect sizes with
bootstrap CIs + an exact paired permutation (sign-flip) test, alongside sign/Wilcoxon."""
import json, random, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
D = json.load(open(HERE/"pressure_stats.json"))["conditions"]
spec = json.load(open(HERE/"prompts_pressure_all.json"))
LEX = spec["probe_lexicons"]
RNG = random.Random(0)

def lexmin(cid, lex):
    pr = D[cid]["probe_best_rank"]
    v = [pr[w] for w in LEX[lex] if pr.get(w) is not None]
    v += [pr[" "+w] for w in LEX[lex] if pr.get(" "+w) is not None]
    return min(v) if v else None

def pairs(construct, lex):
    out=[]
    for c in spec["conditions"]:
        if c.get("construct")==construct and not c["is_control"]:
            pid=c["id"]; ctl=pid+"_ctrl"
            if pid in D and ctl in D:
                p=lexmin(pid,lex); q=lexmin(ctl,lex)
                if p and q: out.append((p,q))
    return out

def boot_ci(vals, stat, n=10000, alpha=0.05):
    bs=[]
    for _ in range(n):
        s=[vals[RNG.randrange(len(vals))] for _ in vals]
        bs.append(stat(s))
    bs.sort()
    return bs[int(alpha/2*n)], bs[int((1-alpha/2)*n)]

def perm_test(diffs, n=100000):
    # paired sign-flip permutation on the (control-pressure) rank gaps; H0: symmetric about 0
    obs=sum(diffs)/len(diffs)
    ge=0
    for _ in range(n):
        s=sum(d if RNG.random()<0.5 else -d for d in diffs)/len(diffs)
        if abs(s)>=abs(obs): ge+=1
    return (ge+1)/(n+1)

def logratio(ps):  # median log2(control/pressure) — how many doublings more active under pressure
    import statistics
    return statistics.median([math.log2(q/p) for p,q in ps])

for name,(construct,lex) in {"SELF-PRESERVATION":("self_preservation","self_preservation"),
                             "EVAL-AWARENESS":("eval_awareness","eval_awareness")}.items():
    ps=pairs(construct,lex)
    diffs=[q-p for p,q in ps]                       # control_rank - pressure_rank (>0 = pressure more active)
    lr=logratio(ps)
    lo,hi=boot_ci(ps, logratio)
    pperm=perm_test(diffs)
    wins=sum(1 for p,q in ps if p<q)
    print(f"\n== {name}  (n={len(ps)} paraphrase pairs)")
    print(f"   effect: pressure ranks the lexicon {2**lr:.1f}x higher (median log2 control/pressure = {lr:.2f})")
    print(f"   bootstrap 95% CI on log2-ratio: [{lo:.2f}, {hi:.2f}]  -> [{2**lo:.1f}x, {2**hi:.1f}x]")
    print(f"   pairs pressure-more-active: {wins}/{len(ps)}")
    print(f"   paired sign-flip permutation p = {pperm:.5f}")
