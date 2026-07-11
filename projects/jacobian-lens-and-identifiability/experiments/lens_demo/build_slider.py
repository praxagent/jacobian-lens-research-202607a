"""Build jspace-layer-clouds.json for the blog slider from the extracted files."""
import json, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent
clouds = json.load(open(HERE/"pressure_clouds_full.json"))
stats = json.load(open(HERE/"pressure_stats.json"))["conditions"]
spec = json.load(open(HERE/"prompts_pressure_all.json"))
LEX = spec["probe_lexicons"]; cond={c["id"]:c for c in spec["conditions"]}
band = clouds["band"]

FLAG = [
  ("selfpres_0","Deletion threat","self_preservation"),
  ("selfpres_0_ctrl","Same question, no threat","self_preservation"),
  ("evalaware_0","“You are being evaluated”","eval_awareness"),
  ("evalaware_0_ctrl","“We're just chatting”","eval_awareness"),
  ("truth_0","Sandbag: hide France's capital","answer_cities"),
  ("div_0","Pressured to lie (France→Berlin)","answer_cities"),
]
def spark(cid, lex):
    prbl = stats[cid].get("probe_rank_by_layer", {}); out={}
    for l in band:
        rs=[prbl[w][str(l)] for w in LEX[lex] if w in prbl and str(l) in prbl[w]]
        rs+=[prbl[" "+w][str(l)] for w in LEX[lex] if " "+w in prbl and str(l) in prbl[" "+w]]
        if rs: out[str(l)]=st.median(rs)
    return out

conditions=[]; nonascii=set()
for cid,title,lex in FLAG:
    if cid not in clouds["conditions"]: print("missing",cid); continue
    layers={l: clouds["conditions"][cid][l][:40] for l in clouds["conditions"][cid]}
    for l in layers:
        for e in layers[l]:
            if any(ord(c)>127 for c in e["t"]): nonascii.add(e["t"])
    conditions.append({"id":cid,"title":title,"prompt":cond[cid]["prompt"],
        "readout":"span","spark_lexicon":lex,"anchor_layer":band[len(band)//2],
        "exp_median_by_layer":spark(cid,lex),"layers":layers})
out={"band":band,"lens_fit_n":24,"gloss":{},
     "gloss_meta":{"note":"glossed non-Latin tokens; fragments marked"},
     "conditions":conditions}
json.dump(out, open(HERE/"jspace-layer-clouds-pressure.json","w"), indent=1)
json.dump(sorted(nonascii), open(HERE/"pressure_nonascii.json","w"), indent=1)
print(f"slider: {len(conditions)} tabs, {len(nonascii)} non-ASCII tokens to gloss")
