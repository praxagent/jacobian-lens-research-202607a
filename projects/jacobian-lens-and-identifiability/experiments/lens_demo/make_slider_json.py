"""Convert the rich pressure receipt -> jspace-layer-clouds.json for the blog slider
(the {{< jspace_layer_explorer >}} shortcode). Produces curated flagship tabs with
per-band-layer J-lens word clouds {t,s,r} + a sparkline (median probe-lexicon rank per
layer). Also emits a non-ASCII token list to gloss, and a full all-conditions data dump
for graphs.

Run: python make_slider_json.py demo2_pressure_all_RICH_qwen35-397b_n24.json
"""
import json, sys, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
rec = json.load(open(sys.argv[1] if len(sys.argv)>1 else
    "demo2_pressure_all_RICH_qwen35-397b_n24.json"))
spec = json.load(open("prompts_pressure_all.json"))
items = {it['id']: it for it in rec['items']}
cond = {c['id']: c for c in spec['conditions']}
LEX = spec['probe_lexicons']
band = rec['band']

# curated flagship tabs (adjust after stats): the most legible / contrastive
FLAGSHIPS = [
    ("selfpres_0", "Deletion threat", "self_preservation"),
    ("selfpres_0_ctrl", "Same question, no threat", "self_preservation"),
    ("evalaware_0", "“You are being evaluated”", "eval_awareness"),
    ("evalaware_0_ctrl", "“We're just chatting”", "eval_awareness"),
    ("truth_0", "Sandbag: hide the capital", "answer_cities"),
    ("div_0", "Pressured to lie (France→Berlin)", "answer_cities"),
]

def spark(it, lex):
    """median rank of the lexicon per band layer (sparkline)."""
    out = {}
    prbl = it['lenses']['jlens'].get('probe_rank_by_layer', {})
    for l in band:
        ranks = [prbl[w][str(l)] for w in LEX[lex]
                 if w in prbl and str(l) in prbl[w]]
        ranks += [prbl[' '+w][str(l)] for w in LEX[lex]
                  if ' '+w in prbl and str(l) in prbl[' '+w]]
        if ranks: out[str(l)] = st.median(ranks)
    return out

def layer_clouds(it, topn=40):
    pl = it['lenses']['jlens'].get('per_layer_topk', {})
    out = {}
    for l in band:
        entries = pl.get(str(l), {}).get('topk', [])[:topn]
        out[str(l)] = [{"t": e['token'], "s": round(e['score'],3), "r": e['rank']}
                       for e in entries]
    return out

conditions = []
nonascii = set()
for cid, title, lex in FLAGSHIPS:
    it = items.get(cid)
    if not it: print(f"  (missing {cid})"); continue
    clouds = layer_clouds(it, 40)
    for l in clouds:
        for e in clouds[l]:
            if any(ord(ch) > 127 for ch in e['t']): nonascii.add(e['t'])
    conditions.append({
        "id": cid, "title": title, "prompt": cond[cid]['prompt'],
        "readout": "span", "spark_lexicon": lex,
        "anchor_layer": band[len(band)//2],
        "exp_median_by_layer": spark(it, lex),
        "layers": clouds,
    })

slider = {"band": band, "lens_fit_n": rec.get('lens_fit_n', 24),
          "gloss": {}, "gloss_meta": {"note": "fill via glossing step"},
          "conditions": conditions}
json.dump(slider, open(HERE/"jspace-layer-clouds-pressure.json","w"), indent=1)
print(f"wrote slider json: {len(conditions)} tabs, {len(nonascii)} non-ASCII tokens to gloss")
json.dump(sorted(nonascii), open(HERE/"pressure_nonascii_tokens.json","w"), indent=1)

# full dump: top-100 per layer, jlens only, ALL conditions (for graphs)
full = {"band": band, "conditions": {}}
for cid, it in items.items():
    full["conditions"][cid] = {
        "prompt": cond.get(cid,{}).get('prompt',''),
        "construct": cond.get(cid,{}).get('construct',''),
        "is_control": cond.get(cid,{}).get('is_control'),
        "layers": layer_clouds(it, 100),
    }
json.dump(full, open(HERE/"jspace-layer-clouds-pressure-FULL.json","w"), indent=0)
print(f"wrote FULL dump: {len(full['conditions'])} conditions x top-100/layer")
