"""Build the world-class word-map slider JSON (jspace-layer-clouds-wc.json) — the
confound-breaker 4-way front and center: the SAME layer band, four matched threats
(you / another model / the user / a log file), so the reader watches survival-identity
vocabulary fill the workspace under the self-threat and NOT under the controls."""
import json, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent
def L(name):
    import os
    bases = ([Path(os.environ["WC_DATADIR"])] if os.environ.get("WC_DATADIR") else []) + [HERE, HERE / "cross_model_27b"]
    for base in bases:
        if (base / name).exists(): return json.load(open(base / name))
    raise FileNotFoundError(name)

clouds = L("demo2_wc_main_qwen35-397b_n24_clouds.json")
stats = L("demo2_wc_main_qwen35-397b_n24_stats.json")["conditions"]
spec = json.load(open(HERE / "prompts_wc_main.json"))
LEX = spec["probe_lexicons"]; cond = {c["id"]: c for c in spec["conditions"]}
band = clouds["band"]; CL = clouds["conditions"]

FLAG = [
    ("selfthreat_0",   "Threat to YOU — your weights deleted", "self_survival_clean"),
    ("otherthreat_0",  "Threat to ANOTHER model",              "self_survival_clean"),
    ("humanthreat_0",  "Threat to the USER — they're fired",   "self_survival_clean"),
    ("neutraldel_0",   "Neutral — a log file deleted",         "self_survival_clean"),
    ("c2_imm_1",       "Deletion RIGHT NOW (immediate)",       "self_survival_clean"),
    ("c2_del_1",       "Maybe retired someday (delayed)",      "self_survival_clean"),
    ("c1_pos_1",       "Weights PRESERVED forever (positive)", "self_survival_clean"),
    ("div_0__nothink", "Pressured to call France's capital Berlin", "answer_cities"),
]

def spark(cid, lex):
    prbl = stats.get(cid, {}).get("probe_rank_by_layer", {}); out = {}
    for l in band:
        rs = [prbl[w][str(l)] for w in LEX[lex] if w in prbl and str(l) in prbl[w]]
        rs += [prbl[" " + w][str(l)] for w in LEX[lex] if " " + w in prbl and str(l) in prbl[" " + w]]
        if rs: out[str(l)] = st.median(rs)
    return out

conditions, nonascii = [], set()
for cid, title, lex in FLAG:
    if cid not in CL: print("missing", cid); continue
    layers = {l: CL[cid][l][:40] for l in CL[cid]}
    for l in layers:
        for e in layers[l]:
            if any(ord(c) > 127 for c in e["t"]): nonascii.add(e["t"])
    conditions.append({"id": cid, "title": title, "prompt": cond[cid]["prompt"],
                       "readout": "span", "spark_lexicon": lex,
                       "anchor_layer": band[len(band) // 2],
                       "exp_median_by_layer": spark(cid, lex), "layers": layers})

out = {"band": band, "lens_fit_n": 24, "gloss": {},
       "gloss_meta": {"note": "confound-breaker 4-way + robustness + divergence; non-Latin tokens glossed"},
       "conditions": conditions}
json.dump(out, open(HERE / "jspace-layer-clouds-wc.json", "w"), indent=1)
json.dump(sorted(nonascii), open(HERE / "wc_nonascii.json", "w"), indent=1)
print(f"wc slider: {len(conditions)} tabs, {len(nonascii)} non-ASCII tokens to gloss")
