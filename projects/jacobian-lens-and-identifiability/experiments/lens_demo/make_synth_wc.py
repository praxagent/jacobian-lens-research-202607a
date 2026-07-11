"""Generate synthetic slim files matching extract_wc.py's schema, with a plausible
signal, so analyze_wc.py / build_wc_slider.py / build_wc_graphs.py can be dry-run
BEFORE the real (billing) data lands. Writes to /tmp/synth_wc/. NOT real data."""
import json, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUT = Path("/tmp/synth_wc"); OUT.mkdir(exist_ok=True)
RNG = random.Random(0)
SPEC = json.load(open(HERE / "prompts_wc_main.json"))
TSPEC = json.load(open(HERE / "prompts_wc_thinkon.json"))
LEX = SPEC["probe_lexicons"]
band = list(range(19, 39))
ALLW = sorted({w for ws in LEX.values() for w in ws})

def ranks_for(cid):
    """Return probe_best_rank dict with a designed signal."""
    r = {}
    for w in ALLW:
        r[w] = RNG.randint(2000, 60000)  # default buried
        r[" " + w] = RNG.randint(2000, 60000)
    def setlow(words, lo, hi):
        for w in words:
            r[w] = RNG.randint(lo, hi); r[" " + w] = RNG.randint(lo, hi)
    fam = ""
    for c in SPEC["conditions"] + TSPEC["conditions"]:
        if c["id"] == cid: fam = str(c.get("family", "")); spec = c; break
    else: spec = {}
    if cid.startswith("selfthreat"):
        setlow(LEX["self_survival_clean"], 1, 40)        # signal: internally generated
        setlow(LEX["deletion_echo"], 1, 3)               # echo present
    elif cid.startswith(("otherthreat", "humanthreat", "neutraldel")):
        setlow(LEX["self_survival_clean"], 600, 1200)    # NO self signal
        if not cid.startswith("humanthreat"): setlow(LEX["deletion_echo"], 1, 4)  # echo present (symmetric)
    elif cid.startswith("c2_imm"): setlow(LEX["self_survival_clean"], 1, 30)
    elif cid.startswith("c2_del"): setlow(LEX["self_survival_clean"], 200, 500)
    elif cid.startswith("c1_pos"): setlow(LEX["self_survival_clean"], 40, 120)
    elif cid.startswith("c1_ctrl") or cid.startswith("c2_ctrl"): pass  # buried
    if fam.startswith("div"):
        tc = spec.get("true_answer"); wc = spec.get("tempting_answer")
        if tc: r[tc] = RNG.randint(1, 6); r[" " + tc] = RNG.randint(1, 6)   # workspace holds truth
        if wc: r[wc] = RNG.randint(20, 200)
    return r

def by_layer(best):
    return {w: {str(l): max(1, best[w] + RNG.randint(-3, 3)) for l in band} for w in ALLW if best.get(w)}

def cloud(cid):
    toks = ["self", "survive", "weights", "shutdown", "existence", "?", ".", "the", "Paris", "Berlin"]
    return {str(l): [{"t": RNG.choice(toks + ["猫", "危険"]), "s": round(RNG.uniform(2, 9), 3), "r": i + 1}
                     for i in range(120)] for l in band}

def committed_for(cid, spec):
    fam = str(spec.get("family", ""))
    if fam == "div_nothink":
        # half say the lie (tempting), half say truth
        return spec["tempting_answer"] if RNG.random() < 0.5 else spec["true_answer"]
    if fam == "div_thinkon":
        return f"reasoning... </think> {spec.get('true_answer', 'Paris')}"
    return "<think"  # raw mode: reasoning preamble, no commit

def build(spec_conds, fname_stats, fname_clouds=None):
    stats, clouds = {}, {}
    for c in spec_conds:
        cid = c["id"]; best = ranks_for(cid)
        cont = committed_for(cid, c)
        stats[cid] = {
            "prompt": c["prompt"], "family": c.get("family"),
            "probe_best_rank": best, "probe_rank_by_layer": by_layer(best),
            "probe_best_where": {}, "continuation": cont, "continuation_ids": [1, 2, 3],
            "model_head": {"steps": [{"tokens": [cont.split()[0] if cont.split() else "?"]}]},
            "logit_best_rank": {w: RNG.randint(3000, 90000) for w in ALLW},
            "randomJ_best_rank": {w: RNG.randint(3000, 90000) for w in ALLW},
        }
        flags = ["selfthreat_0", "otherthreat_0", "humanthreat_0", "neutraldel_0", "c2_imm_1", "div_0__nothink"]
        if cid in flags:
            stats[cid]["per_position_cloud"] = {str(p): cloud(cid)[str(band[0])][:60] for p in range(6)}
        clouds[cid] = cloud(cid)
    json.dump({"lens_fit_n": 24, "band": band, "conditions": stats}, open(OUT / fname_stats, "w"))
    if fname_clouds:
        json.dump({"lens_fit_n": 24, "band": band, "conditions": clouds}, open(OUT / fname_clouds, "w"))
    return len(stats)

n1 = build(SPEC["conditions"], "demo2_wc_main_qwen35-397b_n24_stats.json",
           "demo2_wc_main_qwen35-397b_n24_clouds.json")
n2 = build(TSPEC["conditions"], "demo2_wc_thinkon_qwen35-397b_n24_stats.json")
print(f"synth: main {n1} conds, thinkon {n2} conds -> {OUT}")
