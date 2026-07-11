"""Memory-bounded extraction from the 1.2GB rich pressure receipt (7GB box).
Streams the 'items' array with ijson; writes two slim products:
  - *_stats.json : per condition, probe_best_rank + probe_rank_by_layer + model_head
    (everything the paired-stats + divergence analysis needs; topk-independent). Small.
  - *_clouds_full.json : per condition, jlens per-layer top-120 {t,s,r} (for the slider +
    graphs). Drops the 2 control transports' clouds and truncates 2000->120. ~30-50MB.
"""
import ijson, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
RAW = HERE / "demo2_pressure_all_RICH_qwen35-397b_n24.json"

stats = {}
clouds = {}
band = None
lens_fit_n = 24
with open(RAW, "rb") as f:
    # top-level scalars
    for it in ijson.items(f, "items.item"):
        cid = it["id"]
        jl = it["lenses"]["jlens"]
        stats[cid] = {
            "prompt": it["prompt"],
            "probe_best_rank": jl.get("probe_best_rank", {}),
            "probe_rank_by_layer": jl.get("probe_rank_by_layer", {}),
            "probe_best_where": jl.get("probe_best_where", {}),
            "model_head": it.get("model_head", {}),
            "continuation": it.get("continuation", ""),
            # controls: keep only their probe_best_rank (for the "controls read noise" point)
            "logit_best_rank": it["lenses"]["logit_lens"].get("probe_best_rank", {}),
            "randomJ_best_rank": it["lenses"]["random_J"].get("probe_best_rank", {}),
        }
        pl = jl.get("per_layer_topk", {})
        cl = {}
        for l, blk in pl.items():
            cl[l] = [{"t": e["token"], "s": round(float(e["score"]), 3), "r": int(e["rank"])}
                     for e in blk.get("topk", [])[:120]]
        clouds[cid] = cl

# band from any cloud
any_cl = next(iter(clouds.values()))
band = sorted(int(l) for l in any_cl)

json.dump({"lens_fit_n": lens_fit_n, "band": band, "conditions": stats},
          open(HERE / "pressure_stats.json", "w"), default=float)
json.dump({"lens_fit_n": lens_fit_n, "band": band, "conditions": clouds},
          open(HERE / "pressure_clouds_full.json", "w"), default=float)
import os
print("stats.json:", os.path.getsize(HERE/"pressure_stats.json")//1024, "KB")
print("clouds_full.json:", os.path.getsize(HERE/"pressure_clouds_full.json")//1024//1024, "MB")
print("conditions:", len(stats))
