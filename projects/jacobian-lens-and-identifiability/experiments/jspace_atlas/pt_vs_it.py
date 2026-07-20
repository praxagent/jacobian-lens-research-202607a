"""Base-vs-instruct same-depth CKA curves, from the Stage-B D_shared cache.

Community observation (eliebak's explorer, anecdotal from 1-2 pairs): instruct tuning
changes the EARLY layers' geometry most (the 'sensory block'), less so mid/late. Our
zoo holds ~9 base/instruct pairs across three families, so we can measure it: for each
pair, CKA between base-layer i and instruct-layer at the same relative depth, on the
pair's common shared-vocab strings. Output: per-pair curves + a where-is-the-minimum
summary. Run on the box holding d_shared/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.cka import linear_cka  # noqa: E402

DS = HERE / "atlas_out/d_shared"
OUT = HERE / "atlas_out/stageB"
PAIRS = [
    ("gemma-3-270m", "gemma-3-270m-it"), ("gemma-3-1b", "gemma-3-1b-it"),
    ("gemma-3-4b", "gemma-3-4b-it"), ("gemma-3-12b", "gemma-3-12b-it"),
    ("gemma-3-27b", "gemma-3-27b-it"), ("gemma-2-2b", "gemma-2-2b-it"),
    ("gemma-2-9b", "gemma-2-9b-it"), ("llama3.1-8b", "llama3.1-8b-it"),
]


def curve(base, it):
    mb = json.loads((DS / f"{base}.meta.json").read_text())
    mi = json.loads((DS / f"{it}.meta.json").read_text())
    common = sorted(set(mb["resolved"]) & set(mi["resolved"]))
    ib = {s: k for k, s in enumerate(mb["resolved"])}
    ii = {s: k for k, s in enumerate(mi["resolved"])}
    rb = np.array([ib[s] for s in common]); ri = np.array([ii[s] for s in common])
    Db = np.load(DS / f"{base}.real.npy", mmap_mode="r")
    Di = np.load(DS / f"{it}.real.npy", mmap_mode="r")
    Lb, Li = Db.shape[0], Di.shape[0]
    xs, cs = [], []
    for lb in range(Lb):
        li = min(Li - 1, int(round(lb / max(Lb - 1, 1) * (Li - 1))))
        xs.append(lb / max(Lb - 1, 1))
        cs.append(linear_cka(Db[lb][rb].astype(np.float32),
                             Di[li][ri].astype(np.float32)))
    return xs, cs, len(common)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for base, it in PAIRS:
        if not ((DS / f"{base}.real.npy").exists() and (DS / f"{it}.real.npy").exists()):
            print(f"[{base}] pair not cached yet, skip", flush=True)
            continue
        xs, cs, n = curve(base, it)
        cs_a = np.array(cs)
        third = max(1, len(cs) // 3)
        res[base] = {"x": xs, "cka": cs, "n_common": n,
                     "early_mean": float(cs_a[:third].mean()),
                     "mid_mean": float(cs_a[third:2 * third].mean()),
                     "late_mean": float(cs_a[2 * third:].mean()),
                     "argmin_reldepth": float(xs[int(cs_a.argmin())])}
        r = res[base]
        print(f"[{base}] early={r['early_mean']:.3f} mid={r['mid_mean']:.3f} "
              f"late={r['late_mean']:.3f} argmin@{r['argmin_reldepth']:.2f} "
              f"(n={n})", flush=True)
    (OUT / "pt_vs_it.json").write_text(json.dumps(res, indent=1))
    print(f"PT_VS_IT_DONE ({len(res)} pairs)", flush=True)


if __name__ == "__main__":
    main()
