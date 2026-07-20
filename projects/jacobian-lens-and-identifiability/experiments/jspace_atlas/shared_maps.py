"""Within-model SHARED-vocabulary CKA maps, from Stage-B's D_shared cache.

The atlas (Stage A) measured each model on its OWN 4,096-token probe. For a model
with a very large vocabulary (Gemma: ~256k) that inflates the CKA floor and can flatten
the map into near-uniformity. This recomputes each model's layer x layer CKA on the
SHARED-vocabulary probe (4,096 strings common to every tokenizer) using the D_shared
geometries Stage B already cached, so the maps are comparable across models on equal
footing. Output: atlas_out/shared_maps/<slug>.npz (cka, mid_sep, fitted_sep) + a
shared_summary.csv. Resumable. Run on the box that holds d_shared/.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "jacobian_lens"))
from common.cka import linear_cka          # noqa: E402
from cka_layers import band_stats          # noqa: E402
from atlas_stage_b import family_of      # noqa: E402

DS = HERE / "atlas_out/d_shared"
OUT = HERE / "atlas_out/shared_maps"


def fitted_sep_of(M):
    sys.path.insert(0, str(HERE))
    from atlas_stage_a import fitted_seg as fs
    _, _, sep = fs(M)
    return sep


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    metas = sorted(DS.glob("*.meta.json"))
    for mp in metas:
        slug = mp.stem.replace(".meta", "")
        out = OUT / f"{slug}.npz"
        if out.exists():
            continue
        meta = json.loads(mp.read_text())
        D = np.load(DS / f"{slug}.real.npy")          # (n_layers, n_res, d) fp16
        L = D.shape[0]
        M = np.eye(L)
        for i in range(L):
            gi = D[i].astype(np.float32)
            for j in range(i + 1, L):
                M[i, j] = M[j, i] = linear_cka(gi, D[j].astype(np.float32))
        bs = band_stats(M, meta["layers"])
        fsep = fitted_sep_of(M)
        np.savez_compressed(out, cka=M, layers=np.array(meta["layers"]),
                            mid_sep=np.float64(bs["mid_sep"]),
                            fitted_sep=np.float64(fsep),
                            n_shared=np.int64(meta["n_res"]))
        with open(HERE / "atlas_out/shared_summary.csv", "a", newline="") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["slug", "family", "n_shared", "mid_sep", "fitted_sep"])
            w.writerow([slug, family_of(slug), meta["n_res"],
                        round(bs["mid_sep"], 4), round(fsep, 4)])
        print(f"[{slug}] shared L={L} n={meta['n_res']} "
              f"mid_sep={bs['mid_sep']:+.4f} fitted={fsep:+.4f}", flush=True)
    print("SHARED_MAPS_DONE", flush=True)


if __name__ == "__main__":
    run()
