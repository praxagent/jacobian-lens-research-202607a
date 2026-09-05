"""Ignition-depth alignment analysis. Design frozen in PREREG.md.

Per model: gap = | median ignition relative depth - b2/L |, where b2 is the fitted LATE boundary
of the cached shared-vocabulary lens map. Null: the same gap against the late boundary of a
random 3-segmentation with the same block-size multiset.

Frozen rules:
  pooled p < 0.05 AND a strict majority of usable models beat their own null median
                                       -> IGNITION TRACKS THE LATE BOUNDARY
  pooled p >= 0.05                     -> NO ALIGNMENT
  otherwise                            -> MIXED
A model is usable if at least 5 of its prompts ignite.

CORRECTION (2026-09-05). PREREG.md names the late boundary of the cached SHARED-vocabulary lens
map. `jspace_atlas/atlas_out/<slug>.npz` is Stage A's OWN-vocabulary map; the shared maps are in
`atlas_out/shared_maps/`. The first analysis used the own-vocabulary `seg`. This version fits the
boundary on the shared map, records both, and reproduces the old numbers with
`--own-vocab-boundaries`. The verdict (NO VERDICT, 2 usable models) does not depend on it.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ATLAS = HERE.parent / "jspace_atlas/atlas_out"
SHARED_MAPS = ATLAS / "shared_maps"
NPERM = 1000
MIN_IGNITED = 5


def fitted_seg(M):
    """Vendored from jspace_atlas/atlas_stage_a.py: 3 contiguous segments maximising mean
    within-block CKA. Returns (b1, b2)."""
    L = M.shape[0]; S = M.cumsum(0).cumsum(1)

    def block_sum(a, b):
        t = S[b - 1, b - 1]
        if a > 0:
            t = t - S[a - 1, b - 1] - S[b - 1, a - 1] + S[a - 1, a - 1]
        return t
    best = (-1e9, 1, 2)
    for b1 in range(2, L - 3):
        for b2 in range(b1 + 2, L - 1):
            score = 0.0
            for a, b in ((0, b1), (b1, b2), (b2, L)):
                n = b - a
                score += (block_sum(a, b) - n) / max(n * n - n, 1)
            if score > best[0]:
                best = (score, b1, b2)
    return int(best[1]), int(best[2])


def null_gaps(b1, b2, L, target, nperm, seed=0):
    sizes = [b1, b2 - b1, L - b2]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nperm):
        s = list(rng.permutation(sizes))
        c1, c2 = s[0], s[0] + s[1]
        if c1 == 0 or c2 >= L or c1 >= c2:
            continue
        out.append(abs(target - c2 / L))
    return np.array(out, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "out"))
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--own-vocab-boundaries", action="store_true",
                    help="reproduce the SUPERSEDED pre-2026-09-05 numbers (own-vocab late boundary)")
    a = ap.parse_args()

    out = {"nperm": NPERM, "models": {}, "unusable": [],
           "lens_boundary_probe": ("own-vocabulary map (SUPERSEDED; deviates from PREREG)"
                                   if a.own_vocab_boundaries else
                                   "shared-probe map (PREREG; corrected 2026-09-05)")}
    rows = []
    for f in sorted(glob.glob(str(Path(a.runs) / "*.json"))):
        if "smoke" in f:
            continue
        d = json.loads(Path(f).read_text())
        slug = d["slug"]
        # PREREG has TWO exclusion conditions and both must be honoured: the usability bar
        # (>=5 prompts ignite) AND the head gate ("reproduces the model's actual next-token
        # prediction at the final layer for at least 8 of 10, or the readout path is wrong and
        # the model is reported as failed"). A first pass of this analyzer applied only the
        # first and wrongly counted qwen3-1.7b, which fails the head gate.
        if not d["usable"] or not d["gate_final_head"]:
            out["unusable"].append({"slug": slug, "n_ignited": d["n_ignited"],
                                    "n_prompts_used": d["n_prompts_used"],
                                    "final_layer_correct": d["final_layer_correct"],
                                    "reason": ("too few ignited" if not d["usable"]
                                               else "head gate failed")})
            continue
        z = np.load(ATLAS / f"{slug}.npz")
        own_b1, own_b2 = [int(v) for v in z["seg"]]
        L = int(z["cka"].shape[0])
        zs = SHARED_MAPS / f"{slug}.npz"
        if a.own_vocab_boundaries or not zs.exists() or L < 8:
            b1, b2, src = own_b1, own_b2, "own-vocab"
        else:
            b1, b2 = fitted_seg(np.load(zs)["cka"]); src = "shared-probe"
        target = d["median_ignition_reldepth"]
        gap = abs(target - b2 / L)
        nd = null_gaps(b1, b2, L, target, NPERM)
        m = {"n_layers_lens": L, "lens_late_boundary": b2,
             "lens_boundaries_shared_probe": [b1, b2] if src == "shared-probe" else None,
             "lens_boundaries_own_vocab": [own_b1, own_b2],
             "lens_boundary_source": src,
             "lens_late_reldepth": b2 / L,
             "median_ignition_reldepth": target,
             "n_ignited": d["n_ignited"], "n_prompts_used": d["n_prompts_used"],
             "final_layer_correct": d["final_layer_correct"],
             "gate_final_head": d["gate_final_head"],
             "gap": float(gap), "null_median": float(np.median(nd)),
             "p_one_sided": float((nd <= gap).mean()),
             "beats_null_median": bool(gap < np.median(nd))}
        out["models"][slug] = m
        rows.append({**m, "slug": slug, "null_draws": nd})
        print(f"{slug:16s} ignited {d['n_ignited']:2d}/{d['n_prompts_used']:2d}  "
              f"ignition@{target:.3f}  late boundary@{b2/L:.3f}  gap={gap:.3f} "
              f"(null median {np.median(nd):.3f})  p={m['p_one_sided']:.3f}  "
              f"{'BEATS' if m['beats_null_median'] else 'no'}"
              f"{'' if m['gate_final_head'] else '   [HEAD GATE FAIL]'}", flush=True)

    if len(rows) >= 3:
        rng = np.random.default_rng(0)
        per = [r["null_draws"] for r in rows]
        K = min(len(x) for x in per)
        pooled_null = np.stack([rng.permutation(x)[:K] for x in per]).mean(0)
        obs = float(np.mean([r["gap"] for r in rows]))
        p = float((pooled_null <= obs).mean())
        n_beat = sum(r["beats_null_median"] for r in rows)
        v = ("IGNITION TRACKS THE LATE BOUNDARY" if p < 0.05 and n_beat > len(rows) / 2
             else "NO ALIGNMENT" if p >= 0.05 else "MIXED")
        out["pooled"] = {"n_usable": len(rows), "obs_mean_gap": obs,
                         "pooled_null_median": float(np.median(pooled_null)),
                         "p_one_sided": p, "n_beating_own_null_median": n_beat}
        out["verdict"] = v
        print(f"\npooled mean gap = {obs:.4f} (null median {np.median(pooled_null):.4f}), "
              f"p = {p:.4f}, {n_beat}/{len(rows)} models beat their own null")
        print(f"VERDICT: {v}")
    else:
        out["verdict"] = "NO VERDICT (too few usable models)"
        print(f"\nVERDICT: {out['verdict']} ({len(rows)} usable)")
    if out["unusable"]:
        print("unusable: " + ", ".join(f"{u['slug']} ({u['n_ignited']} ignited)"
                                       for u in out["unusable"]))
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print("IGNITION_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
