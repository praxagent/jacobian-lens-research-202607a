"""Test B re-run analysis. Design frozen in PREREG_B2.md.

Reports the verdict under BOTH gates side by side, as pre-committed, so the sensitivity is
visible rather than a preferred slice:

  OLD gate: exclude when median off-diagonal activation CKA >= 0.999
  NEW gate: exclude when the off-diagonal range < 0.10

and under both the raw and the per-dimension-standardised activation map, the latter being the
pre-registered robustness check on the instrument itself.

Repaired decision rule, scaling with the number of usable models k (k >= 4 required):
  pooled p < 0.05 AND a strict majority of usable models beat their own null median
                                                     -> BANDS TRACK REPRESENTATIONS
  pooled p >= 0.05                                   -> BANDS ARE A READOUT PROPERTY
  otherwise                                          -> MIXED

CORRECTION (2026-09-05). PREREG.md specifies the boundaries of each model's SHARED-vocabulary
lens map. The runner read `jspace_atlas/atlas_out/<slug>.npz`, the file the prereg itself named,
but that file is Stage A's OWN-vocabulary map (see atlas_stage_a.py's docstring); the shared maps
live in `atlas_out/shared_maps/<slug>.npz`. Every per-model receipt therefore carries own-vocabulary
lens boundaries, and for 5 of the 12 models the two probes disagree by 3 to 9 layers. This analyzer
now fits the boundaries from the shared maps (`--lens-maps`), keeps the own-vocabulary values in the
output for the record, and reproduces the superseded numbers with `--own-vocab-boundaries`.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SHARED_MAPS = HERE.parent / "jspace_atlas/atlas_out/shared_maps"
NPERM = 1000


def fitted_seg(M):
    """Vendored from jspace_atlas/atlas_stage_a.py (same objective, equality-checked there by
    activation_boundaries.test_fitted_seg_matches_atlas): the 3 contiguous segments maximising
    mean within-block CKA. Returns (b1, b2)."""
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


def shared_boundaries(maps_dir, slug, L):
    """Lens boundaries from the shared-probe map; None if no map or too few layers to segment."""
    f = Path(maps_dir) / f"{slug}.npz"
    if not f.exists():
        return None
    M = np.load(f)["cka"]
    if M.shape[0] != L or L < 8:
        return None
    return list(fitted_seg(M))
OLD_MEDIAN_GATE = 0.999
NEW_RANGE_GATE = 0.10
MIN_USABLE = 4
ANCHOR, ANCHOR_TARGET, ANCHOR_TOL = "gemma-2-9b", 0.0812, 0.02


def null_agreement(b1, b2, L, act, nperm, seed=0):
    sizes = [b1, b2 - b1, L - b2]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nperm):
        s = list(rng.permutation(sizes))
        c1, c2 = s[0], s[0] + s[1]
        if c1 == 0 or c2 >= L or c1 >= c2:
            continue
        out.append(abs(c1 - act[0]) + abs(c2 - act[1]))
    return np.array(out, float)


def verdict_for(rows, label):
    """Apply the repaired decision rule to a set of usable models."""
    if len(rows) < MIN_USABLE:
        return {"label": label, "n_usable": len(rows),
                "verdict": "NO VERDICT (too few usable models)"}
    rng = np.random.default_rng(0)
    per_null = [r["null_draws"] / r["n_layers"] for r in rows]
    K = min(len(x) for x in per_null)
    pooled_null = np.stack([rng.permutation(x)[:K] for x in per_null]).mean(0)
    obs = float(np.mean([r["agreement"] / r["n_layers"] for r in rows]))
    p = float((pooled_null <= obs).mean())
    n_beat = sum(r["beats_null_median"] for r in rows)
    if p < 0.05 and n_beat > len(rows) / 2:
        v = "BANDS TRACK REPRESENTATIONS"
    elif p >= 0.05:
        v = "BANDS ARE A READOUT PROPERTY"
    else:
        v = "MIXED"
    return {"label": label, "n_usable": len(rows),
            "models": [r["slug"] for r in rows],
            "obs_mean_agreement_norm": obs,
            "pooled_null_median": float(np.median(pooled_null)),
            "p_one_sided": p, "n_beating_own_null_median": n_beat, "verdict": v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "outB"))
    ap.add_argument("--out", default=str(HERE / "results_B2.json"))
    ap.add_argument("--lens-maps", default=str(SHARED_MAPS),
                    help="dir of shared-probe lens maps <slug>.npz (PREREG); boundaries are fitted here")
    ap.add_argument("--own-vocab-boundaries", action="store_true",
                    help="reproduce the SUPERSEDED pre-2026-09-05 numbers (own-vocab boundaries)")
    a = ap.parse_args()

    recs = {}
    for f in sorted(glob.glob(str(Path(a.runs) / "*.json"))):
        if "smoke" in f or "probe" in f:
            continue
        d = json.loads(Path(f).read_text())
        recs[d["slug"]] = d

    out = {"nperm": NPERM, "old_median_gate": OLD_MEDIAN_GATE,
           "new_range_gate": NEW_RANGE_GATE, "n_models_run": len(recs),
           "lens_boundary_probe": ("own-vocabulary map (SUPERSEDED; deviates from PREREG)"
                                   if a.own_vocab_boundaries else
                                   "shared-probe map (PREREG; corrected 2026-09-05)"),
           "models": {}}

    variants = {}   # (map_kind, gate) -> list of usable rows
    for kind in ("raw", "standardised"):
        for gate in ("old", "new"):
            variants[(kind, gate)] = []

    for slug, d in recs.items():
        L = d["n_layers"]; lb_own = d["lens_boundaries"]
        lb_shared = shared_boundaries(a.lens_maps, slug, L)
        lb = lb_own if (a.own_vocab_boundaries or lb_shared is None) else lb_shared
        m = {"n_layers": L, "lens_boundaries": lb,
             "lens_boundaries_own_vocab": lb_own,
             "lens_boundaries_shared_probe": lb_shared,
             "lens_boundary_source": ("own-vocab" if (a.own_vocab_boundaries or lb_shared is None)
                                      else "shared-probe")}
        for kind, bkey, medkey, rngkey in (
                ("raw", "act_boundaries", "act_offdiag_median", "act_range"),
                ("standardised", "std_act_boundaries", "std_act_offdiag_median",
                 "std_act_range")):
            if bkey not in d:
                continue
            ab = d[bkey]
            agree = abs(lb[0] - ab[0]) + abs(lb[1] - ab[1])
            nd = null_agreement(lb[0], lb[1], L, ab, NPERM)
            row = {"slug": slug, "n_layers": L, "agreement": agree,
                   "beats_null_median": bool(agree < np.median(nd)), "null_draws": nd}
            m[kind] = {"boundaries": ab, "agreement": int(agree),
                       "agreement_norm": agree / L,
                       "null_median": float(np.median(nd)),
                       "p_one_sided": float((nd <= agree).mean()),
                       "offdiag_median": d[medkey], "range": d.get(rngkey),
                       "excluded_old_gate": bool(d[medkey] >= OLD_MEDIAN_GATE),
                       "excluded_new_gate": bool((d.get(rngkey) or 0) < NEW_RANGE_GATE)}
            if not m[kind]["excluded_old_gate"]:
                variants[(kind, "old")].append(row)
            if not m[kind]["excluded_new_gate"]:
                variants[(kind, "new")].append(row)
        out["models"][slug] = m

    print(f"lens boundaries: {out['lens_boundary_probe']}")
    print(f"{'model':22s} {'L':>3s} {'lens':>9s} {'(own)':>9s} | {'raw b':>9s} {'agr':>4s} {'rng':>6s} "
          f"{'gates':>9s} | {'std b':>9s} {'agr':>4s} {'rng':>6s} {'gates':>9s}")
    for slug, m in out["models"].items():
        r, s_ = m.get("raw", {}), m.get("standardised", {})
        def g(x):
            if not x: return "-"
            return ("old-EX " if x["excluded_old_gate"] else "old-ok ") + \
                   ("new-EX" if x["excluded_new_gate"] else "new-ok")
        print(f"{slug:22s} {m['n_layers']:3d} {str(m['lens_boundaries']):>9s} "
              f"{str(m['lens_boundaries_own_vocab']):>9s} | "
              f"{str(r.get('boundaries','-')):>9s} {r.get('agreement','-'):>4} "
              f"{r.get('range',0):6.3f} {g(r):>9s} | "
              f"{str(s_.get('boundaries','-')):>9s} {s_.get('agreement','-'):>4} "
              f"{s_.get('range',0):6.3f} {g(s_):>9s}")

    anc = recs.get(ANCHOR)
    if anc:
        d_ = abs(anc["act_mid_sep"] - ANCHOR_TARGET)
        out["anchor_gate"] = {"status": "PASS" if d_ <= ANCHOR_TOL else "FAIL",
                              "raw_act_mid_sep": anc["act_mid_sep"],
                              "std_act_mid_sep": anc.get("std_act_mid_sep"),
                              "target": ANCHOR_TARGET, "abs_diff": d_,
                              "tolerance_used_fraction": d_ / ANCHOR_TOL}
        print(f"\nANCHOR {ANCHOR}: raw mid_sep {anc['act_mid_sep']:+.4f} vs target "
              f"{ANCHOR_TARGET} -> {out['anchor_gate']['status']} "
              f"(used {100*d_/ANCHOR_TOL:.0f}% of tolerance); "
              f"standardised mid_sep {anc.get('std_act_mid_sep'):+.4f}")

    out["verdicts"] = {}
    print()
    for (kind, gate), rows in variants.items():
        v = verdict_for(rows, f"{kind} map, {gate} gate")
        out["verdicts"][f"{kind}_{gate}"] = {k: x for k, x in v.items() if k != "null_draws"}
        print(f"{v['label']:28s} k={v['n_usable']:2d}  "
              + (f"obs={v['obs_mean_agreement_norm']:.4f} null={v['pooled_null_median']:.4f} "
                 f"p={v['p_one_sided']:.4f}  {v['n_beating_own_null_median']}/{v['n_usable']} beat  "
                 f"-> {v['verdict']}" if "p_one_sided" in v else f"-> {v['verdict']}"))

    vs = {k: v["verdict"] for k, v in out["verdicts"].items() if "verdict" in v}
    out["all_verdicts_agree"] = len(set(vs.values())) == 1
    out["primary_verdict"] = out["verdicts"].get("raw_new", {}).get("verdict")
    print(f"\nAll four cells agree: {out['all_verdicts_agree']}")
    print(f"PRIMARY (raw map, repaired gate): {out['primary_verdict']}")
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print("TESTB2_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
