"""Test B restricted to models whose fitted boundaries are IDENTIFIED. Rule frozen in
PREREG_B3_IDENT.md (2026-09-05) after the identifiability spreads had been seen and before this
analysis was run. Everything except the admission gate is analyze_B2.py unchanged.

Admission per cell: the model passes that cell's existing usability gate AND its shared-probe
lens boundaries AND that cell's activation boundaries (raw or standardised map) are identified
(both near-optimal spreads <= 0.25 at tol 0.05). Sensitivity at tol 0.15 and 0.35 is reported.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "jspace_atlas"))
from boundary_identifiability import near_optimal_spread, SPREAD_MAX   # noqa: E402
import analyze_B2 as B2                                                # noqa: E402

SHARED = HERE.parent / "jspace_atlas/atlas_out/shared_maps"


def spreads(M, tol):
    r = near_optimal_spread(M, tol)
    return None if r is None else (r[2], r[3])


def ident(M, tol):
    s = spreads(M, tol)
    return bool(s is not None and s[0] <= SPREAD_MAX and s[1] <= SPREAD_MAX), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "outB"))
    ap.add_argument("--out", default=str(HERE / "results_B3_ident.json"))
    a = ap.parse_args()
    out = {"prereg": "PREREG_B3_IDENT.md", "spread_max": SPREAD_MAX, "tols": [0.05, 0.15, 0.35],
           "models": {}, "verdicts": {}}
    recs = {}
    for f in sorted(glob.glob(str(Path(a.runs) / "*.json"))):
        if "smoke" in f or "probe" in f:
            continue
        d = json.loads(Path(f).read_text()); recs[d["slug"]] = d
    for tol in out["tols"]:
        variants = {(k, g): [] for k in ("raw", "standardised") for g in ("old", "new")}
        for slug, d in recs.items():
            L = d["n_layers"]
            sm = SHARED / f"{slug}.npz"
            if not sm.exists() or L < 8:
                out["models"].setdefault(slug, {})[f"tol{tol}"] = {"admitted": False, "why": "too small / no shared map"}
                continue
            Mlens = np.load(sm)["cka"]
            lens_ok, lens_sp = ident(Mlens, tol)
            lb = list(B2.fitted_seg(Mlens))
            act = np.load(Path(a.runs) / f"{slug}.npz")
            m = {"lens_boundaries": lb, "lens_spreads": lens_sp, "lens_identified": lens_ok}
            for kind, key, bkey, medkey, rk in (("raw", "cka", "act_boundaries", "act_offdiag_median", "act_range"),
                                                ("standardised", "cka_standardised", "std_act_boundaries",
                                                 "std_act_offdiag_median", "std_act_range")):
                if key not in act.files or bkey not in d:
                    continue
                act_ok, act_sp = ident(act[key], tol)
                ab = d[bkey]
                agree = abs(lb[0] - ab[0]) + abs(lb[1] - ab[1])
                nd = B2.null_agreement(lb[0], lb[1], L, ab, B2.NPERM)
                row = {"slug": slug, "n_layers": L, "agreement": agree,
                       "beats_null_median": bool(agree < np.median(nd)), "null_draws": nd}
                exo = d[medkey] >= B2.OLD_MEDIAN_GATE; exn = (d.get(rk) or 0) < B2.NEW_RANGE_GATE
                adm = lens_ok and act_ok
                m[kind] = {"act_boundaries": ab, "act_spreads": act_sp, "act_identified": act_ok,
                           "agreement": int(agree), "null_median": float(np.median(nd)),
                           "admitted_old": bool(adm and not exo), "admitted_new": bool(adm and not exn)}
                if adm and not exo: variants[(kind, "old")].append(row)
                if adm and not exn: variants[(kind, "new")].append(row)
            out["models"].setdefault(slug, {})[f"tol{tol}"] = m
        print(f"\n=== tol {tol}: admitted per cell ===")
        for (k, g), rows in variants.items():
            v = B2.verdict_for(rows, f"{k} map, {g} gate")
            out["verdicts"][f"tol{tol}_{k}_{g}"] = {x: y for x, y in v.items() if x != "null_draws"}
            names = ", ".join(r["slug"] for r in rows)
            if "p_one_sided" in v:
                print(f"  {v['label']:28s} k={v['n_usable']:2d} obs={v['obs_mean_agreement_norm']:.4f} "
                      f"null={v['pooled_null_median']:.4f} p={v['p_one_sided']:.4f} "
                      f"{v['n_beating_own_null_median']}/{v['n_usable']} beat -> {v['verdict']}   [{names}]")
            else:
                print(f"  {v['label']:28s} k={v['n_usable']} -> {v['verdict']}   [{names}]")
    print("\nper-model identifiability at tol 0.05 (lens shared-probe | raw act | std act):")
    for slug, m in out["models"].items():
        t = m.get("tol0.05", {})
        if "lens_spreads" not in t: print(f"  {slug:18s} {t.get('why')}"); continue
        def f(sp): return "n/a" if sp is None else f"{sp[0]:.2f}/{sp[1]:.2f}"
        print(f"  {slug:18s} lens {f(t['lens_spreads'])} {'ID' if t['lens_identified'] else '--'} | "
              f"raw {f(t.get('raw',{}).get('act_spreads'))} {'ID' if t.get('raw',{}).get('act_identified') else '--'} | "
              f"std {f(t.get('standardised',{}).get('act_spreads'))} {'ID' if t.get('standardised',{}).get('act_identified') else '--'}")
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print("TESTB3_IDENT_DONE")


if __name__ == "__main__":
    main()
