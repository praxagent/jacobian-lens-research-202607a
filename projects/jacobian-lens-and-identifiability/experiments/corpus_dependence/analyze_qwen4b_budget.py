"""qwen3-4b budget test: does the whole map converge as early as the band statistic?
Design frozen in PREREG_QWEN4B_BUDGET.md (2026-09-08) before any fit existed.

Fits (fit_our_own/fit_lens.py, WikiText-103, --max-seq-len 128 --match-length --dim-batch 8):
  <pfx>_wiki_a.pt  100 prompts, seed 0   reference
  <pfx>_wiki_b.pt  100 prompts, seed 1   seed null
  <pfx>_n8 / _n16 / _n24 / _n48.pt       seed 0; nested prefixes of wiki_a

Measures, all on the shared-vocabulary probe with the corrected statistic (analyze.py helpers,
identity-checked against common.cka.linear_cka): map distance to the reference (1 - CKA of the
off-diagonal profiles), fitted boundaries and fitted band separation, thirds mid_sep (the release
note's statistic), and the distance of every fit to the public Neuronpedia shared map.

Frozen bars:
  map converged at n   iff  map_distance(n, ref) <= 2 x seed-null map distance
  band converged at n  iff  |mid_sep(n) - mid_sep(ref)|    <= 2 x max(seed mid_sep shift, 0.002)
                       and  |fitted_sep(n) - fitted_sep(ref)| <= 2 x max(seed fitted_sep shift, 0.002)

    analyze_qwen4b_budget.py --fits fits/
    analyze_qwen4b_budget.py --fits fits/ --slug gpt2-small --hf-id openai-community/gpt2 \
        --pfx gpt2 --budgets 25 50 --out /tmp/smoke.json      # mechanics smoke on existing fits
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ATLAS = HERE.parents[1] / "experiments/jspace_atlas"
sys.path.insert(0, str(ATLAS)); sys.path.insert(0, str(HERE.parents[1]))
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("cd_analyze", HERE / "analyze.py")
_cd = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_cd)

PREREG = "PREREG_QWEN4B_BUDGET.md"
BUDGETS = [8, 16, 24, 48]
REFERENCE = 100
MAP_FACTOR = 2.0
BAND_FACTOR = 2.0
BAND_FLOOR = 0.002


def load_map(path, Uc, M):
    """Map for one fit; cached to maps/cache/<fit>.<n_probe>.npz (the float64 Gram path takes minutes per
    map on this box and the harness kills long background jobs when memory is tight)."""
    cache = HERE / "maps" / "cache" / f"{Path(path).stem}.{Uc.shape[0]}.f64.npz"
    if cache.exists():
        z = np.load(cache); return z["C"], float(z["ident"])
    d = torch.load(path, map_location="cpu", weights_only=False)
    J = d["J"]; layers = sorted(J.keys())
    if d["d_model"] >= 2048:
        Js = [(lambda l=l: J[l].float().numpy()) for l in layers]   # never all layers in fp32 at once
    else:
        Js = [J[l].float().numpy() for l in layers]
    C = _cd.cka_from_readout(Js, M, Uc_for_gram=Uc)
    ident = _cd.check_cka_identity(Js, M, Uc, C)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, C=C, ident=ident, statistic=np.array("linear CKA, float64 Gram path"))
    print(f"  cached {cache.name}  identity {ident:.6f} vs {C[0,1]:.6f}", flush=True)
    return C, float(ident)


def map_distance(Cx, Cy):
    iu = np.triu_indices_from(Cx, 1)
    vx, vy = Cx[iu] - Cx[iu].mean(), Cy[iu] - Cy[iu].mean()
    return 1.0 - float((vx @ vy) ** 2 / ((vx @ vx) * (vy @ vy)))


def stats(C):
    b1, b2, fsep = _cd.fitted_boundaries(C)
    return {"boundaries": [b1, b2], "fitted_sep": round(fsep, 5), "mid_sep": round(_cd.band_stats(C), 5)}


def bshift(s, t):
    return abs(s["boundaries"][0] - t["boundaries"][0]) + abs(s["boundaries"][1] - t["boundaries"][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True, help="dir with <pfx>_wiki_a/_wiki_b/_n<budget>.pt")
    ap.add_argument("--slug", default="qwen3-4b")
    ap.add_argument("--hf-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--pfx", default="q4b")
    ap.add_argument("--budgets", type=int, nargs="+", default=BUDGETS)
    ap.add_argument("--out", default=str(HERE / "results_qwen4b_budget.json"))
    a = ap.parse_args()
    fits = Path(a.fits)

    d0 = torch.load(fits / f"{a.pfx}_wiki_a.pt", map_location="cpu", weights_only=False)
    Uc, M, nprobe = _cd.probe_UM(a.slug, a.hf_id, d0["d_model"]); del d0
    tensor = getattr(_cd.probe_UM, "last_tensor", None)
    Cref, ident = load_map(fits / f"{a.pfx}_wiki_a.pt", Uc, M)
    Cb, _ = load_map(fits / f"{a.pfx}_wiki_b.pt", Uc, M)
    sref, sb = stats(Cref), stats(Cb)
    seed_null = {"map_distance": map_distance(Cref, Cb), "boundary_shift": bshift(sref, sb),
                 "mid_sep_shift": abs(sref["mid_sep"] - sb["mid_sep"]),
                 "fitted_sep_shift": abs(sref["fitted_sep"] - sb["fitted_sep"])}
    bars = {"map": MAP_FACTOR * seed_null["map_distance"],
            "mid_sep": BAND_FACTOR * max(seed_null["mid_sep_shift"], BAND_FLOOR),
            "fitted_sep": BAND_FACTOR * max(seed_null["fitted_sep_shift"], BAND_FLOOR)}

    public = None; Cp = None
    pub = _cd.SHARED_MAPS / f"{a.slug}.npz"
    if pub.exists():
        z = np.load(pub)
        if z["cka"].shape == Cref.shape:
            Cp = z["cka"]
            public = {"file": str(pub.relative_to(HERE.parents[1])),
                      "stats_same_functions": stats(Cp),
                      "npz_mid_sep": float(z["mid_sep"]) if "mid_sep" in z.files else None,
                      "npz_fitted_sep": float(z["fitted_sep"]) if "fitted_sep" in z.files else None,
                      "ref_vs_public_map_distance": map_distance(Cref, Cp),
                      "wiki_b_vs_public_map_distance": map_distance(Cb, Cp)}

    rows = {}
    for n in a.budgets:
        p = fits / f"{a.pfx}_n{n}.pt"
        if not p.exists():
            print(f"n={n}: MISSING {p.name}", flush=True); continue
        C, _ = load_map(p, Uc, M); s = stats(C)
        md = map_distance(Cref, C)
        ms, fs = abs(s["mid_sep"] - sref["mid_sep"]), abs(s["fitted_sep"] - sref["fitted_sep"])
        rows[str(n)] = {**s, "map_distance_to_ref": md,
                        "ratio_to_seed_null": md / max(seed_null["map_distance"], 1e-12),
                        "boundary_shift": bshift(s, sref), "mid_sep_shift": ms, "fitted_sep_shift": fs,
                        "map_converged": bool(md <= bars["map"]),
                        "band_converged": bool(ms <= bars["mid_sep"] and fs <= bars["fitted_sep"]),
                        "public_map_distance": map_distance(C, Cp) if Cp is not None else None}

    def conv(n, key):
        r = rows.get(str(n)); return None if r is None else r[key]

    need = [n for n in (16, 24, 48) if str(n) not in rows]
    if need:
        statement = f"INCOMPLETE: budgets {need} missing; no statement"
    elif not (conv(16, "band_converged") and conv(24, "band_converged")):
        statement = ("P1 FAILS: the band statistic is not converged at 16 and 24 under this recipe; "
                     "the July result does not replicate; reported as such")
    elif conv(24, "map_converged"):
        statement = ("on qwen3-4b the whole map is inside the seed null by 24 prompts; the release "
                     "note's parenthetical extends to the map on this model; the 397B caveat is "
                     "weakened, not discharged")
    elif conv(48, "map_converged"):
        statement = ("the band converges before the map: 24 prompts is band-converged and not "
                     "map-converged on qwen3-4b; the caveat on the 24-prompt 397B map stands")
    else:
        statement = ("the map is not converged by 48 prompts on a 4B model; the caveat is strengthened")

    predictions = {
        "P1_band_converged_16_24_48_not_8": (all(conv(n, "band_converged") for n in (16, 24, 48))
                                            and conv(8, "band_converged") is False) if not need else None,
        "P2_map_not_16_24_yes_48": (conv(16, "map_converged") is False and conv(24, "map_converged") is False
                                    and conv(48, "map_converged") is True) if not need else None,
        "anchor_ref_within_0.01": (public["ref_vs_public_map_distance"] <= 0.01) if public else None}

    out = {"prereg": PREREG, "slug": a.slug, "hf_id": a.hf_id, "reference_budget": REFERENCE,
           "budgets": a.budgets, "n_layers": int(Cref.shape[0]), "n_probe": nprobe,
           "unembedding_tensor": tensor, "cka_identity_check_pair01": ident,
           "statistic": "linear CKA of shared-probe readout geometry (cka_from_readout)",
           "reference": sref, "wiki_b": sb, "seed_null": seed_null, "bars": bars,
           "public_lens": public, "by_budget": rows, "predictions": predictions, "statement": statement}
    Path(a.out).write_text(json.dumps(out, indent=1))

    print(f"{a.slug}  L={out['n_layers']}  probe rows from {tensor} (n={nprobe})  identity check {ident:.6f}")
    print(f"  reference n={REFERENCE}: boundaries={sref['boundaries']} fitted_sep={sref['fitted_sep']} mid_sep={sref['mid_sep']}")
    print(f"  seed null: map {seed_null['map_distance']:.6f}  bshift {seed_null['boundary_shift']}  "
          f"mid {seed_null['mid_sep_shift']:.5f}  fitted {seed_null['fitted_sep_shift']:.5f}")
    print(f"  bars: map<={bars['map']:.6f}  mid<={bars['mid_sep']:.5f}  fitted<={bars['fitted_sep']:.5f}")
    if public:
        ps = public["stats_same_functions"]
        print(f"  public lens: boundaries={ps['boundaries']} fitted_sep={ps['fitted_sep']} mid_sep={ps['mid_sep']} "
              f"(npz mid {public['npz_mid_sep']}, fitted {public['npz_fitted_sep']}); ref vs public {public['ref_vs_public_map_distance']:.5f}")
    for n in a.budgets:
        r = rows.get(str(n))
        if r is None: continue
        print(f"  n={n:<3d} map {r['map_distance_to_ref']:.6f} ({r['ratio_to_seed_null']:5.1f}x null) "
              f"{'MAP-OK ' if r['map_converged'] else 'map-no '} bshift {r['boundary_shift']:<2d} "
              f"mid {r['mid_sep']:.4f} (d {r['mid_sep_shift']:.4f}) fitted {r['fitted_sep']:.4f} (d {r['fitted_sep_shift']:.4f}) "
              f"{'BAND-OK' if r['band_converged'] else 'band-no'}"
              + (f"  public {r['public_map_distance']:.5f}" if r['public_map_distance'] is not None else ""))
    print(f"  predictions: {predictions}")
    print(f"  STATEMENT: {statement}")
    print("QWEN4B_BUDGET_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
