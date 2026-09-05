"""Fit-budget analysis. Design frozen in PREREG_FITBUDGET.md before any fit existed.

Refits of gpt2-small and gemma-3-270m on WikiText seed 0 at budgets 25/50/200/400/1000 prompts,
compared to the 100-prompt reference (`*_wiki_a.pt`, from the corpus experiment) with the
identical measures used there. The SEED NULL from that experiment is the reference scale.

Frozen decision rules (PREREG_FITBUDGET.md):
  distance at 200 and 400 within ~2x the seed null   -> CONVERGED
  distance still falling at 400                      -> NOT CONVERGED
  25-prompt map far from the rest                    -> the 24-prompt 397B lens is caveated
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ATLAS = HERE.parents[1] / "experiments/jspace_atlas"
GC = HERE.parents[1] / "experiments/geometry_causality"
sys.path.insert(0, str(ATLAS)); sys.path.insert(0, str(GC))

MODELS = {
    "gpt2-small":   ("openai-community/gpt2", "gpt2"),
    "gemma-3-270m": ("google/gemma-3-270m", "g270m"),
}
BUDGETS = [25, 50, 200, 400, 1000]   # 1000 added by PREREG_FITBUDGET.md Amendment 1
REFERENCE = 100                      # the corpus experiment's wiki_a fit
CONVERGED_FACTOR = 2.0               # frozen: "within ~2x the seed null"

# import THIS directory's analyze.py by path: geometry_causality/ has a same-named module
# that runs work at import time, and it is earlier on sys.path.
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("cd_analyze", HERE / "analyze.py")
_cd = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_cd)
band_stats, fitted_boundaries = _cd.band_stats, _cd.fitted_boundaries
# CORRECTION 2026-09-05: the map statistic is linear CKA of the shared-probe readout geometry
# (cka_from_readout), the same statistic as the atlas. The first run of this analyzer used the
# self-gram cosine now kept as legacy_selfgram_similarity; see analyze.py's module docstring.
LEGACY = False


def load_map(path: Path, Mprobe):
    d = torch.load(path, map_location="cpu", weights_only=False)
    J = d["J"]; layers = sorted(J.keys())
    Js = [J[l].float().numpy() for l in layers]
    if LEGACY:
        return _cd.legacy_selfgram_similarity([(Jl.T @ Mprobe @ Jl).astype(np.float32)
                                               for Jl in Js]), d["d_model"]
    return _cd.cka_from_readout(Js, Mprobe), d["d_model"]


def map_distance(Cx, Cy):
    iu = np.triu_indices_from(Cx, 1)
    vx, vy = Cx[iu] - Cx[iu].mean(), Cy[iu] - Cy[iu].mean()
    return 1.0 - float((vx @ vy) ** 2 / ((vx @ vx) * (vy @ vy)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True, help="dir with <pfx>_n<budget>.pt")
    ap.add_argument("--ref-fits", required=True, help="dir with <pfx>_wiki_a.pt (100 prompts)")
    ap.add_argument("--corpus-results", default=str(HERE / "results.json"))
    ap.add_argument("--out", default=str(HERE / "results_fitbudget.json"))
    a = ap.parse_args()
    import run_geometry_causality as rg

    corpus = json.loads(Path(a.corpus_results).read_text())
    out = {"reference_budget": REFERENCE, "budgets": BUDGETS,
           "converged_factor": CONVERGED_FACTOR, "models": {}}

    for slug, (hf_id, pfx) in MODELS.items():
        try:
            seed_null = corpus[slug]["seed_null"]["map_distance"]
            ref_path = Path(a.ref_fits) / f"{pfx}_wiki_a.pt"
            d0 = torch.load(ref_path, map_location="cpu", weights_only=False)
            Mprobe, nprobe = rg.probe_M(slug, hf_id, d0["d_model"])
            del d0
            Cref, _ = load_map(ref_path, Mprobe)
            bref = fitted_boundaries(Cref); sref = band_stats(Cref)

            rows = {}
            for n in BUDGETS:
                p = Path(a.fits) / f"{pfx}_n{n}.pt"
                if not p.exists():
                    print(f"{slug} n={n}: MISSING {p.name}", flush=True); continue
                C, _ = load_map(p, Mprobe)
                b = fitted_boundaries(C); s = band_stats(C)
                rows[str(n)] = {
                    "map_distance_to_ref": map_distance(Cref, C),
                    "boundary_shift": abs(b[0] - bref[0]) + abs(b[1] - bref[1]),
                    "band_shift": abs(s - sref),
                    "boundaries": list(b[:2]), "band_sep": round(s, 4),
                    "ratio_to_seed_null": map_distance(Cref, C) / max(seed_null, 1e-12)}

            m = {"n_layers": int(Cref.shape[0]), "n_probe": nprobe,
                 "seed_null_map_distance": seed_null,
                 "reference_boundaries": list(bref[:2]),
                 "reference_band_sep": round(sref, 4), "by_budget": rows}

            # frozen rules
            HIGH = tuple(b for b in BUDGETS if b >= 200)
            hi = [rows[str(n)]["ratio_to_seed_null"] for n in HIGH if str(n) in rows]
            m["converged"] = bool(hi) and all(r <= CONVERGED_FACTOR for r in hi)
            have = sorted(b for b in BUDGETS if str(b) in rows)
            if len(have) >= 2:
                top, prev = str(have[-1]), str(have[-2])
                m["still_falling_at_top"] = (rows[top]["map_distance_to_ref"]
                                             < rows[prev]["map_distance_to_ref"] * 0.8)
            m["low_budget_far"] = ("25" in rows and rows["25"]["ratio_to_seed_null"]
                                   > CONVERGED_FACTOR)
            out["models"][slug] = m

            print(f"\n{slug}  L={m['n_layers']}  ref(n={REFERENCE}) boundaries={m['reference_boundaries']} "
                  f"band_sep={m['reference_band_sep']}  seed_null={seed_null:.6f}")
            for n in BUDGETS:
                r = rows.get(str(n))
                if r is None: continue
                print(f"   n={n:<4d} map_dist={r['map_distance_to_ref']:.6f} "
                      f"({r['ratio_to_seed_null']:6.1f}x null)  bshift={r['boundary_shift']:<3d} "
                      f"band={r['band_sep']:.4f}  boundaries={r['boundaries']}", flush=True)
            print(f"   -> converged at {'/'.join(str(b) for b in HIGH)}: {m['converged']};  "
                  f"25-prompt far: {m['low_budget_far']}", flush=True)
        except Exception as e:
            print(f"{slug}: FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
            out["models"][slug] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}

    ok = {k: v for k, v in out["models"].items() if "error" not in v}
    if ok:
        out["verdict"] = ("CONVERGED" if all(v["converged"] for v in ok.values()) else
                          "NOT CONVERGED")
        out["low_budget_caveat"] = any(v["low_budget_far"] for v in ok.values())
        print(f"\nVERDICT: {out['verdict']}  "
              f"(25-prompt caveat needed: {out['low_budget_caveat']})")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("FITBUDGET_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
