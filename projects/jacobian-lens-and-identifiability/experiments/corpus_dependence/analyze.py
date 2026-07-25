"""Corpus-dependence analysis. Design frozen in PREREG.md.

For each model, compare three lenses fitted with an identical recipe:
  wiki_a (WikiText seed 0) | wiki_b (WikiText seed 1) | code (codeparrot seed 0)

The SEED NULL (wiki_a vs wiki_b) is the reference: any two finite-sample fits differ, so a
corpus effect only means something measured against ordinary fitting variation.

Measures per pair, on the shared-vocabulary probe:
  boundary_shift  |b1_x - b1_y| + |b2_x - b2_y|   (layers)
  map_distance    1 - CKA between the two lenses' layer-by-layer maps
  band_shift      |mid_sep_x - mid_sep_y|
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
    "qwen3.5-0.8b": ("Qwen/Qwen3.5-0.8B", "q08b"),
}
ARMS = ["wiki_a", "wiki_b", "code"]


def cka_from_grams(Gs):
    n = len(Gs); nrm = [np.linalg.norm(g, "fro") for g in Gs]; C = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            C[i, j] = C[j, i] = float(np.sum(Gs[i] * Gs[j]) / (nrm[i] * nrm[j]))
    return C


def band_stats(M):
    L = M.shape[0]; th = np.array_split(np.arange(L), 3)
    blk = lambda a, b: float(np.mean([M[i, j] for i in a for j in b if i != j]) or 1.0)
    e, m, l = th
    return blk(m, m) - 0.5 * (blk(e, m) + blk(m, l))


def fitted_boundaries(M):
    from atlas_stage_a import fitted_seg
    b1, b2, sep = fitted_seg(M)
    return int(b1), int(b2), float(sep)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fits", required=True)
    ap.add_argument("--out", default=str(HERE / "results.json"))
    a = ap.parse_args()
    import run_geometry_causality as rg

    out = {}
    for slug, (hf_id, pfx) in MODELS.items():
        try:
            maps, seps, bnds = {}, {}, {}
            Mprobe = None
            for arm in ARMS:
                d = torch.load(Path(a.fits) / f"{pfx}_{arm}.pt", map_location="cpu",
                               weights_only=False)
                J = d["J"]; layers = sorted(J.keys())
                if Mprobe is None:
                    Mprobe, nprobe = rg.probe_M(slug, hf_id, d["d_model"])
                # readout gram per layer: G_l = J^T M J  (the object CKA compares)
                Gs = []
                for l in layers:
                    Jl = J[l].float().numpy()
                    Gs.append((Jl.T @ Mprobe @ Jl).astype(np.float32))
                C = cka_from_grams(Gs)
                maps[arm] = C; seps[arm] = band_stats(C); bnds[arm] = fitted_boundaries(C)
            def cmp(x, y):
                Cx, Cy = maps[x], maps[y]
                iu = np.triu_indices_from(Cx, 1)
                vx, vy = Cx[iu], Cy[iu]
                # CKA between the two maps' off-diagonal profiles
                vx0, vy0 = vx - vx.mean(), vy - vy.mean()
                cka = float((vx0 @ vy0) ** 2 / ((vx0 @ vx0) * (vy0 @ vy0)))
                return {"boundary_shift": abs(bnds[x][0] - bnds[y][0]) + abs(bnds[x][1] - bnds[y][1]),
                        "map_distance": 1.0 - cka,
                        "band_shift": abs(seps[x] - seps[y])}
            out[slug] = {"n_layers": maps["wiki_a"].shape[0], "n_probe": nprobe,
                         "band_sep": {k: round(v, 4) for k, v in seps.items()},
                         "boundaries": {k: list(v[:2]) for k, v in bnds.items()},
                         "seed_null": cmp("wiki_a", "wiki_b"),
                         "corpus": cmp("wiki_a", "code")}
            sn, co = out[slug]["seed_null"], out[slug]["corpus"]
            print(f"{slug:14s} L={out[slug]['n_layers']}")
            print(f"   boundaries wiki_a={bnds['wiki_a'][:2]} wiki_b={bnds['wiki_b'][:2]} code={bnds['code'][:2]}")
            print(f"   band_sep   {out[slug]['band_sep']}")
            print(f"   SEED NULL  shift={sn['boundary_shift']} map_dist={sn['map_distance']:.4f} band={sn['band_shift']:.4f}")
            print(f"   CORPUS     shift={co['boundary_shift']} map_dist={co['map_distance']:.4f} band={co['band_shift']:.4f}")
            print(f"   -> corpus {'EXCEEDS' if co['map_distance'] > sn['map_distance'] else 'within'} seed null", flush=True)
        except Exception as e:
            print(f"{slug}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
            out[slug] = {"error": f"{type(e).__name__}"}

    ok = {k: v for k, v in out.items() if "error" not in v}
    if ok:
        exceeds = sum(1 for v in ok.values() if v["corpus"]["map_distance"] > v["seed_null"]["map_distance"])
        verdict = ("CORPUS MATTERS" if exceeds == len(ok) else
                   "ROBUST" if exceeds == 0 else "MIXED")
        out["_verdict"] = {"models_ok": len(ok), "corpus_exceeds_null_in": exceeds, "verdict": verdict}
        print(f"\nVERDICT: {verdict} ({exceeds}/{len(ok)} models show corpus > seed null)")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("CORPUS_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
