"""Corpus-dependence analysis. Design frozen in PREREG.md.

For each model, compare three lenses fitted with an identical recipe:
  wiki_a (WikiText seed 0) | wiki_b (WikiText seed 1) | code (codeparrot seed 0)

The SEED NULL (wiki_a vs wiki_b) is the reference: any two finite-sample fits differ, so a
corpus effect only means something measured against ordinary fitting variation.

Measures per pair, on the shared-vocabulary probe:
  boundary_shift  |b1_x - b1_y| + |b2_x - b2_y|   (layers)
  map_distance    1 - CKA between the two lenses' layer-by-layer maps
  band_shift      |mid_sep_x - mid_sep_y|

CORRECTION (2026-09-05). The first version of this analyzer built each layer's d x d readout
covariance G_l = J_l^T M J_l and scored a layer pair with <G_i, G_j>_F / (|G_i| |G_j|). That is a
cosine between two SELF-covariances, not linear CKA: CKA of the readout geometries D_l = U_c J_l
is |D_j^T D_i|_F^2 / (|D_i^T D_i|_F |D_j^T D_j|_F) with D_j^T D_i = J_j^T M J_i, i.e. it needs the
CROSS-gram. The two statistics give very different maps for the same lens (gpt2-small: off-diagonal
range [0.94, 1.00] under CKA, [0.03, 0.82] under the old formula), so the fitted boundaries and
map distances reported before this date were computed on a map that is not the atlas's map and
not the map PREREG.md names. `cka_from_readout` below is the pre-registered statistic and is
checked against `common.cka.linear_cka` on real geometries at run time. The old formula is kept
as `legacy_selfgram_similarity` behind `--legacy` so the superseded ledger numbers remain
regenerable; do not use it for new results.
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


def cka_from_readout(Js, M):
    """Layer x layer LINEAR CKA of the readout geometries D_l = U_c J_l, from the Jacobians and
    the centered probe covariance M = U_c^T U_c, without forming D:
        CKA(D_i, D_j) = |J_j^T M J_i|_F^2 / (|J_i^T M J_i|_F |J_j^T M J_j|_F).
    Identical to common.cka.linear_cka(D_i, D_j); the atlas, shared_maps.py and every other map in
    this campaign use that statistic."""
    n = len(Js); MJ = [M @ J for J in Js]
    self_norm = [np.linalg.norm(Js[i].T @ MJ[i], "fro") for i in range(n)]
    C = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            cross = np.linalg.norm(Js[j].T @ MJ[i], "fro") ** 2
            C[i, j] = C[j, i] = float(cross / (self_norm[i] * self_norm[j]))
    return C


def legacy_selfgram_similarity(Gs):
    """SUPERSEDED 2026-09-05, see module docstring. Cosine between d x d self-covariances
    G_l = J_l^T M J_l. NOT linear CKA. Kept only so the pre-correction numbers can be regenerated
    with --legacy."""
    n = len(Gs); nrm = [np.linalg.norm(g, "fro") for g in Gs]; C = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            C[i, j] = C[j, i] = float(np.sum(Gs[i] * Gs[j]) / (nrm[i] * nrm[j]))
    return C


def probe_UM(slug, hf_id, d):
    """Centered shared-probe rows U_c (n x d) and M = U_c^T U_c, built exactly as
    run_geometry_causality.probe_M builds M; U_c is kept so the CKA identity can be checked."""
    import run_geometry_causality as rg
    strings = json.load(open(rg.SHARED_TOKENS))["strings"]
    ids = rg.resolve_ids_inline(hf_id, strings)
    idlist = [ids[s] for s in strings if s in ids]
    Us, dd = rg.probe_rows_inline(hf_id, idlist)
    assert dd == d, f"probe d {dd} != model d {d}"
    Uc = (Us - Us.mean(0, keepdims=True)).astype(np.float32)
    return Uc, (Uc.T @ Uc).astype(np.float32), len(idlist)


def check_cka_identity(Js, M, Uc, C, tol=1e-4):
    """Assert the cross-gram CKA equals linear_cka on explicit geometries for the first pair."""
    from common.cka import linear_cka
    D0, D1 = Uc @ Js[0], Uc @ Js[1]
    ref = linear_cka(D0, D1)
    assert abs(ref - C[0, 1]) < tol, f"CKA identity check failed: {ref:.6f} vs {C[0, 1]:.6f}"
    return ref


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
    ap.add_argument("--legacy", action="store_true",
                    help="reproduce the SUPERSEDED pre-2026-09-05 self-gram numbers (not CKA)")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE.parents[1]))   # for common.cka

    out = {"_statistic": ("legacy self-gram cosine (SUPERSEDED, not CKA)" if a.legacy else
                          "linear CKA of shared-probe readout geometry (corrected 2026-09-05)")}
    for slug, (hf_id, pfx) in MODELS.items():
        try:
            maps, seps, bnds = {}, {}, {}
            Uc = Mprobe = None
            for arm in ARMS:
                d = torch.load(Path(a.fits) / f"{pfx}_{arm}.pt", map_location="cpu",
                               weights_only=False)
                J = d["J"]; layers = sorted(J.keys())
                if Mprobe is None:
                    Uc, Mprobe, nprobe = probe_UM(slug, hf_id, d["d_model"])
                Js = [J[l].float().numpy() for l in layers]
                if a.legacy:
                    C = legacy_selfgram_similarity([(Jl.T @ Mprobe @ Jl).astype(np.float32)
                                                    for Jl in Js])
                else:
                    C = cka_from_readout(Js, Mprobe)
                    ref = check_cka_identity(Js, Mprobe, Uc, C)
                    if arm == "wiki_a":
                        print(f"   [{slug}] CKA identity check: linear_cka(D0,D1)={ref:.6f} "
                              f"cross-gram={C[0, 1]:.6f}", flush=True)
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

    ok = {k: v for k, v in out.items() if not k.startswith("_") and "error" not in v}
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
