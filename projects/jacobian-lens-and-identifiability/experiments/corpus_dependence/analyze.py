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
    # 8B extension, PREREG_8B.md (2026-09-05): untied model, so the probe rows MUST come from
    # lm_head.weight; the shared atlas tool's loader picks it (probe_rows_inline would not).
    "llama3.1-8b":  ("meta-llama/Llama-3.1-8B", "llama8b"),
}
SHARED_MAPS = HERE.parents[1] / "experiments/jspace_atlas/atlas_out/shared_maps"
ARMS = ["wiki_a", "wiki_b", "code"]


def cka_from_readout(Js, M, Uc_for_gram=None):
    """Layer x layer LINEAR CKA of the readout geometries D_l = U_c J_l, from the Jacobians and
    the centered probe covariance M = U_c^T U_c, without forming D:
        CKA(D_i, D_j) = |J_j^T M J_i|_F^2 / (|J_i^T M J_i|_F |J_j^T M J_j|_F).
    Identical to common.cka.linear_cka(D_i, D_j); the atlas, shared_maps.py and every other map in
    this campaign use that statistic."""
    n = len(Js); d = Js[0].shape[0] if hasattr(Js[0], "shape") else None
    if (d is None or d >= 2048) and Uc_for_gram is not None:
        # Wide model: the cross-gram route costs O(n_pairs * d^3). Equivalent and ~15x cheaper: token
        # Grams K_l = D_l D_l^T with D_l = U_c J_l (U_c has zero column means, so K is centered), then
        # CKA(i, j) = <K_i, K_j>_F / (|K_i|_F |K_j|_F). The Grams (n_layers x n_probe^2 float32, 2 GB
        # for 31 layers) are spilled to memory-mapped files so the 7.6 GB box never holds them all;
        # `Js` may be a list of callables that materialise one layer at a time.
        import tempfile, os
        tmp = tempfile.mkdtemp(prefix="cka_grams_")
        nrm = []
        for l in range(n):
            J = Js[l]() if callable(Js[l]) else Js[l]
            D = (Uc_for_gram @ J).astype(np.float32); del J
            K = np.memmap(os.path.join(tmp, f"K{l}.f32"), dtype=np.float32, mode="w+", shape=(D.shape[0], D.shape[0]))
            K[:] = D @ D.T; nrm.append(float(np.linalg.norm(K, "fro"))); K.flush(); del K, D
        C = np.eye(n)
        for i in range(n):
            Ki = np.memmap(os.path.join(tmp, f"K{i}.f32"), dtype=np.float32, mode="r", shape=(Uc_for_gram.shape[0],) * 2)
            for j in range(i + 1, n):
                Kj = np.memmap(os.path.join(tmp, f"K{j}.f32"), dtype=np.float32, mode="r", shape=Ki.shape)
                C[i, j] = C[j, i] = float(np.sum(Ki * Kj, dtype=np.float64) / (nrm[i] * nrm[j])); del Kj
            del Ki
        for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)
        return C
    MJ = [M @ J for J in Js]
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
    """Centered shared-probe rows U_c (n x d) and M = U_c^T U_c. Rows come from the model's
    READOUT tensor: lm_head.weight when the model has one (untied, e.g. llama3.1-8b), else the tied
    embedding. Uses the atlas tool's loader (tools/jlens_atlas), which reads only those rows from
    the safetensors shard; the tensor name is returned so the ledger can record it."""
    sys.path.insert(0, str(HERE.parents[1] / "tools/jlens_atlas"))
    from jlens_atlas import io as jio
    strings = json.load(open(jio.SHARED_TOKENS))["strings"]
    ids = jio.resolve_shared_ids(hf_id, strings)
    idlist = [ids[s] for s in strings if s in ids]
    Us, tensor_name = jio.unembedding_rows(hf_id, idlist)
    assert Us.shape[1] == d, f"probe d {Us.shape[1]} != model d {d}"
    Uc = (Us - Us.mean(0, keepdims=True)).astype(np.float32)
    probe_UM.last_tensor = tensor_name
    return Uc, (Uc.T @ Uc).astype(np.float32), len(idlist)


def check_cka_identity(Js, M, Uc, C, tol=1e-4):
    """Assert the cross-gram CKA equals linear_cka on explicit geometries for the first pair."""
    from common.cka import linear_cka
    J0 = Js[0]() if callable(Js[0]) else Js[0]; J1 = Js[1]() if callable(Js[1]) else Js[1]
    D0, D1 = Uc @ J0, Uc @ J1
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
                if d["d_model"] >= 2048:
                    # wide model: never hold float32 copies of all layers at once
                    Js = [(lambda l=l: J[l].float().numpy()) for l in layers]
                else:
                    Js = [J[l].float().numpy() for l in layers]
                if a.legacy:
                    C = legacy_selfgram_similarity([(Jl.T @ Mprobe @ Jl).astype(np.float32)
                                                    for Jl in Js])
                else:
                    C = cka_from_readout(Js, Mprobe, Uc_for_gram=Uc)
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
            (HERE / "maps").mkdir(exist_ok=True)
            np.savez_compressed(HERE / "maps" / f"{slug}.npz", **{k: v.astype(np.float32) for k, v in maps.items()},
                                statistic=np.array(out["_statistic"]))
            # PREREG_8B anchor: our wiki_a map vs the public Neuronpedia shared map of the same model
            anchor = None
            pub = SHARED_MAPS / f"{slug}.npz"
            if pub.exists():
                Cp = np.load(pub)["cka"]
                if Cp.shape == maps["wiki_a"].shape:
                    iu = np.triu_indices_from(Cp, 1); vx, vy = Cp[iu], maps["wiki_a"][iu]
                    vx0, vy0 = vx - vx.mean(), vy - vy.mean()
                    anchor = {"public_vs_wiki_a_map_distance": 1.0 - float((vx0 @ vy0) ** 2 / ((vx0 @ vx0) * (vy0 @ vy0))),
                              "public_boundaries": list(fitted_boundaries(Cp)[:2])}
            out[slug] = {"n_layers": maps["wiki_a"].shape[0], "n_probe": nprobe,
                         "unembedding_tensor": getattr(probe_UM, "last_tensor", None),
                         "anchor_public_fit": anchor,
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
            print(f"   -> corpus {'EXCEEDS' if co['map_distance'] > sn['map_distance'] else 'within'} seed null"
                  f"   [probe rows from {out[slug]['unembedding_tensor']}; anchor {anchor}]", flush=True)
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
