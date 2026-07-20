"""J-space atlas Stage B — cross-model structure (CPU; run on the warm-cache box).

Reloads each model's lens (cached), computes its per-layer token geometry on the
SHARED-vocabulary probe (real + scale-matched random null), then builds the 36x36
distance matrix (distance = 1 - median same-relative-depth CKA on each pair's common
resolved strings), a 2-D MDS embedding, and evaluates the two pre-registered predictions:
  P1 family clustering significant on the real matrix AND absent on the null;
  P2 the 397B's nearest neighbor is a Qwen-family model.

Sub-stage 1 (b1): per-model D_shared to disk (heavy: lens reloads).  --precompute
Sub-stage 2 (b2): matrices + MDS + permutation tests from the D_shared cache. --analyze
Both by default. Small result files land in atlas_out/stageB/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
JL = HERE.parent / "jacobian_lens"
sys.path.insert(0, str(JL))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.cka import linear_cka          # noqa: E402
from atlas_stage_a import load_lens, load_U, SHARED, SEED  # noqa: E402
from shared_vocab import resolve_ids       # noqa: E402

OUT = HERE / "atlas_out"
DS = OUT / "d_shared"          # per-model geometry cache (big; stays on the box)
BOUT = OUT / "stageB"          # small result files (rsync home)
STRINGS = SHARED["strings"]
FAMILY = {}                    # slug prefix -> family


def family_of(slug):
    for fam in ("gemma", "qwen", "llama", "olmo"):
        if slug.startswith(fam):
            return fam
    return "other"             # gpt2 / pythia / gpt-oss


def precompute_one(slug):
    real_p = DS / f"{slug}.real.npy"
    if real_p.exists():
        print(f"[{slug}] D_shared exists, skip", flush=True)
        return
    t0 = time.time()
    jac, layers, hf_id = load_lens(slug)
    U = load_U(slug, hf_id)
    ids_map = resolve_ids(hf_id, STRINGS)
    resolved = [i for i, s in enumerate(STRINGS) if s in ids_map]   # indices into 4096
    rows = U[[ids_map[STRINGS[i]] for i in resolved]].float()       # (n_res, d)
    del U
    gen = torch.Generator().manual_seed(SEED)
    real, null = [], []
    for l in layers:
        J = jac[l].float()
        real.append((rows @ J).to(torch.float16).numpy())
        R = torch.randn(J.shape, generator=gen) * J.std()
        null.append((rows @ R).to(torch.float16).numpy())
        del J, R
    np.save(real_p, np.stack(real))                # (n_layers, n_res, d) fp16
    np.save(DS / f"{slug}.null.npy", np.stack(null))
    (DS / f"{slug}.meta.json").write_text(json.dumps(
        {"slug": slug, "family": family_of(slug), "layers": layers,
         "resolved": resolved, "n_res": len(resolved), "d": int(jac[layers[0]].shape[0])}))
    print(f"[{slug}] D_shared {len(layers)}x{len(resolved)} ({time.time()-t0:.0f}s)",
          flush=True)


def cross_distance(A, B, kind):
    """1 - median over A-layers of CKA at matched relative depth, on common strings."""
    ma = json.loads((DS / f"{A}.meta.json").read_text())
    mb = json.loads((DS / f"{B}.meta.json").read_text())
    common = sorted(set(ma["resolved"]) & set(mb["resolved"]))
    if len(common) < 256:
        return np.nan
    ia = {s: k for k, s in enumerate(ma["resolved"])}
    ib = {s: k for k, s in enumerate(mb["resolved"])}
    ra = np.array([ia[s] for s in common]); rb = np.array([ib[s] for s in common])
    Da = np.load(DS / f"{A}.{kind}.npy", mmap_mode="r")
    Db = np.load(DS / f"{B}.{kind}.npy", mmap_mode="r")
    La, Lb = len(ma["layers"]), len(mb["layers"])
    ckas = []
    for la in range(La):
        lb = min(Lb - 1, int(round(la / max(La - 1, 1) * (Lb - 1))))
        X = Da[la][ra].astype(np.float32)
        Y = Db[lb][rb].astype(np.float32)
        ckas.append(linear_cka(X, Y))
    return 1.0 - float(np.median(ckas))


def perm_test(D, fams, n=10000, seed=0):
    """Observed = mean(between-family) - mean(within-family); permutation p (one-sided)."""
    rng = np.random.default_rng(seed)
    n_m = len(fams)
    iu = np.triu_indices(n_m, 1)
    d = D[iu]
    fa = np.array(fams)

    def stat(labels):
        same = labels[iu[0]] == labels[iu[1]]
        m = ~np.isnan(d)
        return float(np.mean(d[m & ~same]) - np.mean(d[m & same]))
    obs = stat(fa)
    ge = 0
    for _ in range(n):
        if stat(fa[rng.permutation(n_m)]) >= obs:
            ge += 1
    return obs, (ge + 1) / (n + 1)


def analyze(slugs):
    fams = [family_of(s) for s in slugs]
    n_m = len(slugs)
    res = {}
    for kind in ("real", "null"):
        D = np.full((n_m, n_m), np.nan)
        for i in range(n_m):
            for j in range(i + 1, n_m):
                D[i, j] = D[j, i] = cross_distance(slugs[i], slugs[j], kind)
            D[i, i] = 0.0
            print(f"  [{kind}] row {i+1}/{n_m} ({slugs[i]})", flush=True)
        obs, p = perm_test(D, fams)
        res[kind] = {"matrix": D.tolist(), "family_sep": obs, "perm_p": p}
        np.save(BOUT / f"dist_{kind}.npy", D)

    # P2: 397B nearest neighbor among the 35 zoo models (real matrix)
    idx397 = slugs.index("qwen35-397b-own")
    row = np.array(res["real"]["matrix"])[idx397].copy()
    row[idx397] = np.inf
    nn = int(np.nanargmin(row))
    p2 = {"nearest": slugs[nn], "nearest_family": family_of(slugs[nn]),
          "is_qwen": family_of(slugs[nn]) == "qwen", "distance": float(row[nn])}

    # MDS (classical, from the real distance matrix) for display
    D = np.array(res["real"]["matrix"]); D = np.nan_to_num(D, nan=float(np.nanmax(D)))
    n = D.shape[0]; J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    coords = V[:, -2:] * np.sqrt(np.maximum(w[-2:], 0))

    verdict = {
        "P1_family_clustering": {
            "real": {"family_sep": res["real"]["family_sep"], "p": res["real"]["perm_p"]},
            "null": {"family_sep": res["null"]["family_sep"], "p": res["null"]["perm_p"]},
            "pass": res["real"]["perm_p"] < 0.05 and res["null"]["perm_p"] > 0.1},
        "P2_397b_nearest_qwen": {**p2, "pass": p2["is_qwen"]},
    }
    out = {"slugs": slugs, "families": fams,
           "mds": coords.tolist(), "verdict": verdict,
           "real_family_sep": res["real"]["family_sep"],
           "real_perm_p": res["real"]["perm_p"],
           "null_family_sep": res["null"]["family_sep"],
           "null_perm_p": res["null"]["perm_p"]}
    (BOUT / "stageB_results.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n=== VERDICTS ===")
    print(f"P1 real family_sep={res['real']['family_sep']:+.4f} p={res['real']['perm_p']:.4f}"
          f" | null family_sep={res['null']['family_sep']:+.4f} p={res['null']['perm_p']:.4f}"
          f" -> {'PASS' if verdict['P1_family_clustering']['pass'] else 'FAIL'}")
    print(f"P2 397B nearest = {p2['nearest']} ({p2['nearest_family']}) "
          f"-> {'PASS' if p2['is_qwen'] else 'FAIL'}")
    print("STAGE_B_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    both = not (a.precompute or a.analyze)
    DS.mkdir(parents=True, exist_ok=True); BOUT.mkdir(parents=True, exist_ok=True)
    slugs = [r["slug"] for r in csv.DictReader(open(OUT / "summary.csv"))]
    slugs = list(dict.fromkeys(slugs))     # de-dup, preserve order
    if a.precompute or both:
        for s in slugs:
            try:
                precompute_one(s)
            except Exception as e:
                print(f"[{s}] PRECOMPUTE FAILED: {type(e).__name__}: {e}", flush=True)
        print("PRECOMPUTE_DONE", flush=True)
    if a.analyze or both:
        analyze(slugs)


if __name__ == "__main__":
    main()
