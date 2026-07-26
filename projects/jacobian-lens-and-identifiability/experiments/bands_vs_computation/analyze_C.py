"""Test C analysis: does the fitting corpus change which boundary predicts damage?

Design frozen in PREREG.md. The damage matrix D is held completely fixed within each cell; only
the segmentation label applied to it varies. That is the identification.

For each model and each damage corpus (prose, code), fit

    D(i,j) ~ dummies(|i-j|) + dummies(mean position) + beta * crosses(i,j)

twice: once with the boundaries from that model's WikiText lens, once with the boundaries from
its code lens. Then

    C1 = beta in the MATCHED cells, against a random-3-segmentation null
    C2 = beta(matched) - beta(mismatched), the within-model paired contrast

gpt2-small is the mechanical control: its two segmentations are identical, so its C2 must be
exactly 0 or the run is VOID.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CORPUS_RESULTS = HERE.parent / "corpus_dependence/results.json"
CORPORA = ["prose", "code"]
# which lens arm supplies boundaries for each damage corpus, per the frozen 2x2
LENS_ARM = {"prose": "wiki_a", "code": "code"}
NPERM = 1000


def design(D, blk):
    """D ~ distance dummies + position dummies + beta * crosses. Returns beta and n rows."""
    L = D.shape[0]
    rows = [(i, j) for i in range(L) for j in range(L) if i != j and np.isfinite(D[i, j])]
    y = np.array([D[i, j] for i, j in rows])
    dist = [abs(i - j) for i, j in rows]
    pos = [int(round((i + j) / 2)) for i, j in rows]
    du, pu = sorted(set(dist)), sorted(set(pos))
    X = np.zeros((len(y), len(du) + len(pu) - 1 + 1))
    for k, (i, j) in enumerate(rows):
        X[k, du.index(dist[k])] = 1.0
        if pos[k] != pu[0]:
            X[k, len(du) + pu.index(pos[k]) - 1] = 1.0
        X[k, -1] = 1.0 if blk[i] != blk[j] else 0.0
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[-1]), len(y)


def seg_labels(b1, b2, L):
    return np.array([0 if i < b1 else (1 if i < b2 else 2) for i in range(L)])


def null_betas(D, b1, b2, L, nperm, seed):
    """Random 3-segmentations with the same block-size multiset."""
    sizes = [b1, b2 - b1, L - b2]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nperm):
        s = list(rng.permutation(sizes))
        c1, c2 = s[0], s[0] + s[1]
        if c1 == 0 or c2 >= L or c1 >= c2:
            continue
        out.append(design(D, seg_labels(c1, c2, L))[0])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "out"))
    ap.add_argument("--out", default=str(HERE / "results_C.json"))
    a = ap.parse_args()

    corpus = json.loads(CORPUS_RESULTS.read_text())
    out = {"nperm": NPERM, "models": {}}

    for slug in ["gpt2-small", "gemma-3-270m", "qwen3.5-0.8b"]:
        try:
            bnds = corpus[slug]["boundaries"]          # {"wiki_a": [b1,b2], "code": [...], ...}
            m = {"boundaries": {k: bnds[k] for k in ("wiki_a", "code")}, "cells": {}}
            identical = bnds["wiki_a"] == bnds["code"]
            m["segmentations_identical"] = identical

            for dc in CORPORA:                          # damage corpus
                p = Path(a.runs) / f"{slug}_{dc}.json"
                if not p.exists():
                    print(f"{slug}/{dc}: MISSING {p.name}", flush=True); continue
                r = json.loads(p.read_text())
                D = np.array(r["D"], float)
                L = D.shape[0]
                gates = {"sanity": r["diag_max_abs"] < 1e-6,
                         "calibration": 0.05 <= r["median_D_far"] <= 5.0}
                cell = {"gates": gates, "median_D_far": r["median_D_far"], "n_layers": L}
                for arm in ("wiki_a", "code"):
                    b1, b2 = bnds[arm]
                    if not (0 < b1 < b2 < L):
                        cell[arm] = {"error": f"boundaries {b1},{b2} outside 0..{L}"}
                        continue
                    beta, n = design(D, seg_labels(b1, b2, L))
                    nb = null_betas(D, b1, b2, L, NPERM, seed=0)
                    cell[arm] = {"beta": beta, "n_pairs": n,
                                 "null_mean": float(nb.mean()), "null_sd": float(nb.std()),
                                 "p_two_sided": float((np.abs(nb - nb.mean())
                                                       >= abs(beta - nb.mean())).mean()),
                                 "matched": LENS_ARM[dc] == arm}
                m["cells"][dc] = cell

            # C2: matched minus mismatched, averaged over the two damage corpora
            diffs = []
            for dc, cell in m["cells"].items():
                mt, mm = LENS_ARM[dc], ("code" if LENS_ARM[dc] == "wiki_a" else "wiki_a")
                if "beta" in cell.get(mt, {}) and "beta" in cell.get(mm, {}):
                    diffs.append(cell[mt]["beta"] - cell[mm]["beta"])
            m["C2_matched_minus_mismatched"] = float(np.mean(diffs)) if diffs else None
            m["C1_matched_betas"] = [c[LENS_ARM[dc]]["beta"] for dc, c in m["cells"].items()
                                     if "beta" in c.get(LENS_ARM[dc], {})]
            out["models"][slug] = m

            print(f"\n{slug}  L={m['cells'].get('prose',{}).get('n_layers','?')}  "
                  f"wiki_b={bnds['wiki_a']} code_b={bnds['code']}"
                  f"{'  [IDENTICAL -> mechanical control]' if identical else ''}")
            for dc, cell in m["cells"].items():
                g = "PASS" if all(cell["gates"].values()) else "FAIL"
                print(f"   damage on {dc:6s} gates={g} (median D {cell['median_D_far']:.3f})")
                for arm in ("wiki_a", "code"):
                    c = cell.get(arm, {})
                    if "beta" not in c:
                        print(f"      {arm:7s} {c.get('error','n/a')}"); continue
                    tag = "MATCHED   " if c["matched"] else "mismatched"
                    print(f"      {arm:7s} {tag} beta={c['beta']:+.4f} "
                          f"null sd={c['null_sd']:.4f} p={c['p_two_sided']:.3f}")
            print(f"   C2 (matched - mismatched) = {m['C2_matched_minus_mismatched']}")
        except Exception as e:
            print(f"{slug}: FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
            out["models"][slug] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}

    ok = {k: v for k, v in out["models"].items() if "error" not in v and v.get("cells")}
    ctrl = out["models"].get("gpt2-small", {})
    if ctrl.get("segmentations_identical") and ctrl.get("C2_matched_minus_mismatched") is not None:
        out["control_c2"] = ctrl["C2_matched_minus_mismatched"]
        out["control_void"] = abs(ctrl["C2_matched_minus_mismatched"]) > 1e-12
    contrast = {k: v for k, v in ok.items() if not v.get("segmentations_identical")}
    if contrast:
        out["C2_pooled"] = float(np.mean([v["C2_matched_minus_mismatched"]
                                          for v in contrast.values()
                                          if v["C2_matched_minus_mismatched"] is not None]))
        allm = [b for v in ok.values() for b in v["C1_matched_betas"]]
        out["C1_pooled_matched_beta"] = float(np.mean(allm)) if allm else None
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nCONTROL gpt2-small C2 = {out.get('control_c2')} "
          f"(must be exactly 0; VOID={out.get('control_void')})")
    print(f"C1 pooled matched beta = {out.get('C1_pooled_matched_beta')}")
    print(f"C2 pooled (contrast models only) = {out.get('C2_pooled')}")
    print("TESTC_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
