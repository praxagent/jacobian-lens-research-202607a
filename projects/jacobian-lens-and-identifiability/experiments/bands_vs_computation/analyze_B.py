"""Test B analysis: do lens boundaries coincide with activation boundaries?

Design frozen in PREREG.md, Amendment 1 (index alignment). Primary statistic per model:

    agreement = |b1_lens - b1_act| + |b2_lens - b2_act|     (layers, lower = better)

against a null of random 3-segmentations with the same block-size multiset as the lens
segmentation. Depth-normalised as agreement / L for pooling.

Frozen decision rules:
  pooled p < 0.05 AND >= 6 of 8 models beat their own null median -> BANDS TRACK REPRESENTATIONS
  pooled p >= 0.05                                               -> BANDS ARE A READOUT PROPERTY
  anything between                                               -> MIXED
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = ["gpt2-small", "pythia-70m-deduped", "gemma-3-270m", "gemma-2-2b",
          "gemma-2-9b", "qwen3.5-0.8b", "qwen3-4b", "llama3.1-8b"]
NPERM = 1000
ANCHOR = "gemma-2-9b"
ANCHOR_TARGET, ANCHOR_TOL = 0.110, 0.03


def null_agreement(b1, b2, L, act, nperm, seed):
    """Distribution of agreement when the lens segmentation is replaced by a random one
    with the same block-size multiset. The ACTIVATION boundaries are held fixed."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "out"))
    ap.add_argument("--out", default=str(HERE / "results_B.json"))
    a = ap.parse_args()

    out = {"nperm": NPERM, "models": {}, "failed": []}
    rows = []
    for slug in MODELS:
        p = Path(a.runs) / f"{slug}.json"
        if not p.exists():
            out["failed"].append({"slug": slug, "reason": "no receipt"}); continue
        r = json.loads(p.read_text())
        if not r["diag_ok"] or r["degenerate"]:
            out["failed"].append({"slug": slug,
                                  "reason": "degenerate activation map"
                                            if r["degenerate"] else "diagonal not 1.0",
                                  "act_offdiag_median": r["act_offdiag_median"]})
            continue
        L = r["n_layers"]
        lb, ab = r["lens_boundaries"], r["act_boundaries"]
        agree = abs(lb[0] - ab[0]) + abs(lb[1] - ab[1])
        nb = null_agreement(lb[0], lb[1], L, ab, NPERM, seed=0)
        m = {"n_layers": L, "lens_boundaries": lb, "act_boundaries": ab,
             "agreement": int(agree), "agreement_norm": agree / L,
             "null_median": float(np.median(nb)), "null_mean": float(nb.mean()),
             "null_sd": float(nb.std()),
             "p_one_sided": float((nb <= agree).mean()),   # P(null at least this good)
             "beats_null_median": bool(agree < np.median(nb)),
             "act_mid_sep": r["act_mid_sep"],
             "act_offdiag_median": r["act_offdiag_median"],
             "vendored_fitted_seg_check": r.get("vendored_fitted_seg_check")}
        out["models"][slug] = m
        rows.append(m)
        print(f"{slug:20s} L={L:3d}  lens={lb}  act={ab}  agreement={agree:3d} "
              f"(null median {np.median(nb):5.1f})  p={m['p_one_sided']:.3f}  "
              f"{'BEATS' if m['beats_null_median'] else 'no'}", flush=True)

    # anchor gate
    anc = out["models"].get(ANCHOR)
    if anc is None:
        out["anchor_gate"] = {"status": "MISSING", "note": f"{ANCHOR} produced no usable map"}
    else:
        d = abs(anc["act_mid_sep"] - ANCHOR_TARGET)
        out["anchor_gate"] = {"status": "PASS" if d <= ANCHOR_TOL else "FAIL",
                              "act_mid_sep": anc["act_mid_sep"],
                              "target": ANCHOR_TARGET, "abs_diff": d}
    print(f"\nANCHOR GATE ({ANCHOR} activation band sep must be {ANCHOR_TARGET} +/- {ANCHOR_TOL}): "
          f"{out['anchor_gate']}")

    if rows:
        # pooled test: mean normalised agreement against the mean of per-model nulls,
        # combined by a permutation over the same per-model null draws
        rng = np.random.default_rng(0)
        per_null = []
        for slug, m in out["models"].items():
            nb = null_agreement(m["lens_boundaries"][0], m["lens_boundaries"][1],
                                m["n_layers"], m["act_boundaries"], NPERM, seed=0)
            per_null.append(nb / m["n_layers"])
        K = min(len(x) for x in per_null)
        stack = np.stack([rng.permutation(x)[:K] for x in per_null])   # (models, K)
        pooled_null = stack.mean(0)
        obs = float(np.mean([m["agreement_norm"] for m in rows]))
        p = float((pooled_null <= obs).mean())
        n_beat = sum(m["beats_null_median"] for m in rows)
        out["pooled"] = {"n_models_used": len(rows), "obs_mean_agreement_norm": obs,
                         "pooled_null_median": float(np.median(pooled_null)),
                         "p_one_sided": p, "n_models_beating_own_null_median": n_beat}
        if p < 0.05 and n_beat >= 6:
            v = "BANDS TRACK REPRESENTATIONS"
        elif p >= 0.05:
            v = "BANDS ARE A READOUT PROPERTY"
        else:
            v = "MIXED"
        out["verdict"] = v
        print(f"\npooled mean normalised agreement = {obs:.4f} "
              f"(null median {np.median(pooled_null):.4f}), p = {p:.4f}, "
              f"{n_beat}/{len(rows)} models beat their own null median")
        print(f"VERDICT: {v}")
        if out["failed"]:
            print(f"FAILED/excluded ({len(out['failed'])}): "
                  + ", ".join(f["slug"] + " (" + f["reason"] + ")" for f in out["failed"]))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("TESTB_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
