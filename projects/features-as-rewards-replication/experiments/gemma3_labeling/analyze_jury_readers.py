"""Jury-labeled evaluation of the label-free readers on Gemma-3-12B — CPU, $0.

Joins the arm-4 per-span reader scores (receipt_gemma3_agreement.json — computed
BEFORE any labels existed, frozen) with the jury labels (jury_gemma3_labels.json,
majority+ tier). Positive class = "Not Supported" (hallucinated=1), matching the
public-label arms. Lens/heuristic scores are surprisal-style (higher -> predicts
hallucinated) so their orientation is a-priori, not fitted; the SAE max-activation
has no a-priori sign, so its sign is chosen on a validation fold of completions
(idx % 5 == 0) and its AUROC reported on the remaining completions only (disclosed).

CIs: completion-clustered bootstrap (resample completions with replacement), the
same procedure as the main arms. This arm evaluates the ZERO-TRAINING readers only;
the supervised probe needs cached activations -> the follow-up GPU arm.

  .venv/bin/python analyze_jury_readers.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "common"))
from readers import auroc  # noqa: E402

ARM4 = HERE.parents[1] / "gemma3_agreement/receipts/receipt_gemma3_agreement.json"
JURY = HERE.parent / "receipts/jury_gemma3_labels.json"
OUT = HERE.parent / "receipts/jury_readers_eval.json"
READERS = ["logit_lens", "native_head", "random_null", "heuristic", "sae_max_act"]
N_BOOT, SEED = 2000, 0


def cluster_ci(scores, labels, comps, rng):
    """95% CI: resample completions with replacement (labels correlate within one)."""
    by_comp = defaultdict(list)
    for i, c in enumerate(comps):
        by_comp[c].append(i)
    keys = sorted(by_comp)
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([by_comp[k] for k in rng.choice(keys, len(keys))])
        a = auroc(scores[idx], labels[idx])
        if not np.isnan(a):
            vals.append(a)
    return [round(float(np.percentile(vals, q)), 3) for q in (2.5, 97.5)]


def eval_tier(rows, tier_filter, rng):
    sel = [r for r in rows if tier_filter(r["tier"])]
    labels = np.array([r["label"] for r in sel])
    comps = [r["cid"] for r in sel]
    out = {"n": len(sel), "n_pos": int(labels.sum())}
    for name in READERS:
        scores = np.array([r[name] for r in sel])
        if name == "sae_max_act":
            # sign on val fold (completion idx % 5 == 0), AUROC on the rest
            val = np.array([int(c.split(":")[1]) % 5 == 0 for c in comps])
            a_val = auroc(scores[val], labels[val])
            sign = 1.0 if a_val >= 0.5 else -1.0
            test = ~val
            out[name] = {
                "auroc": round(auroc(sign * scores[test], labels[test]), 3),
                "ci95": cluster_ci(sign * scores[test], labels[test],
                                   [c for c, t in zip(comps, test) if t], rng),
                "sign": sign, "val_fold_auroc": round(a_val, 3),
                "note": "sign val-fold-selected; AUROC on held-out completions only"}
        else:
            out[name] = {"auroc": round(auroc(scores, labels), 3),
                         "ci95": cluster_ci(scores, labels, comps, rng)}
    return out


def main():
    arm4 = json.loads(ARM4.read_text())
    jury = json.loads(JURY.read_text())
    jmap = {(r["cid"], r["entity"].strip()): r for r in jury["rows"]}
    rows = []
    for r in arm4["per_span"]:
        j = jmap.get((r["cid"], r["entity"].strip()))
        if j is None or j["jury_label"] is None or j["tier"] == "split":
            continue
        rows.append({**{k: r[k] for k in READERS}, "cid": r["cid"],
                     "entity": r["entity"].strip(), "tier": j["tier"],
                     "label": 1 if j["jury_label"] == "Not Supported" else 0})
    print(f"  joined {len(rows)} definitive-labeled spans "
          f"({sum(r['label'] for r in rows)} Not Supported)")

    rng = np.random.default_rng(SEED)
    res = {"majority_plus": eval_tier(rows, lambda t: True, rng),
           "unanimous_only": eval_tier(rows, lambda t: t == "unanimous", rng)}
    for tier, o in res.items():
        print(f"  [{tier}] n={o['n']} pos={o['n_pos']}")
        for name in READERS:
            print(f"    {name:13s} AUROC {o[name]['auroc']:.3f}  CI {o[name]['ci95']}")

    receipt = {
        "what": "Jury-labeled AUROC of the LABEL-FREE readers on Gemma-3-12B. Scores "
                "come unchanged from the arm-4 receipt (computed before labels "
                "existed); labels = cross-provider jury (kappa .598 majority / .691 "
                "unanimous vs public gold — agreement with rubric, NOT ground truth). "
                "Positive class = Not Supported. Surprisal readers a-priori oriented; "
                "SAE sign val-fold-selected. Supervised probe requires the GPU arm.",
        "inputs": {"arm4_receipt": str(ARM4.name), "jury_receipt": str(JURY.name)},
        "n_joined": len(rows), "bootstrap": {"n": N_BOOT, "seed": SEED,
                                             "cluster": "completion"},
        "results": res}
    OUT.write_text(json.dumps(receipt, indent=1))
    print(f"  JURY_READERS_EVAL_DONE -> {OUT}")


if __name__ == "__main__":
    main()
