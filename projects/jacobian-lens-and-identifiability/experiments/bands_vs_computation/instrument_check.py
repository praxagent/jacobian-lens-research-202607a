"""Is activation CKA a usable instrument, and was our degeneracy gate measuring the right thing?

Test B excluded three of eight models on `median off-diagonal CKA >= 0.999`. This checks whether
that gate identifies "no structure to compare against", which is what it was for.

Free: reads the CKA matrices already saved by activation_boundaries.py. No model, no GPU.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OLD_GATE = 0.999          # what we froze: median off-diagonal
RANGE_FLOOR = 0.10        # candidate repair: does the map have any dynamic range at all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "out"))
    ap.add_argument("--out", default=str(HERE / "results_instrument.json"))
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(str(Path(a.runs) / "*.json"))):
        if "smoke" in f or "probe" in f:
            continue
        d = json.loads(Path(f).read_text())
        M = np.load(f.replace(".json", ".npz"))["cka"]
        tri = M[np.triu_indices_from(M, 1)]
        rows.append({
            "slug": d["slug"], "n_layers": d["n_layers"],
            "median": float(np.median(tri)), "min": float(tri.min()),
            "p05": float(np.percentile(tri, 5)), "max": float(tri.max()),
            "range": float(tri.max() - tri.min()),
            "iqr": float(np.percentile(tri, 75) - np.percentile(tri, 25)),
            "act_mid_sep": d["act_mid_sep"],
            "excluded_by_old_gate": bool(np.median(tri) >= OLD_GATE),
            "excluded_by_range_gate": bool(tri.max() - tri.min() < RANGE_FLOOR),
        })

    print(f"{'model':20s} {'median':>8s} {'min':>8s} {'range':>8s}   "
          f"{'old gate':>10s}  {'range gate':>11s}")
    for r in rows:
        print(f"{r['slug']:20s} {r['median']:8.4f} {r['min']:8.4f} {r['range']:8.4f}   "
              f"{'EXCLUDE' if r['excluded_by_old_gate'] else 'keep':>10s}  "
              f"{'EXCLUDE' if r['excluded_by_range_gate'] else 'keep':>11s}")

    disagree = [r for r in rows
                if r["excluded_by_old_gate"] != r["excluded_by_range_gate"]]
    out = {"old_gate_median_threshold": OLD_GATE, "range_gate_floor": RANGE_FLOOR,
           "models": rows,
           "gates_disagree_on": [r["slug"] for r in disagree],
           "n_excluded_old": sum(r["excluded_by_old_gate"] for r in rows),
           "n_excluded_range": sum(r["excluded_by_range_gate"] for r in rows)}

    print(f"\nThe two gates disagree on {len(disagree)} of {len(rows)} models: "
          + ", ".join(r["slug"] for r in disagree))
    for r in disagree:
        why = ("median is high but the map spans a wide range, so there IS structure"
               if r["excluded_by_old_gate"] else
               "median is below threshold but the map is nearly constant, so there is NOT")
        print(f"  {r['slug']}: {why} (median {r['median']:.4f}, range {r['range']:.4f})")

    # the decisive comparison: does the old gate order models by how much structure they have?
    med = np.array([r["median"] for r in rows])
    rng = np.array([r["range"] for r in rows])
    order_med = np.argsort(med)
    out["spearman_median_vs_range"] = float(
        np.corrcoef(np.argsort(np.argsort(med)), np.argsort(np.argsort(rng)))[0, 1])
    print(f"\nrank correlation between median and range across models: "
          f"{out['spearman_median_vs_range']:+.3f}")
    print("A gate on the median is only a proxy for 'has structure' if this is strongly "
          "negative.")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("INSTRUMENT_CHECK_DONE")


if __name__ == "__main__":
    main()
