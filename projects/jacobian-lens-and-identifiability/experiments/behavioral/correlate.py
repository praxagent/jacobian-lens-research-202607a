"""Turn the behavioral triple into a FRONTIER result: does the geometric workspace
band quantitatively predict behavioral workspace signatures across models?

Merges per-model behavioral outputs (verbal_report_*.json swap-success; ignition_*.json
ignition sharpness + share-span) with the geometric band statistic (mid_sep from the
emergence sweep), then computes rank correlations across the whole model set.

Predictions (the two-level / Eleos-tiered story):
  - SWAP success vs mid_sep: ~flat / uncorrelated  -> the privileged-SET tier (reportable,
    steerable) is universal, present even without a geometric band.
  - IGNITION sharpness (frac_sharp, and share-span) vs mid_sep: POSITIVELY correlated
    -> the all-or-none workspace signature tracks the band. This is the headline: a
    behavioral correlate of the geometric band, across N models.

If ignition does NOT correlate, the honest result flips to "our band statistic does not
capture a functional workspace difference" — equally reportable. Either way, N-model
correlation >> a 3-model anecdote.

Run:  uv run python correlate.py   (reads all *_*.json here + emergence.csv)
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EMERGENCE = HERE.parents[1] / "experiments" / "jacobian_lens" / "emergence.csv"


def load_midsep() -> dict[str, float]:
    m = {}
    for src in (EMERGENCE, EMERGENCE.with_name("emergence_fp32path.csv")):
        if src.exists():
            for r in csv.DictReader(open(src)):
                m.setdefault(r["slug"], float(r["mid_sep"]))  # prefer uniform (first)
    return m


def spearman(xs, ys) -> float:
    """Rank correlation, dependency-free. Average ranks for ties (the standard
    treatment) — an earlier version assigned distinct ranks to tied values in
    input order, which systematically inflated rho here (8 Gemmas tied at
    share_span 0.000 in a mid_sep-sorted table); see results.md estimator note."""
    n = len(xs)
    if n < 3:
        return float("nan")
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        rk = [0.0] * n
        pos = 0
        while pos < n:
            end = pos
            while end + 1 < n and v[order[end + 1]] == v[order[pos]]:
                end += 1
            avg = (pos + end) / 2.0
            for k in range(pos, end + 1):
                rk[order[k]] = avg
            pos = end + 1
        return rk
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    vy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


# verbal_report is invalid for these: the Gemma-4 base models emit empty greedy output
# for "Think of a {cat}. Answer in one word:" (they don't answer bare instructions), so
# swap/reportability are meaningless. Their ignition (share_span) IS valid — it interpolates
# embeddings and reads the J-lens, no answer needed — so we keep share_span, drop swap only.
_BROKEN_VERBAL_REPORT = {"gemma-4-e2b", "gemma-4-e4b"}


def main() -> None:
    midsep = load_midsep()
    rows = []
    for vr in sorted(glob.glob(str(HERE / "verbal_report_*.json"))):
        slug = Path(vr).stem.replace("verbal_report_", "")
        v = json.load(open(vr))
        ig_path = HERE / f"ignition_{slug}.json"
        ig = json.load(open(ig_path)) if ig_path.exists() else {}
        ssr = v.get("swap_success_rate", {})
        swap_best = max((float(x) for x in ssr.values()), default=float("nan"))
        def nn(x):
            return float("nan") if x is None else float(x)
        report_hit = nn(v.get("reportability_hit_rate"))
        if slug in _BROKEN_VERBAL_REPORT:
            swap_best = float("nan")   # empty greedy answers -> swap undefined (keep share_span)
            report_hit = float("nan")
        rows.append({
            "slug": slug,
            "mid_sep": midsep.get(slug, float("nan")),
            "swap_best": swap_best,
            "report_hit": report_hit,
            "ign_sharp": nn(ig.get("frac_sharp_lt_0.25")),
            "ign_width": nn(ig.get("median_transition_width")),
            "ign_share_span": nn(ig.get("median_share_span")),
            "ign_resolved": nn(ig.get("frac_readouts_resolved")),
        })
    rows.sort(key=lambda r: (r["mid_sep"] if r["mid_sep"] == r["mid_sep"] else -1))

    out_csv = HERE / "behavioral_correlation.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def col(k):
        return [r[k] for r in rows]
    def clean(a, b):
        pairs = [(x, y) for x, y in zip(col(a), col(b)) if x == x and y == y]
        return ([p[0] for p in pairs], [p[1] for p in pairs])

    print(f"{'slug':16s} {'mid_sep':>8s} {'swap':>6s} {'ign_sharp':>9s} {'share_span':>10s}")
    for r in rows:
        print(f"{r['slug']:16s} {r['mid_sep']:8.4f} {r['swap_best']:6.2f} "
              f"{r['ign_sharp'] if r['ign_sharp']==r['ign_sharp'] else -1:9.2f} "
              f"{r['ign_share_span'] if r['ign_share_span']==r['ign_share_span'] else -1:10.3f}")
    n = len(rows)
    print(f"\nN models = {n}")
    for label, key in [("swap_best", "swap_best"), ("ign_sharp", "ign_sharp"),
                       ("ign_share_span", "ign_share_span")]:
        xs, ys = clean("mid_sep", key)
        rho = spearman(xs, ys)
        print(f"  Spearman(mid_sep, {label:14s}) = {rho:+.3f}   (n={len(xs)})")
    print("\nHeadline test: ign_sharp / ign_share_span should correlate +vely with "
          "mid_sep; swap_best should not. Needs n>=~6 for a meaningful rho.")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
