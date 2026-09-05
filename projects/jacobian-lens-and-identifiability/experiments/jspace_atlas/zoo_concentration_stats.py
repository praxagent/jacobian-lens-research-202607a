"""Receipt for the readout-concentration correlations quoted in the note (2026-09-05).
Spearman of shared-probe band separation with the top-16 readout energy share and with the
width-normalised participation ratio, plus partial Spearmans controlling for log width.
Reads decompose_out/zoo_concentration.json + atlas_out/shared_summary.csv; writes
decompose_out/zoo_concentration_stats.json."""
import csv, json, numpy as np
from pathlib import Path
from scipy.stats import spearmanr, rankdata
HERE = Path(__file__).resolve().parent
conc = json.load(open(HERE / "decompose_out/zoo_concentration.json"))
shared = {r["slug"]: float(r["mid_sep"]) for r in csv.DictReader(open(HERE / "atlas_out/shared_summary.csv"))}
rows = [(s, v["top16_share"], v.get("pr_norm"), v["d"], shared[s]) for s, v in conc.items()
        if s in shared and v.get("top16_share") is not None and v.get("pr_norm") is not None and v.get("d")]
slugs, t16, prn, d, ms = map(np.array, zip(*rows)); n = len(rows)
def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        A = np.vstack([b, np.ones_like(b)]).T; coef, *_ = np.linalg.lstsq(A, a, rcond=None); return a - A @ coef
    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])
out = {"n": int(n), "models": list(slugs),
       "spearman_top16_vs_shared_mid_sep": float(spearmanr(t16, ms).correlation),
       "spearman_top16_vs_shared_mid_sep_p": float(spearmanr(t16, ms).pvalue),
       "spearman_prnorm_vs_shared_mid_sep": float(spearmanr(prn, ms).correlation),
       "spearman_prnorm_vs_shared_mid_sep_p": float(spearmanr(prn, ms).pvalue),
       "spearman_logd_vs_shared_mid_sep": float(spearmanr(np.log(d), ms).correlation),
       "partial_top16_given_logd": partial_spearman(t16, ms, np.log(d)),
       "partial_logd_given_top16": partial_spearman(np.log(d), ms, t16),
       "note": "convenience sample with family clustering; descriptive, not population inference"}
json.dump(out, open(HERE / "decompose_out/zoo_concentration_stats.json", "w"), indent=1)
for k, v in out.items():
    if k not in ("models", "note"): print(f"{k:42s} {v}")
