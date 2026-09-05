"""Evidentiary numbers after the 397B map sections, each anchored to a receipt and asserted present
in the note's prose (RESEARCH_NOTE section 5). This is the check that would have caught the
superseded corpus, fit-budget, Test B and perturbation numbers that survived from July to September.

    verify_note_numbers.py           check; exit 1 on any missing or mismatched number
    verify_note_numbers.py --write   also write numbers.json (the manifest) into the bundle
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E = HERE.parent
POST = Path("/home/ubuntu/PRAX/pre-blog/blog-source/content/posts/2026/07/jlens-cka-397b")


def get(d, ptr):
    for k in ptr.split("/"):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


def load(path):
    return json.loads(Path(path).read_text())


# (label, receipt path, json pointer, transform, prose string that must appear)
B2 = E / "bands_vs_computation/results_B2.json"
B3 = E / "bands_vs_computation/results_B3_ident.json"
CI = E / "geometry_causality/out/analysis_fp32_ci.json"
CO = E / "corpus_dependence/results.json"
FB = E / "corpus_dependence/results_fitbudget.json"
ID = E / "jspace_atlas/atlas_out/boundary_identifiability.json"
TZ = POST / "block-excess-397b.receipt.json"
IG = E / "ignition_depth/results.json"
UN = E / "bands_vs_computation/results_B2_uniform_null.txt"
Q = "qwen3.5-0.8b C32 (local-calibrated dose, 2026-09-05 pod re-run)"
G = "gemma-3-270m C32 (local-calibrated dose)"
P2 = "gpt2-small C32 (local-calibrated dose)"

ENTRIES = [
    ("testB.raw_new.obs", B2, "verdicts/raw_new/obs_mean_agreement_norm", lambda v: f"{v:.3f}", "0.601"),
    ("testB.raw_new.null", B2, "verdicts/raw_new/pooled_null_median", lambda v: f"{v:.3f}", "0.568"),
    ("testB.raw_new.p", B2, "verdicts/raw_new/p_one_sided", lambda v: f"{v:.3f}", "0.613"),
    ("testB.std.obs", B2, "verdicts/standardised_new/obs_mean_agreement_norm", lambda v: f"{v:.3f}", "0.743"),
    ("testB.std.null", B2, "verdicts/standardised_new/pooled_null_median", lambda v: f"{v:.3f}", "0.561"),
    ("testB.std.p", B2, "verdicts/standardised_new/p_one_sided", lambda v: f"{v:.3f}", "0.965"),
    ("testB.raw_old.obs", B2, "verdicts/raw_old/obs_mean_agreement_norm", lambda v: f"{v:.3f}", "0.715"),
    ("testB.anchor.std_mid_sep", B2, "anchor_gate/std_act_mid_sep", lambda v: f"+{v:.4f}", "+0.2086"),
    ("testB3.std.obs", B3, "verdicts/tol0.05_standardised_new/obs_mean_agreement_norm", lambda v: f"{v:.3f}", "0.669"),
    ("testB3.std.p", B3, "verdicts/tol0.05_standardised_new/p_one_sided", lambda v: f"{v:.3f}", "0.853"),
    ("perturb.gpt2.ratio", CI, f"{P2}/ratio", lambda v: f"{v:.1f}x", "7.9x"),
    ("perturb.gpt2.ci_lo", CI, f"{P2}/ratio_ci95/0", lambda v: f"{v:.1f}", "7.0"),
    ("perturb.gpt2.ci_hi", CI, f"{P2}/ratio_ci95/1", lambda v: f"{v:.1f}", "9.0"),
    ("perturb.gemma.ratio", CI, f"{G}/ratio", lambda v: f"{v:.1f}x", "11.2x"),
    ("perturb.gemma.ci_lo", CI, f"{G}/ratio_ci95/0", lambda v: f"{v:.1f}", "10.1"),
    ("perturb.gemma.ci_hi", CI, f"{G}/ratio_ci95/1", lambda v: f"{v:.1f}", "12.4"),
    ("perturb.qwen.ratio", CI, f"{Q}/ratio", lambda v: f"{v:.1f}x", "5.3x"),
    ("perturb.qwen.ci_lo", CI, f"{Q}/ratio_ci95/0", lambda v: f"{v:.1f}", "4.7"),
    ("perturb.qwen.ci_hi", CI, f"{Q}/ratio_ci95/1", lambda v: f"{v:.1f}", "6.0"),
    ("perturb.qwen.C", CI, f"{Q}/median_C", lambda v: f"{v:.3f}", "0.026"),
    ("perturb.gpt2.C", CI, f"{P2}/median_C", lambda v: f"{v:.3f}", "0.053"),
    ("corpus.gpt2.ratio", CO, "gpt2-small", lambda d: f"{d['corpus']['map_distance']/d['seed_null']['map_distance']:.0f}x", "84x"),
    ("corpus.gemma.ratio", CO, "gemma-3-270m", lambda d: f"{d['corpus']['map_distance']/d['seed_null']['map_distance']:.0f}x", "888x"),
    ("corpus.qwen.ratio", CO, "qwen3.5-0.8b", lambda d: f"{d['corpus']['map_distance']/d['seed_null']['map_distance']:.0f}x", "86x"),
    ("corpus.gemma.shift_code", CO, "gemma-3-270m/corpus/boundary_shift", lambda v: str(v), "0, 0 and 1 layers"),
    ("corpus.qwen.code_dist", CO, "qwen3.5-0.8b/corpus/map_distance", lambda v: f"{v:.3f}", "0.044"),
    ("fitbudget.gemma.25", FB, "models/gemma-3-270m/by_budget/25/ratio_to_seed_null", lambda v: f"{v:.1f}x", "14.2x"),
    ("fitbudget.gemma.50", FB, "models/gemma-3-270m/by_budget/50/ratio_to_seed_null", lambda v: f"{v:.1f}x", "3.6x"),
    ("fitbudget.gpt2.25", FB, "models/gpt2-small/by_budget/25/ratio_to_seed_null", lambda v: f"{v:.1f}x", "1.8x"),
    ("fitbudget.verdict", FB, "verdict", lambda v: v, "converged"),
    ("ident.n_identified", ID, "models", lambda m: f"{sum(1 for r in m.values() if r.get('shared',{}).get('identified'))} of {sum(1 for r in m.values() if 'shared' in r and not r['shared'].get('too_small'))}", "20 of 35"),
    ("ident.397b.b1", ID, "models/qwen35-397b-own/shared/spread_b1", lambda v: f"{v:.2f}", "0.08"),
    ("ident.397b.b2", ID, "models/qwen35-397b-own/shared/spread_b2", lambda v: f"{v:.2f}", "0.07"),
    ("toeplitz.mid_sep", TZ, "plotted_values/mid_sep_toeplitz", lambda v: f"{v:.3f}", "0.275"),
    ("toeplitz.excess", TZ, "plotted_values", lambda d: f"+{d['fitted_sep_real']-d['fitted_sep_toeplitz']:.3f}", "+0.125"),
    ("toeplitz.zoo_median", TZ, "plotted_values/zoo_excess_median", lambda v: f"+{v:.3f}", "+0.017"),
    ("ignition.gemma.late", IG, "models/gemma-2-9b/lens_late_reldepth", lambda v: f"{v:.3f}", "0.415"),
    ("ignition.qwen.late", IG, "models/qwen3.5-0.8b/lens_late_reldepth", lambda v: f"{v:.3f}", "0.652"),
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    md = (POST / "index.md").read_text().replace("−", "-")
    fails, out = [], []
    for label, path, ptr, fmt, want in ENTRIES:
        try:
            d = load(path); v = get(d, ptr) if ptr else d
            got = fmt(v)
        except Exception as e:
            fails.append(f"{label}: cannot derive ({type(e).__name__}: {e})"); continue
        if isinstance(got, str) and want.replace("x", "").replace("+", "").strip() and got.lower() != want.lower() and not (label.endswith("shift_code")):
            fails.append(f"{label}: receipt gives {got!r}, manifest expects {want!r}")
        if want not in md:
            fails.append(f"{label}: {want!r} not in prose")
        out.append({"label": label, "receipt": str(Path(path).relative_to(E.parent.parent.parent)) if str(path).startswith(str(E)) else Path(path).name,
                    "pointer": ptr, "value_str": got if isinstance(got, str) else str(got), "prose": want})
    if a.write:
        (POST / "numbers.json").write_text(json.dumps({"what": "evidentiary numbers after the 397B map sections; each re-derived from its receipt and asserted present in prose", "entries": out}, indent=1))
    if fails:
        print("NUMBERS VERIFY FAILED:"); [print("  -", f) for f in fails]; sys.exit(1)
    print(f"NUMBERS VERIFY OK: {len(ENTRIES)} evidentiary numbers re-derived from receipts and present in prose")


if __name__ == "__main__":
    main()
