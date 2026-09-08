# Pre-registration: corpus dependence at 8B, second attempt at 400 prompts (llama3.1-8b)

Frozen 2026-09-08 before any 400-prompt fit exists. This is a **second, separately frozen
attempt** at the question in `PREREG_8B.md` (73fb7cb). The first attempt (100 prompts per arm,
run 2026-09-05/06) failed its anchor by 0.001 and issued no corpus statement under its frozen
table. It stays on the record in `results.md`, `results.json` and `results_8b.json`; this attempt
does not replace it, and both are reported.

## Why a second attempt, and what had been seen when this was written

Everything in the first attempt's ledger entry had been inspected: anchor 0.0111 against the 0.01
gate; seed null 0.0183 (6x to 180x the three small models' seed nulls); corpus map distance 0.0913
(5x that seed null); early boundary 13 -> 6 on code; band shift below the seed null. The diagnosis
is under-convergence at 100 prompts: our `wiki_a` fit is closer to the public lens (0.011) than to
our own `wiki_b` resample (0.018), so the gate was missed on sampling noise, not on a recipe
difference. **This attempt is therefore not blind to the first attempt's numbers.** The predictions
below are stated so that a miss is on the record.

## Design: identical recipe, four times the budget

Same model (`meta-llama/Llama-3.1-8B`), same three arms, same `fit_our_own/fit_lens.py` recipe
(`--max-seq-len 128 --match-length --dim-batch 8`, bf16, seeds 0 / 1 / 0), **400 prompts per arm**.
`load_corpus` shuffles by seed and takes a prefix, so each 400-prompt fit extends the corresponding
100-prompt fit (the first 100 qualifying passages are identical); the two attempts are nested, not
independent. Fits are named `llama8b400_{wiki_a,wiki_b,code}.pt`.

Measures unchanged: shared-vocabulary probe, rows from `lm_head.weight` (asserted), the corrected
linear-CKA statistic with its run-time identity check, boundary shift, map distance (`1 - CKA`
between off-diagonal map profiles), band shift; seed null = `wiki_a` vs `wiki_b`; anchor = our
`wiki_a` map against the same public Neuronpedia llama3.1-8b shared map (`ANCHOR_SLUG` in
`analyze.py`). The analysis runs as `analyze.py --only llama3.1-8b-n400 --out results_8b_n400_raw.json`
and `analyze_8b.py --results results_8b_n400_raw.json --slug llama3.1-8b-n400 --out results_8b_n400.json`;
`results.json`'s 100-prompt entry is not touched.

## Predictions

- **Anchor**: passes. If map distance falls roughly as 1/n from 0.011, the 400-prompt `wiki_a` map
  lands near 0.003 to 0.006 from the public map. The gate stays at 0.01.
- **Seed null** shrinks to roughly a quarter of 0.018 (about 0.005). Stated as a check on the
  under-convergence diagnosis; it is not a gate.
- **P1** (map): corpus map distance exceeds the seed null by more than 10x. Predicted to hold: if the
  corpus distance stays near 0.09 while the null falls to about 0.005, the ratio is about 20x.
- **P2** (boundaries): corpus boundary shift at most max(2, seed boundary shift). **No confident
  prediction.** The 100-prompt code fit moved the early boundary by 7 layers; whether that survives
  a converged fit is exactly what this attempt asks.
- **P3** (descriptive): band separation on code differs from WikiText by more than the seed band
  shift. No verdict attached.

## Decision table (unchanged from PREREG_8B.md, deliberately)

| outcome | statement in the note |
|---|---|
| P1 and P2 hold | the corrected corpus picture replicates at 8B: map moves, boundaries do not |
| P1 holds, P2 fails | at 8B the corpus does move the boundaries; the small-model result was scale-limited |
| P1 fails | the corpus effect on the map shrinks with scale; reported as such |
| anchor fails | run reported as failed; no corpus statement at 8B |

If the anchor fails again at 400 prompts, a recipe difference (not budget) becomes the leading
explanation, the 8B arm is closed with no statement, and **we will not run a third attempt.**

## Cost and gating

Measured throughput from the first attempt: three 100-prompt fits took 9 h 47 min on one RTX A6000
including three model loads, about 117 s per prompt. Three 400-prompt fits are therefore about
**39 GPU-hours**, run as **three single-fit pods in parallel** (one arm each, about 13 h wall-clock)
at $0.33 to $0.53 per hour: **$13 to $21 in total.** The "about 12 h, about $7" figure in the
2026-09-06 ledger entry was wrong (it dropped the factor of three) and is corrected here before any
spend. No timing gate this time; the throughput is measured. Each pod writes a `DONE` marker and a
sha256 receipt with all-finite and layer-count checks; a scheduled check from the work box fetches,
verifies, and terminates each pod (CLAUDE.md lesson 22). No API key is placed on any pod.
