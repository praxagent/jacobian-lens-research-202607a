# Behavioral results — does the geometric workspace band have a functional correlate?

_2026-07-09. Anthropic's own causal experiments (verbal-report swap; ignition) run on
open models via jlens, then correlated against our geometric band statistic (`mid_sep`).
The geometry→function test. All runs on GPU (A6000 + A100); data in
`behavioral_correlation.csv` + per-model `verbal_report_*.json` / `ignition_*.json`._

## The headline: behavioral workspace signatures track the geometric band — but not perfectly

N = **20 valid** open models (22 run; 2 excluded as broken — see below), band range
0.0007 → 0.21, 5 families (Gemma-2/3, Qwen2.5/3/3.5, Llama, OLMo, Pythia, GPT-2). Rank
correlations with the geometric `mid_sep`:

| behavioral measure | what it tests | Spearman ρ vs mid_sep | n | sig. |
|---|---|---|---|---|
| **share_span** | does an injected concept *resolve* in the J-lens readout (A-vs-B) | **+0.735** | 20 | p<0.001 |
| swap_best | can steering the J-space direction flip the output | **+0.736** | 20 | p<0.001 |
| ign_sharp | is concept entry all-or-none (ignition), among resolvers | **+0.460** | 14 | p<0.05 |

**The correlation is real and highly significant, but going higher n made it more honest,
not more impressive.** The n=13 snapshot (share_span 0.835, ign_sharp 0.714) was optimistic;
adding 7 models pulled share_span to 0.735 and ign_sharp to a borderline 0.460. Both
share_span and swap still clear the p<0.001 critical value (~0.68 at n=20) — the
geometry→function bridge holds — but it is a ρ≈0.74 correlation, not a near-identity.

### ⚠️ The counterexample that matters: qwen2.5-7b-it (predicted wrong, reported honestly)

We added `qwen2.5-7b-it` as a deliberate confound-breaker: a **Qwen with essentially no
geometric band** (`mid_sep` = 0.0029, Gemma-like). Prediction (committed before the run):
if the band drives the behavior, it should *fail to resolve* like Gemma. **It did the
opposite** — share_span **0.965**, frac-resolved 0.83, ign_sharp 0.87: a full behavioral
workspace with no geometric band. So the band is **not necessary** for the function. Two
readings, both important:
1. **Family confound.** Concept-resolution may track family-level training factors (every
   Qwen we ran resolves; every Gemma fails) that *correlate* with the band across models
   but are not the band itself. Our cross-model ρ is then partly a Qwen-vs-Gemma effect.
2. **Metric false-negative.** `mid_sep` keys on a specific layer×layer block structure;
   qwen2.5-7b-it may host a real workspace whose geometry our statistic under-scores (the
   H5 self-critique). The function is there; our *measurement* missed it.

Either way, the clean "geometric band = behavioral workspace" identity is too strong. Fair
claim: **the band strongly predicts the behavior across models (ρ≈0.74, p<0.001), with at
least one clear dissociation — a low-band Qwen with full function.** That is a richer, more
credible result than the n=13 number, and exactly the kind of limit an independent audit
should surface.

**share_span still separates most models cleanly** (the counterexample aside):

| models | share_span | band |
|---|---|---|
| Gemma-2/3 (2b/9b/27b, 3-1b/4b/12b) | **0.000** — concept never resolves | ≤ 0.046 |
| Qwen (incl. qwen2.5-7b-it), Llama, OLMo | **0.95–0.99** — resolves cleanly | 0.003–0.21 |
| Pythia-70M (tiny) | 0.31 — barely | ~0 |

## Honest limits of the behavioral result (do not overclaim)

- **Two Gemma-4 runs are broken, not negatives.** `gemma-4-e2b`/`gemma-4-e4b` (the banded
  Gemmas we ran as the other confound-breaker) produced **empty greedy answers** for every
  category and no ignition output — a runner/chat-template incompatibility with the Gemma-4
  base models, not a scientific "doesn't resolve." They are **excluded** from all
  correlations (share_span sentinel −1). Fixing the runner for Gemma-4 and re-running is
  open work — it's the missing test of whether a *banded* Gemma resolves.
- **Ignition is the weakest leg (ρ=0.460, n=14, borderline p<0.05).** It is only defined
  for the 14 models whose concepts resolve at all, and it is not monotone within them
  (qwen3.5-0.8b band 0.146 → sharp 0.00; olmo-3 band 0.076 → sharp 0.39 are low outliers).
  Fair claim: "ignition sharpness weakly tracks the band among models that form one," not
  "the workspace shows human-like ignition."
- **Steerability tracks the band** (ρ=0.736): most Gemmas have low swap (0.01–0.41), Qwen/
  Llama/OLMo high (0.79–1.00). gemma-2-9b (swap 0.82, zero concept-resolution, near-zero
  band) remains a genuine odd case — steerable yet nothing resolves.
- **The correlate is closer to two clusters than a smooth gradient** (Gemma≈0 vs the rest
  ≈0.97), now with qwen2.5-7b-it sitting in the "resolves" cluster despite a Gemma-like
  band — the datum that turns the clean split into an honest ρ≈0.74.
- **n = 20 valid.** Added this round: qwen2.5-7b-it, qwen3-1.7b, qwen3-8b, qwen3.5-0.8b,
  qwen3.5-9b-pt, olmo-3-1025-7b (new family), gemma-3-12b (lowest band 0.0007, never
  resolves — as predicted). Still failing: gpt-oss-20b (mxfp4) and the 2 Gemma-4 runs.
- Geometry↔function here is *correlational*; we did not run Anthropic's full ablation
  battery. `share_span`/`swap`/`ign` are our operationalizations of their experiments.

## Method

Both tests reuse Anthropic's Apache-2.0 prompt sets (`verbal-report.json`,
`ignition.json`) and jlens. **verbal-report swap:** add the J-lens steering vector
(strength·mean_norm·(unit dir_cand − unit dir_ans)) at the band layers, check if the
output flips (strength-0 control ≈ 0 on all models — clean). **ignition:** interpolate a
slot token's input embedding A↔B (α 0→1), measure the J-lens A-share per band layer;
`share_span` = range swept, `ign_sharp` = fraction of curves with a sharp (<0.25-α)
10→90% transition. Band = middle third of fitted layers. Runners: `verbal_report.py`,
`ignition.py`; merge/correlation: `correlate.py`.
