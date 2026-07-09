# Behavioral results — does the geometric workspace band have a functional correlate?

_2026-07-09. Anthropic's own causal experiments (verbal-report swap; ignition) run on
open models via jlens, then correlated against our geometric band statistic (`mid_sep`).
The geometry→function test. All runs on GPU (A6000 + A100); data in
`behavioral_correlation.csv` + per-model `verbal_report_*.json` / `ignition_*.json`._

## The headline: concept-propagation to the workspace tracks the band (ρ = +0.76)

N = 11 open models, band range 0.002 → 0.20, 4 families (Gemma-2/3, Qwen3/3.5, Llama,
Pythia, GPT-2). Rank correlations with the geometric `mid_sep`:

| behavioral measure | what it tests | Spearman ρ vs mid_sep | n |
|---|---|---|---|
| **share_span** | does an injected concept *resolve* in the J-lens readout (A-vs-B) | **+0.76** | 11 |
| swap_best | can steering the J-space direction flip the output | +0.53 | 11 |
| ign_sharp | is concept entry all-or-none (ignition), among resolvers | +0.49 | 6 |

**share_span is the clean result** — and it's close to a categorical separation:

| models | share_span | band |
|---|---|---|
| all 5 Gemma (gemma-2-2b/9b/27b, gemma-3-1b/4b) | **0.000** — concept never resolves | ≤ 0.046 |
| Qwen3-4B, Qwen3.5-2B/4B, Llama-3.1-8B | **0.96–0.98** — resolves cleanly | 0.056–0.20 |
| Pythia-70M (tiny) | 0.31 — barely | ~0 |

Read: **models with a geometric workspace band actually route an injected concept into
their verbalizable workspace; models without one (all of Gemma) do not.** ρ = 0.76 at
n = 11 exceeds the p < 0.01 critical value (~0.735) — a real geometry→function bridge.

## Honest limits of the behavioral result (do not overclaim)

- **Ignition is NOT well-supported.** ign_sharp only correlates at ρ = 0.49 and only n = 6
  (the 5 Gemmas never resolve concepts, so sharpness is undefined for them). Within the
  resolvers it's noisy (qwen3-4b band 0.056 → sharp 0.91, but qwen3.5-2b band 0.147 →
  sharp 0.27). So "all-or-none ignition tracks the band" is **not** established; the
  honest claim is only "among models that resolve concepts, some show sharp entry."
- **Steerability is not universal after all.** An earlier read (swap works everywhere)
  was based on gemma-2-9b (swap 0.82) — an *outlier*. Most Gemmas have low swap too
  (0.01–0.41), so swap partially tracks the band (ρ = 0.53), not a flat privileged-set
  signal. gemma-2-9b (high swap, zero concept-resolution, near-zero band) is a genuine
  odd case worth a sentence, not a headline.
- **The correlate is a two-cluster separation** (Gemma≈0 vs Qwen/Llama≈0.97) more than a
  smooth gradient — a strong effect, honestly characterized as clustered.
- **n = 11.** Three high-band models (qwen3-14b, qwen3.5-27b, gpt-oss-20b) failed on
  trivial infra issues (flaky HF download; gpt-oss mxfp4 needs accelerate) — retryable
  for ~$4 but the high band is already represented (qwen3.5-4b = 0.20), so we shipped
  n=11 on budget. Re-adding them is cheap future work.
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
