# Behavioral results — does the geometric workspace band have a functional correlate?

_2026-07-09. Anthropic's own causal experiments (verbal-report swap; ignition) run on
open models via jlens, then correlated against our geometric band statistic (`mid_sep`).
The geometry→function test. All runs on GPU (A6000 + A100); data in
`behavioral_correlation.csv` + per-model `verbal_report_*.json` / `ignition_*.json`._

## The headline: behavioral workspace signatures track the geometric band

N = 13 open models, band range 0.002 → 0.21, 4 families (Gemma-2/3, Qwen3/3.5, Llama,
Pythia, GPT-2). Rank correlations with the geometric `mid_sep`:

| behavioral measure | what it tests | Spearman ρ vs mid_sep | n | sig. |
|---|---|---|---|---|
| **share_span** | does an injected concept *resolve* in the J-lens readout (A-vs-B) | **+0.835** | 13 | p<0.001 |
| ign_sharp | is concept entry all-or-none (ignition), among resolvers | **+0.714** | 8 | p<0.05 |
| swap_best | can steering the J-space direction flip the output | **+0.698** | 13 | p<0.01 |

**All three behavioral measures track the band; share_span most strongly.** The two
highest-band models anchor the high end exactly as predicted — qwen3-14b (band 0.21) →
ignition-sharp 0.98 / share_span 0.99, qwen3.5-27b (0.20) → sharp 0.97 / span 0.99.
(These n=13 numbers supersede an earlier n=11 pass; adding the two high-band anchors
raised every correlation — share_span 0.76→0.835, ign_sharp 0.49→0.714, swap 0.53→0.698.)

**share_span is the cleanest result** — close to a categorical separation:

| models | share_span | band |
|---|---|---|
| all 5 Gemma (gemma-2-2b/9b/27b, gemma-3-1b/4b) | **0.000** — concept never resolves | ≤ 0.046 |
| Qwen3-4B, Qwen3.5-2B/4B, Llama-3.1-8B | **0.96–0.98** — resolves cleanly | 0.056–0.20 |
| Pythia-70M (tiny) | 0.31 — barely | ~0 |

Read: **models with a geometric workspace band actually route an injected concept into
their verbalizable workspace; models without one (all of Gemma) do not.** ρ = 0.835 at
n = 13 clears the p < 0.001 critical value (~0.78) — a real geometry→function bridge.

## Honest limits of the behavioral result (do not overclaim)

- **Ignition tracks the band but on n=8.** ign_sharp correlates at ρ = 0.714 (p<0.05) —
  now genuinely supported (the two high-band anchors are also the sharpest: qwen3-14b
  0.98, qwen3.5-27b 0.97), but it is only defined for the 8 models whose concepts resolve
  at all (the 5 Gemmas never resolve, so sharpness is undefined for them — itself part of
  the finding). Within the resolvers it's not perfectly monotone (qwen3.5-2b band 0.147 →
  sharp 0.27 is a low outlier). Fair claim: "ignition sharpness tracks the band among
  models that form one," not "the workspace shows human-like ignition."
- **Steerability also tracks the band (not universal).** An earlier read (swap works
  everywhere) was wrong — based on gemma-2-9b (swap 0.82), an *outlier*. Most Gemmas have
  low swap (0.01–0.41), so swap tracks the band too (ρ = 0.698). gemma-2-9b (high swap,
  zero concept-resolution, near-zero band) remains a genuine odd case worth a sentence.
- **The correlate is a two-cluster separation** (Gemma≈0 vs Qwen/Llama≈0.97) more than a
  smooth gradient — a strong effect, honestly characterized as clustered.
- **n = 13.** Two of the three earlier infra failures were recovered (qwen3-14b band 0.21,
  qwen3.5-27b band 0.20) and both landed as the highest-band anchors — adding them raised
  every correlation. gpt-oss-20b still fails (mxfp4 backward incompatible with jlens's
  repeated-backward under the current stack); it is the one remaining gap. Going higher n
  means fitting behavioral runs on more of the 38 lensed models — cheap but GPU-bound.
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
