# Behavioral results — does the geometric workspace band have a functional correlate?

_2026-07-09. Anthropic's own causal experiments (verbal-report swap; ignition) run on
open models via jlens, then correlated against our geometric band statistic (`mid_sep`).
The geometry→function test. All runs on GPU (A6000 + A100); data in
`behavioral_correlation.csv` + per-model `verbal_report_*.json` / `ignition_*.json`._

## The headline: behavioral workspace signatures track the geometric band — but not perfectly

N = **22** open models, band range 0.0007 → 0.21, 5 families (Gemma-2/3/4, Qwen2.5/3/3.5,
Llama, OLMo, Pythia, GPT-2). Rank correlations with the geometric `mid_sep`:

| behavioral measure | what it tests | Spearman ρ vs mid_sep | n | sig. |
|---|---|---|---|---|
| **share_span** | does an injected concept *resolve* in the J-lens readout (A-vs-B) | **+0.784** | 22 | p<0.001 |
| swap_best | can steering the J-space direction flip the output | **+0.759** | 20 | p<0.001 |
| ign_sharp | is concept entry all-or-none (ignition), among resolvers | **+0.473** | 14 | p<0.05 |

**The correlation is real and highly significant** (share_span/swap clear p<0.001), but
going higher n made it more honest, not more impressive: the n=13 snapshot (share_span
0.835) was optimistic; at n=22 it's ρ≈0.78, a strong correlation, not a near-identity.
(swap n=20: the two Gemma-4 base models don't answer the reportability prompt, so swap is
undefined for them — their `share_span` from the interpolation test is valid and included.)

### ⚠️ The correlation is largely a FAMILY effect — two confound-breakers prove it

We ran two deliberate confound-breakers to separate "the band drives the behavior" from
"the model *family* drives both." Both committed their predictions before the run; **both
came out against the clean band→behavior story:**

- **A Qwen with no band still resolves.** `qwen2.5-7b-it`: `mid_sep` = 0.0029 (Gemma-like,
  near-zero) yet share_span **0.965**, frac-resolved 0.83. Predicted to fail like Gemma; it
  didn't. Band **not necessary** for the function.
- **Banded Gemmas still don't resolve.** `gemma-4-e2b`/`gemma-4-e4b`: bands 0.047/0.050
  (higher than several *resolving* models) yet share_span **0.000** — the concept never
  reaches the readout, exactly like the bandless Gemmas. Band **not sufficient** within
  Gemma.

The cleanest single datum is a **matched-band pair**: at essentially the same geometric band,
`gemma-4-e4b` (0.050) resolves **0.000** while `qwen3-4b` (0.056) resolves **0.976**. Same
band, opposite behavior — the discriminating variable is *family*, not `mid_sep`. So the
strong ρ≈0.78 is real but confounded: **Gemma uniquely has both a near-absent band and no
concept-resolution** (consistent with our KD-pretraining mechanism — see `hypotheses.md`),
and the cross-model correlation is largely that Gemma-vs-rest split. Two readings of
qwen2.5-7b-it remain open: a genuine family confound, or a `mid_sep` false-negative (it
hosts a workspace our statistic under-scores; the H5 self-critique). Fair claim: **the band
strongly predicts the behavior across families, but a matched-band comparison shows family,
not the band itself, is the causal variable.**

**share_span is near-categorical by family:**

| models | share_span | band |
|---|---|---|
| all Gemma-2/3/4 (2b/9b/27b, 3-1b/4b/12b, 4-e2b/e4b) | **0.000** — never resolves | 0.0007–0.050 |
| Qwen (incl. low-band qwen2.5-7b-it), Llama, OLMo | **0.95–0.99** — resolves cleanly | 0.003–0.21 |
| Pythia-70M (tiny) | 0.31 — barely | ~0 |

## Honest limits of the behavioral result (do not overclaim)

- **Gemma-4 verbal-report is undefined (base models don't answer).** `gemma-4-e2b`/`e4b`
  emit empty greedy output for "Think of a {cat}. Answer in one word:", so swap/reportability
  are undefined and excluded (swap n=20). Their **ignition `share_span` is valid** — it
  interpolates embeddings and reads the J-lens, no answer needed — so the mirror test (banded
  Gemma → 0.000) stands. (Fixing the runner needed an embed-output-hook injection because
  Gemma-4's forward rejects arbitrary `inputs_embeds`; validated to reproduce qwen3-4b's
  number within bf16 noise.)
- **Ignition is the weakest leg (ρ=0.473, n=14, borderline p<0.05).** It is only defined
  for the 14 models whose concepts resolve at all, and it is not monotone within them
  (qwen3.5-0.8b band 0.146 → sharp 0.00; olmo-3 band 0.076 → sharp 0.39 are low outliers).
  Fair claim: "ignition sharpness weakly tracks the band among models that form one," not
  "the workspace shows human-like ignition."
- **Steerability tracks the band** (ρ=0.759, n=20): most Gemmas have low swap (0.01–0.41),
  Qwen/Llama/OLMo high (0.79–1.00). gemma-2-9b (swap 0.82, zero concept-resolution, near-zero
  band) remains a genuine odd case — steerable yet nothing resolves.
- **The correlate is closer to two clusters than a smooth gradient** (Gemma≈0 vs the rest
  ≈0.97), with qwen2.5-7b-it in the "resolves" cluster despite a Gemma-like band and the
  Gemma-4s in the "never resolves" cluster despite mid-range bands — the matched-band
  datum that makes this a family split, not a band gradient.
- **n = 22** (share_span; swap n=20, ign_sharp n=14). Added this round: qwen2.5-7b-it,
  qwen3-1.7b/8b, qwen3.5-0.8b/9b-pt, olmo-3-1025-7b (new family), gemma-3-12b (lowest band
  0.0007, never resolves — as predicted), gemma-4-e2b/e4b (banded Gemma → 0.000, the mirror
  test). Still failing: gpt-oss-20b (mxfp4 — dequantize-to-bf16 retry in progress).
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
