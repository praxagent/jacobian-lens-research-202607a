# Behavioral results — does the geometric workspace band have a functional correlate?

_2026-07-09. Anthropic's own causal experiments (verbal-report swap; ignition) run on
open models via jlens, then correlated against our geometric band statistic (`mid_sep`).
The geometry→function test. All runs on GPU (A6000 + A100); data in
`behavioral_correlation.csv` + per-model `verbal_report_*.json` / `ignition_*.json`._

## The headline: behavioral workspace signatures track the geometric band — but not perfectly

N = **23** open models, band range 0.0007 → 0.21, 6 families (Gemma-2/3/4, Qwen2.5/3/3.5,
Llama, OLMo, GPT-OSS, Pythia, GPT-2). Rank correlations with the geometric `mid_sep`:

| behavioral measure | what it tests | Spearman ρ vs mid_sep | n | sig. |
|---|---|---|---|---|
| **share_span** | does an injected concept *resolve* in the J-lens readout (A-vs-B) | **+0.739** | 23 | p<0.001 |
| swap_best | can steering the J-space direction flip the output | **+0.741** | 21 | p<0.001 |
| ign_sharp | is concept entry all-or-none (ignition), among resolvers | **+0.452** | 15 | p=0.091 two-tailed (0.045 one-tailed) |

> **Estimator correction (2026-07-10, self-caught in adversarial verification):** all ρ
> in this file were originally computed with a hand-rolled Spearman that assigned
> *distinct* ranks to tied values in table order — with 8 Gemmas tied at share_span
> 0.000 in a mid_sep-sorted table, that systematically inflated ρ (share_span read
> +0.786; tie-corrected it is **+0.739**, matching `scipy.stats.spearmanr` exactly).
> `correlate.py` now uses average ranks; every ρ/p in this file is the corrected value.
> Historical snapshots quoted in the narrative (e.g. the n=13 share_span "0.835") were
> computed with the pre-fix estimator — the qualitative story they tell (more data made
> the correlation more honest) is unchanged. No CSV data was affected, only the
> correlation statistics.

`gpt-oss-20b` (OpenAI MoE, band 0.076) is the cleanest *intermediate* point — share_span
0.52, with only 13.5% of readouts resolving: a genuinely *partial* workspace, exactly where
its moderate band predicts. It's the datum that shows the relationship isn't purely two
clusters, and it's a sixth family that isn't Qwen or Gemma.

**The correlation is real and highly significant** (share_span/swap clear p<0.001), but
going higher n made it more honest, not more impressive: the n=13 snapshot (share_span
0.835, pre-fix estimator) was optimistic; at n=23 it's ρ≈0.74, a strong correlation, not
a near-identity.
(swap n=21: the two Gemma-4 base models don't answer the reportability prompt, so swap is
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
strong ρ≈0.74 is real but confounded: **Gemma uniquely has both a near-absent band and no
concept-resolution** (consistent with our KD-pretraining mechanism — see `hypotheses.md`),
and the cross-model correlation is largely that Gemma-vs-rest split. Two readings of
qwen2.5-7b-it remain open: a genuine family confound, or a `mid_sep` false-negative (it
hosts a workspace our statistic under-scores; the H5 self-critique). Fair claim: **the band
strongly predicts the behavior across families, but a matched-band comparison shows family,
not the band itself, is the causal variable.**

**share_span splits mostly by family — two clusters plus a couple of intermediate points:**

| models | share_span | band |
|---|---|---|
| all Gemma-2/3/4 (2b/9b/27b, 3-1b/4b/12b, 4-e2b/e4b) | **0.000** — never resolves | 0.0007–0.050 |
| Qwen (incl. low-band qwen2.5-7b-it), Llama, OLMo | **0.95–0.99** — resolves cleanly | 0.003–0.21 |
| GPT-OSS-20B (MoE) | **0.52** — partially resolves | 0.076 |
| Pythia-70M (tiny) | 0.31 — barely | ~0 |

### Shared-vocab robustness (2026-07-10) — the correlation survives, the family verdict sharpens

The tokenizer-confound re-sweep (`emergence_shared.csv`; see
`../jacobian_lens/results.md`) showed own-vocab probing systematically understated Gemma
bands. Re-running this correlation with the shared-probe `mid_sep`
(`correlate_shared.py` → `behavioral_correlation_shared.csv`), same 23 models:

| behavioral measure | ρ own-vocab | ρ shared-probe | n | p (shared, two-tailed) |
|---|---|---|---|---|
| share_span | +0.739 | **+0.534** | 23 | 0.009 |
| swap_best | +0.741 | **+0.524** | 21 | 0.015 |
| ign_sharp | +0.452 | **+0.508** | 15 | 0.053 |

(All values tie-corrected — see the estimator note above; they match
`scipy.stats.spearmanr` exactly. P's are two-tailed; shared ign_sharp at p=0.053 is
borderline and we don't claim it.)

**The correlation survives commensurable probes but attenuates** (share_span 0.739 →
0.534), and the attenuation is itself informative: under shared probes several Gemmas
turn out to have solid bands yet STILL never resolve (share_span 0.000). The matched-band
evidence gets much stronger than the gemma-4 pair we started with:

| matched pair (shared-probe band) | band | share_span |
|---|---|---|
| gemma-2-27b vs qwen3.5-2b-pt | 0.1134 vs 0.1143 | **0.000 vs 0.970** |
| gemma-3-12b vs qwen3.5-0.8b | 0.1137 vs 0.1149 | **0.000 vs 0.948** |
| gemma-4-e4b vs qwen3-4b | 0.098 vs 0.038 | **0.000 vs 0.976** (Gemma's band ~2.5× larger!) |

Under own-vocab probing the story was "Gemma uniquely lacks both the band and the
function." Shared probes revise it: **Gemma does not uniquely lack the band — it uniquely
lacks the function.** Concept-resolution is family-bound (Gemma 0.000 across all eight
Gemmas measured, regardless of band 0.005–0.114) while the geometry, measured fairly, is
not. Band remains not-necessary (qwen2.5-7b-it: shared band 0.014, resolves 0.965) and
now decisively not-sufficient. The geometry→function dissociation is the finding.

## Honest limits of the behavioral result (do not overclaim)

- **Gemma-4 verbal-report is undefined (base models don't answer).** `gemma-4-e2b`/`e4b`
  emit empty greedy output for "Think of a {cat}. Answer in one word:", so swap/reportability
  are undefined and excluded (swap n=21). Their **ignition `share_span` is valid** — it
  interpolates embeddings and reads the J-lens, no answer needed — so the mirror test (banded
  Gemma → 0.000) stands. (Fixing the runner needed an embed-output-hook injection because
  Gemma-4's forward rejects arbitrary `inputs_embeds`; validated to reproduce qwen3-4b's
  number within bf16 noise.)
- **Ignition is the weakest leg (ρ=0.452, n=15, p=0.091 two-tailed — not significant;
  0.045 one-tailed).** It is only defined
  for the 15 models whose concepts resolve at all, and it is not monotone within them
  (qwen3.5-0.8b band 0.146 → sharp 0.00; olmo-3 band 0.076 → sharp 0.39 are low outliers).
  Fair claim: "ignition sharpness weakly tracks the band among models that form one," not
  "the workspace shows human-like ignition."
- **Steerability tracks the band** (ρ=0.741, n=21): most Gemmas have low swap (0.01–0.41),
  Qwen/Llama/OLMo high (0.79–1.00). gemma-2-9b (swap 0.82, zero concept-resolution, near-zero
  band) remains a genuine odd case — steerable yet nothing resolves.
- **The correlate is closer to two clusters than a smooth gradient** (Gemma≈0 vs the rest
  ≈0.97), with qwen2.5-7b-it in the "resolves" cluster despite a Gemma-like band and the
  Gemma-4s in the "never resolves" cluster despite mid-range bands — the matched-band
  datum that makes this a family split, not a band gradient.
- **n = 23** (share_span; swap n=21, ign_sharp n=15). Added this round: qwen2.5-7b-it,
  qwen3-1.7b/8b, qwen3.5-0.8b/9b-pt, olmo-3-1025-7b (new family), gemma-3-12b (lowest band
  0.0007, never resolves — as predicted), gemma-4-e2b/e4b (banded Gemma → 0.000, the mirror
  test), and gpt-oss-20b (now works via mxfp4→bf16 dequant; share_span 0.52 — the
  intermediate point). No models remain failing.
- Geometry↔function here is *correlational*; we did not run Anthropic's full ablation
  battery. `share_span`/`swap`/`ign` are our operationalizations of their experiments.

## Method

> **Naming correction (2026-07-10, from an external review of the demo code that
> reuses this protocol):** what this file calls the "causal swap" is ADDITIVE
> steering — the paper's *verbal-introspection* injection recipe (unit direction ×
> mean residual norm × strength) applied to the verbal-report task. The paper's own
> verbal-report "swap" is a coordinate clamp/exchange, which we have not implemented.
> `swap_best` therefore measures additive-injection flip success. Conclusions are
> unaffected (the metric is internally consistent across all models), but the name
> overstated equivalence to the paper's intervention.

Both tests reuse Anthropic's Apache-2.0 prompt sets (`verbal-report.json`,
`ignition.json`) and jlens. **verbal-report swap:** add the J-lens steering vector
(strength·mean_norm·(unit dir_cand − unit dir_ans)) at the band layers, check if the
output flips (strength-0 control ≈ 0 on all models — clean). **ignition:** interpolate a
slot token's input embedding A↔B (α 0→1), measure the J-lens A-share per band layer;
`share_span` = range swept, `ign_sharp` = fraction of curves with a sharp (<0.25-α)
10→90% transition. Band = middle third of fitted layers. Runners: `verbal_report.py`,
`ignition.py`; merge/correlation: `correlate.py`.
