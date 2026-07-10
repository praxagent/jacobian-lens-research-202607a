---
license: apache-2.0
base_model: Qwen/Qwen3.5-397B-A17B
tags:
  - interpretability
  - jacobian-lens
  - global-workspace
  - mechanistic-interpretability
---

# Jacobian Lens for Qwen3.5-397B-A17B

A fitted **Jacobian lens** ([Anthropic's jlens](https://github.com/anthropics/jacobian-lens),
Apache-2.0) for **Qwen/Qwen3.5-397B-A17B** — to our knowledge the first publicly
available Jacobian lens for a ~0.4T-parameter model (the Neuronpedia collection spans
70M–70B). Fitted and validated by [praxagent](https://github.com/praxagent) as part of an
independent replication/audit of Anthropic's *Verbalizable Representations Form a Global
Workspace in Language Models* (2026).

## Headline measurement

The layer×layer CKA of the lens's token geometry shows the strongest "workspace band"
we have measured on any model:

```
mid_sep = +0.3434   (n=24; interim n=16 read +0.3796)
within early/MID/late CKA = 0.890 / 0.937 / 0.814 (n=24)
```

— 1.6× the strongest model in the 36-model Neuronpedia sweep (Qwen3-14B: 0.211). The
workspace band **grows** to frontier scale in this lineage; it is not a small-model
transient.

## Files

| file | what |
|---|---|
| `jlens/wikitext/qwen35_397b.pt` | fp32 lens (canonical jlens format, `JacobianLens.load`-able) |
| `jlens/wikitext/qwen35_397b_fp16.pt` | fp16 Jacobians (`{"J": {layer: tensor}, "source_layers": [...]}`), half the size |
| `band.json` | the band statistic + per-block CKA numbers |
| `evals.json` | held-out lens-fidelity evals (below) |
| `ignition_qwen3.5-397b.json` | workspace-extraction (ignition) results run through this lens |
| `SHA256SUMS` | checksums, matching the fit machine's originals |

## Fit configuration (exact)

- jlens `fit()` — `J_l = E[∂h_final/∂h_l]`, all 60 source layers, target = final layer
- **n = 24 prompts**, wikitext-103 (`Salesforce/wikitext`, seed 0), `max_seq_len 128`,
  `dim_batch 16`
- Model loaded as `Qwen3_5MoeForConditionalGeneration` (⚠️ NOT `AutoModelForCausalLM`,
  which silently mismatches this checkpoint's `model.language_model.*` keys), text
  backbone via `jlens.Layout(path="model.language_model")`, bf16,
  `attn_implementation="eager"`, `device_map="auto"` with an even 110 GiB/GPU cap,
  on 8×H200. ~10 min/prompt.
- Prompt-count calibration: on smaller Qwen models the band statistic converges by
  n≈16 (n=8 is under-converged); n=24 is safely in the converged regime. See the
  [repo](https://github.com/praxagent/research-and-replications) for the curve.

## Validation (what makes this trustworthy)

1. **Pipeline calibration**: the same fitting code reproduces Neuronpedia's gpt2 lens at
   mean CKA 0.9992 and lands within sampling noise of Neuronpedia's mid_sep on
   qwen3-4b and qwen3.5-0.8b.
2. **Fidelity — A.6-faithful, calibrated against published lenses.** The canonical
   readout is rank-based, not absolute agreement (a healthy J-lens is deliberately the
   *worst* absolute next-token predictor mid-network — paper A.6). Our lens: unembed-path
   identity holds exactly; motor-layer argmax agreement rises monotonically with depth
   (mid 0.000 → last fitted layer **0.5625**); this matches known-good **Neuronpedia**
   lenses on the identical eval (qwen3-4b **0.722**, the architecture-matched qwen3.5-0.8b
   **0.549**). J-lens beats the logit-lens baseline at pass@10 intermediate recovery.
   Files: `evals_v2_397b.json`, `calg2_neuronpedia_calibration.log`. (An earlier eval,
   `evals_v1_misspecified.json`, used absolute agreement — the wrong metric — and is
   retained only as a transparency receipt.)
3. **Function — the lens extracts workspace content (near-perfectly)**: Anthropic-style
   ignition test run through this lens on the 397B itself — interpolated concepts
   injected at a carrier slot resolve in the lens readout with **median share_span
   0.988** (full 0.006→0.995 sweep), 94.6% of 480 band-layer readouts resolving, 83.7%
   with sharp (<0.25-α) all-or-none transitions. Raw results in
   `ignition_qwen3.5-397b.json`.
4. **Consumer-path check — run, PASSED exactly (2026-07-10).** A 7 GB CPU box
   downloaded this repo's lens (sha256 `668c3bf1…99e97`, byte-identical to the fit
   pod's original) plus only the lm_head shard of the base model, and recomputed the
   band statistic from the public copy: **mid_sep = +0.343363**, agreeing with the
   shipped `band.json` to 2×10⁻⁸. Runnable protocol:
   `experiments/fit_our_own/consumer_check_397b.py` in the repo below.
5. **Independent behavioral trial — PASSED (2026-07-10).** A fresh 8×H200 pod pulling
   only the HF artifacts (this lens + the base model), pre-registered protocol: hidden
   two-hop bridge entities (in neither prompt nor output, 0 leaks in 20/20) read at
   **median rank 43 of 248,320** — vs 620 for identity transports and 7,121 for
   scale-matched random transports through the identical readout path; and **32/50
   causally steered answer flips vs 0/50** at strength zero and 0/50 for norm-matched
   random directions. Full receipts: `experiments/lens_demo/` in the repo below.
6. **Reproduce our number from this artifact** (CPU-only, no GPU needed):

```python
# pip install torch safetensors numpy huggingface_hub
# 1) download the fp16 lens from this repo + lm_head shard from Qwen/Qwen3.5-397B-A17B
# 2) J_l = lens["J"][l].float(); probe 4096 vocab rows (numpy default_rng(0))
# 3) geometry_l = U_probe @ J_l ; layer-x-layer linear CKA; band statistic
# full script: experiments/fit_our_own/consumer_check_397b.py in the repo below
```

## Caveats (read before using)

- Fitted on **24 prompts** of English wikitext. Converged for the band statistic per our
  calibration, but far fewer than Neuronpedia's n≈1000; per-token/direction analyses may
  be noisier than band-level geometry. Fit lenses on more prompts before microscopy.
- The lens targets the **text backbone only** (`model.language_model`); the vision tower
  and MTP heads are untouched.
- bf16 fit; fp16 export loses a little precision vs the fp32 canonical file.
- Our broader audit found the band's *behavioral* interpretation is confounded by model
  family (a no-band Qwen still resolves concepts; banded Gemmas don't) — see the write-up
  before drawing global-workspace conclusions from geometry alone.

## Provenance & links

- Fit code, receipts, logs, and the full audit:
  [praxagent/research-and-replications](https://github.com/praxagent/research-and-replications)
- Blog write-up: [PENDING LINK]
- Method: Anthropic, *Verbalizable Representations Form a Global Workspace in Language
  Models* (transformer-circuits, 2026) + [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)
- Related: [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens)
  (38 pre-fitted lenses, 70M–70B), Neel Nanda's independent replication (LessWrong, 2026).

License: Apache-2.0 (matching both the base model and jlens). Please cite/attribute
`praxagent` and link the repo if you build on this artifact.
