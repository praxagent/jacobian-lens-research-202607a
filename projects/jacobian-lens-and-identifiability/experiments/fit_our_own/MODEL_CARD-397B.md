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
2. **Held-out fidelity evals** (12 prompts, seed 1, disjoint from the fit corpus):
   lens-vs-model top-1/top-10 agreement and KL by depth: [EVALS TABLE PENDING]
3. **Function — the lens extracts workspace content**: Anthropic-style ignition test run
   through this lens on the 397B itself (interpolated concept injection at a carrier
   slot, read out via the lens at band layers): [IGNITION RESULTS PENDING]
4. **Reproduce our number from this artifact** (CPU-only, no GPU needed):

```python
# pip install torch safetensors numpy huggingface_hub
# 1) download the fp16 lens from this repo + lm_head shard from Qwen/Qwen3.5-397B-A17B
# 2) J_l = lens["J"][l].float(); probe 4096 vocab rows (numpy default_rng(0))
# 3) geometry_l = U_probe @ J_l ; layer-x-layer linear CKA; band statistic
# full script: experiments/jacobian_lens/ in the repo below
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
