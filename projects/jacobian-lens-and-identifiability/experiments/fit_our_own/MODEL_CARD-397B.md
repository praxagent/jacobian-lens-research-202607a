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

The layer×layer CKA of the lens's token geometry shows three depth regimes, with a mid band
whose separation is the largest we have measured across 36 public lenses:

```
mid_sep = +0.3434   (n=24; interim n=16 read +0.3796)
within early/MID/late CKA = 0.890 / 0.937 / 0.814 (n=24)
```

Two controls travel with that number (see the atlas note for both): a Frobenius-matched
random-transport null is featureless (mid_sep -0.000113), and a distance-only surrogate with the
same similarity-versus-layer-distance profile reproduces 0.275 of the 0.343, so about a fifth of
the statistic is block structure beyond smooth decay with layer distance. On that fitted statistic
the 397B's excess over its surrogate (+0.125) is the largest among the 36 lenses. Cross-model
comparisons of the band statistic are only meaningful on a shared probe (own-vocabulary values,
including an earlier "1.6x Qwen3-14B" comparison in this card, are not comparable across models
and have been withdrawn); on the shared probe the 397B's mid_sep is 0.386 and its fitted
separation 0.472, the highest in the collection. We make no claim that the band grows with scale:
across the collection banding tracks size only loosely (rank correlation about +0.4 to +0.5).

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

- jlens `fit()` — `J_l = E[∂h_final/∂h_l]` for each of the **59 source layers, indices 0–58** (every decoder layer except the last). Qwen3.5-397B-A17B has **60 layers total (0–59)**; layer 59 is the **target** `h_final` that the lens transports *toward*, so it is not itself a source. The artifact holds 59 Jacobians (`source_layers = [0, 1, …, 58]`). The workspace-readout demos use the middle-third **band: layers 19–38** (20 layers).
- **n = 24 prompts**, wikitext-103 (`Salesforce/wikitext`, seed 0), `max_seq_len 128`,
  `dim_batch 16`
- Model loaded as `Qwen3_5MoeForConditionalGeneration` (⚠️ NOT `AutoModelForCausalLM`,
  which silently mismatches this checkpoint's `model.language_model.*` keys), text
  backbone via `jlens.Layout(path="model.language_model")`, bf16,
  `attn_implementation="eager"`, `device_map="auto"` with an even 110 GiB/GPU cap,
  on 8×H200. ~10 min/prompt.
- Prompt-count calibration: on smaller Qwen models the **band statistic** converges by
  n≈16 (n=8 is under-converged), and this lens's own interim read at n=16 (+0.380) is close to
  its final n=24 read (+0.343). **A later fit-budget sweep tempers this**: on the whole
  layer-by-layer map (linear CKA, two small models, seed-null reference), convergence holds from
  about 200 prompts up, one model at 25 prompts sits 14 times the seed null, and an 8B lens at
  100 prompts is still not converged. Under that experiment's frozen rule this 24-prompt lens
  carries a **budget caveat**: its band statistic is reproducible, its full map should not be
  assumed converged. See the
  [repo](https://github.com/praxagent/jacobian-lens-research-202607a) (`corpus_dependence`).

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
3. **Function, injected-concept readout**: Anthropic-style
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

- Fitted on **24 prompts** of English wikitext, far fewer than Neuronpedia's 1,000. The band
  statistic is reproducible at this budget (n=16 vs n=24 above), but our fit-budget sweep can only
  vouch for map-level convergence from about 200 prompts, so treat the map as budget-caveated and
  fit lenses on more prompts before microscopy. Boundaries and bands are also corpus-conditional:
  the same recipe on code moves the map by two orders of magnitude more than a WikiText resample
  does (fitted boundaries barely move).
- The lens targets the **text backbone only** (`model.language_model`); the vision tower
  and MTP heads are untouched.
- bf16 fit; fp16 export loses a little precision vs the fp32 canonical file.
- Our broader audit found the band's *behavioral* interpretation is confounded by model
  family (a no-band Qwen still resolves concepts; banded Gemmas don't) — see the write-up
  before drawing global-workspace conclusions from geometry alone.

## Provenance & links

- Fit code, receipts, logs, and the full audit:
  [praxagent/jacobian-lens-research-202607a](https://github.com/praxagent/jacobian-lens-research-202607a)
- Blog write-up: [PENDING LINK]
- Method: Anthropic, *Verbalizable Representations Form a Global Workspace in Language
  Models* (transformer-circuits, 2026) + [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)
- Related: [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens)
  (38 pre-fitted lenses, 70M–70B), Neel Nanda's independent replication (LessWrong, 2026).

License: Apache-2.0 (matching both the base model and jlens). Please cite/attribute
`praxagent` and link the repo if you build on this artifact.


## Layer x layer CKA atlas (added 2026-07-18)

The full 59x59 centered-kernel-alignment matrix behind the released band statistic,
recomputed via the hash-verified consumer path (this repo's lens + the base model's
lm_head, seed-0 n_probe=4096): `cka/cka_397b.npz` + heatmaps. Correctness gate: the
matrix reproduces the released mid_sep +0.343363 to |delta| = 2e-8. A Frobenius-matched
random-transport control (`cka/cka_397b_null.png`) is structure-free (mid_sep -0.000):
the early / mid-band / late block structure comes from the fitted lens, not from the
shared unembedding. That null is necessary and not sufficient: a distance-only surrogate
(every cell replaced by the mean CKA at that layer distance) reproduces 0.275 of the 0.343, so
read the block structure as the fifth of the statistic that exceeds smooth decay with distance
(fitted separation 0.407 real against 0.282 surrogate). Correction history: this card's earlier
"1.6x Qwen3-14B", "grows to frontier scale", "safely converged" and "near-perfectly" wordings
were withdrawn on 2026-09-06 after our own later measurements (36-lens atlas, fit-budget sweep,
8B extension); the previous text is in this repository's commit history. Generator:
`projects/jacobian-lens-and-identifiability/experiments/fit_our_own/cka_heatmap_397b.py`
in the research repo (commit c6c7bb1).
