# Feasibility snapshot — artifacts and access (2026-07-14)

Integrity §5: external resources that a selection or training rule depends on are
part of the design and can change under you. This file snapshots every artifact the
replication needs, its access status, and the retrieval date. Verified 2026-07-14 with
the campaign `HF_TOKEN`.

## Models (all downloadable — HTTP 200 on `config.json` with our token)

| Model | HF id | Access | Role |
|---|---|---|---|
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | 200 | **primary / cheapest**, gold labels |
| Llama-3.3-70B-Instruct | `meta-llama/Llama-3.3-70B-Instruct` | 200 | gold labels, we own a J-lens |
| Gemma-3-12B-IT | `google/gemma-3-12b-it` | 200 (gated: manual, granted) | **paper's exact model**; multimodal, use text path |

## Hallucination labels (the expensive grader output — public)

`obalcells/longfact-annotations` (Oscar Balcells Obeso et al., the LongFact++ /
"Real-Time Detection of Hallucinated Entities" work, arXiv 2509.03531,
hallucination-probes.com). Token-level entity annotations.

- **Configs (per model):** `Llama-3.3-70B-Instruct`, `Meta-Llama-3.1-8B-Instruct`,
  `Mistral-Small-24B-Instruct-2501`, `Qwen2.5-7B-Instruct`, `gemma-2-9b-it`.
- **Features:** `subset, model, conversation, annotations, canary`.
- **Splits:** `train / validation / test` (proper holdout exists).
- Also: `obalcells/hallucination-heads-longfact-augmented` (+ medical/legal/biography/
  citations augmented variants) with `verification`-style fields.

**⚠️ Label gap for the Gemma-3 arm.** Gold labels exist for **gemma-2-9b-it**, NOT
gemma-3-12b-it. So the paper's exact model has **no public labels**. Options for the
Gemma arm, to be decided when we sequence it (last):
1. **Grade Gemma-3-12B completions** with an LLM+web-search grader (cost + a
   stochastic-grader confound; the very step the public data otherwise saves).
2. **Swap to gemma-2-9b-it** for the Gemma-family arm — gold labels + Gemma Scope
   SAE (`gemma-scope-9b-pt-res`), cheaper, but not the paper's 12B model.
3. Restrict the Gemma-3-12B arm to the **label-free** comparisons only (unsupervised
   logit/J-lens vs SAE readout agreement), which needs no gold labels.

The Llama-3.1-8B and Llama-3.3-70B arms have gold labels directly and carry no
grader cost or confound.

## SAEs (for the sparse-reader arm — all resolve)

| Model | SAE repo | Notes |
|---|---|---|
| Llama-3.1-8B | `EleutherAI/sae-llama-3.1-8b-32x` (also `-64x`), `Goodfire/Llama-3.1-8B-Instruct-SAE-l19` | residual-stream |
| Llama-3.3-70B | `Goodfire/Llama-3.3-70B-Instruct-SAE-l50` | residual-stream, layer 50 |
| Gemma-3-12B-IT | `google/gemma-scope-2-12b-it` | **Gemma Scope 2** — DeepMind SAE suite on the Gemma-3 family incl. 12B-IT; `resid_post/attn_out/mlp_out` at 25/50/65/85% depth, multiple widths/L0. Exact model match. |

## J-lens (unsupervised fitted reader)

- **Llama-3.3-70B:** we already own a fitted J-lens from the campaign (Neuronpedia lens).
- **Llama-3.1-8B, Gemma-3-12B:** none yet — must fit (cheap for 8B, moderate for 12B).
  TJ approved fitting J-lenses (logit lens + fitted J-lens for the unsupervised arm).
- **Logit lens:** free on any model (identity transport → unembed); the mandated
  cheapest-prior-method baseline (integrity §5) and the reader our
  workspace-under-pressure lie-catch used.

## Consequence for the design
- Feasibility gate PASSED for all core artifacts.
- Sequence cheapest-first: **Llama-3.1-8B** validates the whole pipeline on gold
  labels at minimal cost, then Llama-3.3-70B, then Gemma-3-12B (label decision above).
- Snapshot the exact dataset revision (commit hash) at freeze time before caching.
