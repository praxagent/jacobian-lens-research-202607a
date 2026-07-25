# Pre-registration: are J-space depth blocks causal boundaries, or only geometric ones?

**Status: FROZEN before any patching run.** Committed before results exist. Written
2026-07-25. No patching data had been collected at the time of this commit; the only prior
numbers used are the atlas's published CKA maps and fitted segmentations.

## The question

The J-space atlas shows that layer-by-layer CKA of a Jacobian lens forms contiguous depth
blocks, which Anthropic's workspace paper labels sensory / workspace / motor. Everything we
have established so far is **geometric**: the readout geometry is similar within a band and
changes across a band boundary. That is a statement about representation similarity, not
about computation.

This experiment asks whether those boundaries are **causal**: is a block a region within
which activations are interchangeable in a way they are not across a boundary?

## Prediction and its confound

The naive prediction, "activation patches transfer better within a block than across a
boundary," is **not testable as stated**, because patch transfer degrades with layer
distance `\|i-j\|` regardless of any block structure. A within-block pair is on average
closer than a cross-block pair, so distance alone would confirm it.

The testable version is a **discontinuity at the boundary after controlling for distance**.

## Design

Cross-layer activation patching. For a source prompt S and target prompt T that differ in
one attribute, capture S's residual stream at layer `i` and insert it at layer `j` of T's
forward pass, at the final token position, then measure how far T's output moves toward S's.

Normalized patch effect, per (i, j) and prompt pair:

    E(i,j) = (LD_patched - LD_target) / (LD_source - LD_target)

where `LD = logit(answer_S) - logit(answer_T)`. E = 1 means the patch fully carried the
source's answer; E = 0 means no movement. Pairs whose clean denominator is degenerate
(`|LD_source - LD_target| < 1.0`) are excluded before any patching (rule frozen here).

**Blocks** are the atlas's own fitted 3-segmentation of each model's shared-vocabulary CKA
map (`atlas_stage_a.fitted_seg`), computed before this experiment:

| model | layers | boundaries | block sizes | fitted_sep |
|---|---|---|---|---|
| gemma-3-270m (pilot, CPU, free) | 17 | 3, 15 | 3 / 12 / 2 | +0.227 |
| qwen3-8b (confirmation, GPU) | 35 | 19, 22 | 19 / 3 / 13 | +0.359 |

**Primary analysis.** Over all ordered pairs `i != j`, fit

    E(i,j) ~ distance-dummies(|i-j|) + beta * crosses(i,j)

`crosses` is 1 if `i` and `j` fall in different fitted blocks. `beta` is the boundary
effect with layer distance fully absorbed by per-distance dummies. Prediction: **beta < 0**.

**Primary control: the random-boundary null.** Recompute `beta` for 1,000 random
3-segmentations with the same block-size multiset, and compare the observed `beta` to that
null distribution. This controls for distance, for any smooth positional trend, and for the
particular block sizes. Reported as a two-sided permutation p-value.

**Power (checked before freezing, from geometry only).** Distances with at least 3 within
and 3 cross pairs: 1-10 for the pilot (138 within / 92 cross usable) and 1-17 for qwen3-8b
(502 within / 382 cross).

## Predictions

- **P1 (primary, confirmatory).** `beta < 0` and outside the random-boundary null at
  `p < 0.05`, in both models.
- **P2 (secondary, exploratory).** The effect is larger at the early boundary than the late
  one. Labeled exploratory; no decision rides on it.
- **Sanity gate (must pass or the run is void).** `E(i,i)` self-patches, run as a separate
  arm, must be near 1 (>= 0.9 median), and `E` must decrease with `|i-j|`. A pipeline that
  fails this is broken, and no verdict may be read from it.

## Decision rules (frozen)

| outcome | verdict |
|---|---|
| P1 holds in both models | **CONFIRMED**: block boundaries are causal, not merely geometric |
| P1 holds in one, null in the other | **PARTIAL**: report both, no general claim |
| P1 null in both | **NULL**: the CKA bands describe geometry and do not partition computation |
| sanity gate fails | **VOID**: fix the pipeline, re-run, do not interpret |

A null is a publishable result and will be reported as prominently as a confirmation. It
would bound how much the sensory/workspace/motor taxonomy can claim, which is worth knowing.

## Materials

20 single-token-answer factual pairs (capital-of and similar), frozen in `prompts.json`
before running, each verified to produce its expected answer as the top-1 clean prediction
in the pilot model. Greedy decoding, no sampling. Patch at the final token position only.

## Cost and order

1. Pilot: gemma-3-270m on this CPU box. **$0.**
2. Confirmation: qwen3-8b, single A100-40 / A6000, inference only. **~$5-15**, dominated by
   the model download, and only if the pilot pipeline passes its sanity gate.

The 397B is explicitly **not** part of this experiment.

## What this does not establish

Whether boundaries are causal is not the same as what each block *does*. A confirmation here
would license "blocks are computational units", not "the middle block is a workspace". The
functional labels need ablation-by-capability and per-block feature analysis, which is a
separate and larger effort.
