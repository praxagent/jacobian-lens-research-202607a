# Pre-registration: two tests of what the J-space depth bands are

**FROZEN before any run.** Written 2026-07-26. No data for either test existed at commit.

## Why these two, and why now

Three pre-registered designs across six models failed to show that CKA depth bands partition
computation, and two adversarial reviews diagnosed the failure as an identification problem
rather than a null effect (`../block_patching/results.md`, `ADJUDICATION_V3.md`). The published
position is **inconclusive, not negative**.

Two live explanations survive that campaign, and each admits a cheap identified test that the
campaign could not run. **Test B** asks whether the bands are a property of the model's
representations at all, or only of the first-order readout we view them through. **Test C** asks
whether "the boundary" is even a stable target, using a fact we did not have when the campaign
ran: fitted boundaries move up to a combined fifteen layers when the lens is fitted on code
instead of prose (`../corpus_dependence/results.md`).

Both are correlational. Neither can establish that the bands *are* computational. Both can
substantially change what we believe they are, which is why they are worth running.

---

# Test B: do lens boundaries coincide with activation boundaries?

## The hypothesis being tested

The lens is the first-order **output** map. Its depth bands may therefore describe how the
model's output-relevant linearization rotates, which is downstream of computation rather than a
map of it. We already have one model where the lens and the representations disagree sharply:
gemma-2-9b's activations are structured (activation band separation +0.110) while its lens is
nearly flat (+0.005). If that dissociation is general, the bands are a readout phenomenon.

## Design

For each frozen model below, compute the layer-by-layer **activation** CKA over WikiText and fit
the same three-segmentation used everywhere in this campaign (`atlas_stage_a.fitted_seg`). Compare
those boundaries to the boundaries of that model's **shared-vocabulary lens** map, which is
already cached in `../jspace_atlas/atlas_out/<slug>.npz` and is not recomputed here.

Primary statistic, per model:

    agreement = |b1_lens - b1_act| + |b2_lens - b2_act|      (in layers, lower = better)

normalised by depth as `agreement / L` so models of different depths are comparable.

## Frozen model set (8 models)

Chosen before running for family spread, size spread, and coverage of the flat-to-structured
range. All are small enough to run activation capture on one mid-size GPU.

| slug | HF id | lens mid_sep | why in the set |
|---|---|---|---|
| `gpt2-small` | `openai-community/gpt2` | 0.0149 | shallow, flat, oldest lineage |
| `pythia-70m-deduped` | `EleutherAI/pythia-70m-deduped` | 0.0155 | tiny, independent lineage |
| `gemma-3-270m` | `google/gemma-3-270m` | 0.1336 | tiny but strongly structured |
| `gemma-2-2b` | `google/gemma-2-2b` | 0.0074 | the flat-lens family, small |
| `gemma-2-9b` | `google/gemma-2-9b` | 0.0047 | the flat-lens case we already have activation CKA for; acts as a **replication anchor** |
| `qwen3.5-0.8b` | `Qwen/Qwen3.5-0.8B` | 0.1149 | structured, small |
| `qwen3-4b` | `Qwen/Qwen3-4B` | 0.0383 | structured control from the readout-specificity test |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B` | 0.1059 | third family, structured |

No model may be added or dropped after seeing any result. If a model fails to run (OOM, gated
weights, architecture unsupported by the capture hook), it is reported as **failed**, not
silently replaced.

## Null

Random three-segmentations of the same depth with the same block-size multiset as the lens
segmentation, 1,000 draws, giving the distribution of `agreement` under no relationship. This is
the same null family used throughout the block campaign.

## Decision rules (frozen)

| outcome | verdict |
|---|---|
| mean normalised agreement beats the null at `p < 0.05` **and** at least 6 of 8 models individually beat their own null median | **BANDS TRACK REPRESENTATIONS**: theory B is weakened, the bands are not merely a readout artifact |
| agreement indistinguishable from null (`p >= 0.05`) | **BANDS ARE A READOUT PROPERTY**: lens boundaries do not mark where the representations reorganise, and every functional reading of the bands in our note must be re-scoped |
| anything between | **MIXED**: report per-model, claim nothing general |

## Predictions

We predict **BANDS ARE A READOUT PROPERTY**, i.e. the null result. We say so in advance because
it is the outcome that costs us the most: it would force the atlas to be described as a map of
the output linearization rather than of the model's depth organisation. The gemma-2-9b
dissociation is our reason for expecting it, and gemma-2-9b is in the set as an anchor precisely
so that a failure to reproduce that known dissociation invalidates the run.

## Gates

- **Replication gate.** gemma-2-9b must reproduce its previously measured activation band
  separation (+0.110) to within 0.03, or the pipeline is VOID rather than informative.
- **Sanity gate.** Activation CKA diagonal is exactly 1.0 and the off-diagonal is not degenerate
  (median < 0.999), or that model is reported as failed.

## What this cannot establish

Agreement between two descriptive maps is not causation. A positive result says the lens bands
coincide with where representations reorganise; it does not say either one partitions
computation. A negative result does not prove the bands are meaningless, only that they are not
about the representations.

## Cost

Forward passes only, eight models up to 8B, one mid-size GPU. Estimated **under $3**.

---

# Test C: does the fitting corpus change which boundary predicts damage?

## The hypothesis being tested

Fitted boundaries move with the fitting corpus. If a boundary's location depends on the
estimation distribution, then every causal test we ran had large measurement error in its
independent variable, which attenuates any real effect toward the null. That is a concrete
alternative explanation for the whole block campaign's failure, and it did not exist as a
testable idea until the corpus result landed.

## Design: crossed, within-model, boundaries-only contrast

For each model we hold the **damage measurement completely fixed** and vary only the
segmentation label applied to it. This is the design property the earlier rounds lacked.

Measure `D(i, j)`, the KL damage from swapping layer `i`'s captured activation into layer `j`,
using the existing `../block_patching/swap_v2.py` runner unchanged, on **two prompt sets**:
prose and code. Then fit the v2.1 statistic

    D(i,j) ~ dummies(|i-j|) + dummies(mean position) + beta * crosses(i,j)

four times per model, crossing **which corpus the boundaries came from** with **which corpus the
damage was measured on**. The lens boundaries are already computed and recorded in
`../corpus_dependence/results.json`; no lens is refitted.

|  | damage on prose | damage on code |
|---|---|---|
| **boundaries from the WikiText lens** | matched | mismatched |
| **boundaries from the code lens** | mismatched | matched |

## Two questions, both frozen

- **C1 (does any segmentation predict damage?).** Is `beta` for the matched cells above the
  random-3-segmentation null, pooled across models, at `p < 0.05`?
- **C2 (does the fitting corpus matter?).** Is `beta(matched) - beta(mismatched)` positive,
  pooled across models, at `p < 0.05` against the same null?

## Frozen model set (3 models)

The three models for which we already hold both a WikiText lens and a code lens.

| slug | WikiText boundaries | code boundaries | role |
|---|---|---|---|
| `gpt2-small` | 6 / 8 | 6 / 8 | **mechanical control**: the boundaries are identical, so `beta(matched) - beta(mismatched)` must come out **exactly 0**. A non-zero value here means the pipeline is broken, and the run is VOID |
| `gemma-3-270m` | 3 / 15 | 12 / 14 | contrast model |
| `qwen3.5-0.8b` | 15 / 19 | 4 / 15 | contrast model |

## Decision rules (frozen)

| C1 | C2 | verdict |
|---|---|---|
| significant | significant | **BOUNDARIES PREDICT DAMAGE, AND THE CORPUS MATTERS**: the block campaign's nulls are plausibly attenuation from an unstable independent variable, and the strand is worth reopening with matched corpora |
| significant | not | **BOUNDARIES PREDICT DAMAGE, CORPUS-INVARIANTLY**: a positive we did not get in three earlier rounds; would need replication before any claim |
| not | significant | **INCOHERENT**: report as such and claim nothing; a corpus effect with no main effect is not interpretable |
| not | not | **NO PURCHASE**: consistent with the published inconclusive position, and we stop for good |

## Predictions

We predict **NO PURCHASE**, with C2 the more likely of the two to move. Stated in advance because
we have been wrong in the optimistic direction on this question three times.

## Gates

- `D(i, i) = 0` exactly, per model, per prompt set.
- Median `D` over `|i-j| >= 2` inside `[0.05, 5.0]` nats, the calibration band frozen in
  `../block_patching/PREREG_V2.md`, or that cell is reported as out of dynamic range.
- The gpt2-small mechanical control must return exactly 0 for C2, or the whole run is VOID.

## Honest power statement, written before running

This is **underpowered by construction and we are not going to pretend otherwise**. Two
contrast models, both small, both with lopsided fitted blocks (3/12/2 and the qwen equivalent),
which the v1 post-mortem already identified as a weak setting for detecting a boundary. The
random-segmentation null's spread is driven by segmentation variability, not pair count, so the
3,000-plus pairs per cell buy nothing. A null here is **weak evidence**, and we commit now to
reporting it as weak rather than as confirmation of the earlier nulls.

The reason to run it anyway is C2, which is a **within-model paired contrast** where model,
layer, position, distance and prompts are literally identical across arms. That is far better
identified than anything in the block campaign, and if the effect is large it will show up even
here. If C2 lands significant in this weak setting, that is a strong signal worth spending real
money on; if it does not, we have not learned much and will say so.

## Cost

`swap_v2.py` on three sub-1B models runs on this box's CPU, as the v1 round did. Estimated
**$0**, with a small GPU as fallback if CPU wall-clock is impractical.
