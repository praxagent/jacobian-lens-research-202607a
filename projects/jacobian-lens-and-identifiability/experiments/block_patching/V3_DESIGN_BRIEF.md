# v3 design for audit: do CKA depth boundaries mark discontinuities in native-time causal influence?

This is a **prospective design**, not results. It is the successor to a campaign your previous
review judged NOT READY TO FREEZE. We accepted all four blocking findings and rebuilt around
them. Audit this design before we run it. Attack construct validity first, then inference.

## What the prior review established (accepted, not re-litigated)

1. **B01.** Cross-layer state substitution does not identify computational partitioning:
   residual streams are running sums, transplants skip and repeat transformations and land
   off-distribution. Our old estimand is abandoned.
2. **B02.** We had no smallest effect of interest, so non-significance was not absence.
3. **B03.** A within-model segmentation permutation does not license cross-model claims.
4. **B04.** We had no positive control, so an insensitive instrument and a real null were
   indistinguishable.

## Construct and claim boundary

Claim we intend to be able to make or refute:

> At CKA-derived depth boundaries, the **causal influence profile** of adjacent layers changes
> discontinuously, beyond the smooth depth trend and beyond depth-matched placebo cuts,
> consistently across independently trained model families.

We are explicitly **not** claiming anything about sensory/workspace/motor semantics.

## Intervention: native-time, in-place, dose-parameterised

Transformer layer `l` contributes an additive update: `h_{l+1} = h_l + Delta_l`. We perturb
that layer's **own** update at its **normal execution time**, by attenuation:

    h_{l+1} = h_l + (1 - alpha) * Delta_l,    alpha in {0.25, 0.5, 1.0}

Nothing is moved between depths, nothing is transplanted, and at `alpha = 0` the model is
exactly unmodified. `alpha = 1` ablates the layer's contribution entirely. We apply this at all
token positions, and we do **not** touch attention/MLP internals or normalisation, only the
residual addition.

Rationale against B01: the network is only ever asked to consume states it produced itself
under a smaller update, so the off-distribution objection is bounded by `alpha` rather than by
depth distance.

## Outcomes (predeclared)

Per layer `l` and dose `alpha`, over a held-out prompt set:

- `KL_l = KL(p_perturbed || p_clean)` on the next-token distribution. Direction frozen as
  written.
- `Acc_l` = exact-match accuracy on a short factual/task set, so the claim touches behaviour
  and not only distributional drift.

The **causal influence profile** is the vector `e_l = (KL_l, 1 - Acc_l)` at the primary dose
`alpha = 0.5`, with the other doses reserved for the dose-response gate.

## Primary statistic

1. Fit a smooth depth trend to `e_l` across `l` **without any boundary labels** (monotone
   spline or local regression; frozen before outcomes).
2. Define the discontinuity at cut position `p` as the residual jump
   `d_p = |resid_{p+1} - resid_p|`.
3. For each model: `Z_model = (mean d_p at its two frozen CKA boundaries - mean d_p at
   depth-matched placebo cuts) / sd(d_p at placebo cuts)`.
   Placebo cuts are all admissible 3-segmentations with the same block-size multiset,
   used **only for within-model standardisation**, per B03.
4. **Confirmatory inference is across model families**, one `Z_model` per family, testing the
   mean `Z` against zero and against an equivalence margin.

## Equivalence margin

`delta = 0.5` in units of placebo-cut SD: a boundary that is less than half a placebo standard
deviation more discontinuous than an arbitrary cut is not a meaningful partition. We are
unsure this is the right justification and want it challenged.

Interpretation, frozen:

| CI for mean Z | verdict |
|---|---|
| entirely inside `[-0.5, +0.5]` | evidence against a meaningful discontinuity |
| excludes 0, outside equivalence region | boundaries mark real discontinuities |
| overlaps both | inconclusive / underpowered |
| positive-control gate fails | invalid |

## Positive controls (the B04 fix)

**G1, dose-response.** In every eligible model, mean `KL_l` must increase monotonically across
`alpha in {0.25, 0.5, 1.0}`. Failure means the intervention is not biting; that model is
invalid, not null.

**G2, imposed-boundary recovery.** We construct a control model with a **known** stage boundary
by inserting a rank-`r` linear projection (`r << d_model`) into the residual stream of a small
pretrained model at a chosen depth `k`, creating a genuine information bottleneck at a location
we chose. The frozen pipeline must recover a discontinuity at `k`. If it cannot detect a
boundary we built by hand, the instrument cannot support a null anywhere, and the whole round
is invalid.

We are unsure whether an inserted low-rank projection is a *fair* positive control, or whether
it makes the task artificially easy in a way natural boundaries would not be. We want this
attacked specifically.

## Eligibility and units (B03, I01)

- Predeclared before any causal outcome: any model with a published Jacobian lens, a fitted
  3-segmentation, and at least 3 layers per block.
- **No balance requirement**, unlike our previous round: the adjacent-cut statistic does not
  need balanced blocks, which removes the >=27B selection effect that biased the old campaign.
- Checkpoints from one lineage (base and instruct, size variants) are **clustered as one
  family**; the inferential N is families, not checkpoints.
- Target N = 8 families. We have not done a formal power calculation and want to be told what
  it should be.

## Cost

`O(L)` interventions per model rather than `O(L^2)`: for a 64-layer model, 3 doses times 64
layers times one batched prompt set is a few hundred forward passes, minutes of GPU. Small
models run free on CPU. Estimated total for 8 families: under $30, dominated by model
downloads.

## What we want attacked, in priority order

1. **Does attenuation of a layer's own update identify "computational stage boundary" any
   better than our abandoned transplant did?** Our worry: attenuation measures *how much a
   layer matters*, and its derivative across depth may be smooth for reasons unrelated to
   stage structure, so a boundary discontinuity might be undetectable in principle even if
   stages exist. Is there a sharper intervention?
2. **Is the inserted-bottleneck positive control fair, or does it validate the instrument only
   against an artefact it cannot miss?** If unfair, what is a better known-positive?
3. **Is `delta = 0.5` placebo SD a defensible equivalence margin**, and is a standardised
   within-model `Z` the right thing to pool across families of very different depths?
4. **What N of families do we actually need**, and is 8 remotely adequate for an equivalence
   conclusion?
5. **Is there a fundamentally better and cheaper design** we still have not considered, given
   that the object of interest is defined by a *geometric* statistic (CKA of Jacobian readout
   geometries) and we are trying to give it *causal* meaning?

Give the single strongest reason this design will also fail to answer the question.
