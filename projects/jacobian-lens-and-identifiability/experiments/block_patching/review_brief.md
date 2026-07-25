# Design audit: a negative result about "depth blocks" in language models

We are about to publish a negative result and want it attacked before we do. Please attack
**construct validity first**: is our estimand the right proxy for the claim, or have we
carefully measured the wrong thing and called it a null?

## Background (what is established, not at issue)

A **Jacobian lens** gives, per layer `l`, a matrix `J_l = E[d h_final / d h_l]`, the model's
own averaged linear map from layer `l`'s residual stream to its final residual stream. Project
it through the unembedding and you get a per-layer "readout geometry". Taking layer-by-layer
centered kernel alignment (CKA) of those geometries produces a map that, in most models, shows
**contiguous depth bands**: early layers resemble each other, middle layers resemble each
other, late layers resemble each other. Anthropic labels these sensory / workspace / motor.

We replicated this geometry across 36 models. It is robust and not in question here.

## The claim we want to make

> The depth bands are a description of representational **geometry**. We find no evidence they
> partition **computation**.

## What we did

Three rounds. In all of them, "blocks" are each model's own fitted 3-segmentation of its CKA
map, and significance comes from a permutation null of random 3-segmentations with the same
block-size multiset.

**v1. Cross-prompt activation patching.** Source and target prompts differ in one attribute
(`The capital of France is` vs `... Japan is`). Capture the source's full residual stream at
layer `i`, insert at layer `j` of the target, measure how far the output moves toward the
source answer. Prediction: transfer is worse across a block boundary. Critically, transfer
decays with `|i-j|` regardless of blocks, so we regressed on per-distance dummies plus a
crossing indicator rather than comparing raw means.

Result: 3 models, two nulls, one significant *wrong-signed* positive. Post hoc we suspected
absolute layer position was confounded with crossing; adding position dummies collapsed that
model's estimate from +0.148 to -0.046, confirming it.

Also, the measure saturated: patched outputs sat at 0.65-0.99 of full transfer.

**v2. Same-prompt cross-layer swap damage.** To remove the semantic dimension entirely: take
one prompt, capture its residual stream at layer `i`, re-run the same prompt substituting that
state at layer `j`, measure `KL(patched || clean)` on the next-token distribution. Zero by
construction when `i = j`, no ceiling. Controls: distance dummies **and** mean-position dummies.
Target chosen for block balance (most models' fitted blocks are one large block plus slivers;
we used the one affordable model with balanced blocks, 25/14/24).

Result: `beta = +0.220`, p = 0.18. Null.

**v2.1. Pre-registered localized test.** Exploratory decomposition of v2 showed its diffuse
estimate was concentrated at the *first* boundary (+0.437, p = 0.018) with nothing at the
second (-0.018). We pre-registered that as a single directional hypothesis and tested it on a
new model, holding the generating model out.

Result: the new model gave `beta_b1 = -0.818` (p one-sided 0.87), the **opposite sign**, and
its own effect sat at the *other* boundary (+0.740). A third model (small, poorly balanced
blocks) gave -1.91. The pattern inverts across models; per-model null SDs vary 4x.

## Summary table

| round | measure | models | headline |
|---|---|---|---|
| v1 | cross-prompt transfer (saturating) | 3 | 2 nulls; 1 wrong-signed positive, shown to be a position artifact |
| v2 | same-prompt swap damage, distance+position controlled | 1 balanced | +0.220, p = 0.18 |
| v2.1 | localized to boundary 1, pre-registered | 1 balanced | -0.818, does not replicate; inverts |

## Specific things we want challenged

1. **Construct validity.** Is "can layer `j` consume layer `i`'s representation" the right
   operationalization of "blocks are computational units"? A plausible alternative: blocks
   could be real computational stages while cross-layer state substitution is simply the wrong
   probe, because a residual stream is additive and every layer reads a running sum rather
   than a stage-typed object. If so our null is uninformative rather than negative. What would
   a better estimand be?
2. **Is the null hypothesis test the right one?** Our null randomizes the *segmentation* within
   a model. With one true segmentation per model, is that null too conservative, too liberal,
   or simply answering a different question than we think?
3. **Power.** Null SDs are large and vary 4x across models. Are we entitled to any negative
   claim at n = 2 balanced models, or should the honest statement be "underpowered, no claim"?
4. **The saturation and balance problems.** Only models >= 27B have balanced fitted blocks.
   Does that alone make small-model results uninterpretable, and does it bias the whole
   campaign toward nulls?
5. **The cheapest serious alternative.** Is there a substantially better and cheaper test of
   the same claim we have not considered? Layer shuffling within vs across blocks, boundary
   effects on ablation damage, probing for stage-typed features, something else?
6. **Should this be published as a negative result at all**, or is it a null that mainly
   reflects instrument limitations and should be reported as inconclusive?

Be direct about the strongest reason our conclusion could be wrong.
