# Is this negative result real, or a careful null of the wrong thing?

Answer in **under 700 words**. Spend your effort on reasoning, not exposition.

## Setup

A **Jacobian lens** gives per layer `l` the matrix `J_l = E[d h_final / d h_l]`: the model's own
averaged linear map from layer `l`'s residual stream to its final residual stream. Project
through the unembedding to get a per-layer "readout geometry"; take layer-by-layer CKA of those
geometries and most models show contiguous **depth bands** (early / middle / late), which
Anthropic labels sensory / workspace / motor. We replicated this geometry across 36 models. It
is robust and not in question.

We want to claim: **the bands describe representational geometry; we find no evidence they
partition computation.**

## The probes we ran

Blocks are each model's own fitted 3-segmentation of its CKA map. Significance always comes
from a permutation null over random 3-segmentations with the same block-size multiset.

**v1, cross-prompt patching.** Source/target prompts differ in one attribute. Insert the
source's residual stream from layer `i` at layer `j` of the target; measure movement toward the
source answer. Transfer decays with `|i-j|` regardless of blocks, so we regressed on
per-distance dummies plus a crossing indicator. 3 models: two nulls, one significant
wrong-signed positive that we later showed was absolute-layer-position confounded with crossing
(adding position dummies collapsed +0.148 to -0.046). The measure also saturated (0.65-0.99 of
full transfer).

**v2, same-prompt swap damage.** Removes semantics entirely: one prompt, capture its residual
stream at layer `i`, re-run the same prompt substituting that state at layer `j`, measure
`KL(patched || clean)` on the next token. Zero by construction at `i = j`, no ceiling. Controls:
distance **and** mean-position dummies. On the one affordable model with balanced blocks
(25/14/24): `beta = +0.220`, p = 0.18.

**v2.1, pre-registered localized test.** Exploratory decomposition of v2 put the effect at the
first boundary (+0.437, p = 0.018) and nothing at the second (-0.018). We pre-registered that
single directional hypothesis and tested a new model, holding out the generator. Result:
`beta_b1 = -0.818` (one-sided p 0.87), opposite sign, with that model's own effect at the
*other* boundary (+0.740). A third, poorly-balanced model gave -1.91. Per-model null SDs vary
4x. Only models >= 27B have balanced fitted blocks at all.

## The two questions

**1. Construct validity.** Is "can layer `j` consume layer `i`'s representation" the right
estimand for "blocks are computational units"?

The specific worry we cannot resolve ourselves: a residual stream is **additive**, so every
layer reads a running sum rather than a stage-typed object. Cross-layer state substitution may
therefore be guaranteed to produce smooth distance-dependent damage with no boundary structure
*even if* blocks are genuine computational stages, because the substituted state is still a
valid partial sum. If that is right, our null is uninformative rather than negative, and we
should not publish it as evidence about computation.

Is that objection correct? If it is, what estimand would actually test the claim?

**2. Publishable negative, or underpowered?** With n = 2 balanced models, inverting boundary
patterns, and null SDs varying 4x, are we entitled to "no evidence that bands partition
computation", or only to "our probes cannot detect it"? These are different papers.

Give the single strongest reason our conclusion is wrong, and name one cheaper better test if
one exists.
