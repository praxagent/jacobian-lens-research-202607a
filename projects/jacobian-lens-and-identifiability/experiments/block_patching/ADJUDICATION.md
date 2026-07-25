# Adjudication of the GPT-5.6-sol Pro design audit

Review artifacts: `out/pro_review_2/` (model `gpt-5.6-sol`, `reasoning.mode=pro`, effort medium,
completed, $0.55). Packet was a decision memo, not raw data. A first attempt at effort=high
returned `incomplete` (reasoning exceeded the 6,000-token output cap) and cost ~$0.68 for no
output; recorded in `out/pro_review_1/failure.json`.

Per the integrity playbook, the executing agent must adjudicate every material finding and may
reject one that rests on a factual misunderstanding. Here is that adjudication. **Nearly all of
it is accepted, and our published headline was too strong.**

## Verdict of the review

> The current study does **not** support the broad claim that depth bands fail to partition
> computation. ... As a paper, this is presently an informative failed-probe sequence or
> inconclusive result, not a decisive negative result. **NOT READY TO FREEZE**

## Finding-by-finding

**B01, the estimand does not identify the construct. ACCEPTED.** Cross-layer consumability is
neither necessary nor sufficient for computational partitioning: residual streams are running
sums rather than stage-typed interfaces, and a transplanted state additionally skips some
transformations, repeats others, and lands off-distribution. So a boundary penalty could be
distribution shift, and its absence is compatible with genuine staged computation. We raised
this objection ourselves in the packet; an independent channel confirming it is decisive rather
than reassuring. **Consequence: the headline claim must be narrowed, not merely hedged.**

**B02, a non-significant test is being read as evidence of absence. ACCEPTED.** We never
defined a smallest effect of interest or ran an equivalence test, so we cannot distinguish
"small", "variable", and "poorly measured". Our own numbers make the point: null SDs vary 4x
across models and the boundary effects invert. **Consequence: the correct disposition is
inconclusive, not negative, unless we run an equivalence design.**

**B03, the permutation null does not license population-level inference. ACCEPTED with a
clarification.** The segmentation null is the right test of "is *this* model's fitted boundary
special versus placebo cuts in *this* model", and we used it that way per model. The reviewer
is correct that it does not make models independent replicates or support a claim about
language models generally, which is what our summary sentence implied. **Consequence: model
families, not layer pairs, are the inferential unit.**

**B04, no positive control. ACCEPTED, and this is the finding we most clearly missed.** We
never demonstrated the instrument can detect computational staging when staging is known to
exist. Without that, an insensitive instrument and a real null are indistinguishable, which is
precisely the ambiguity our v1 saturation and 4x-varying null SDs should have flagged. A
synthetic or deliberately bottlenecked model with an imposed stage boundary is the missing
gate. **Consequence: any future round is invalid, not negative, if it cannot recover a known
boundary.**

**I01 (eligibility selection), I02 (single protected endpoint), I03 (KL standardization).
ACCEPTED.** I02 in particular is fair: the campaign moved among global, first-boundary, and
second-boundary summaries, and although each step was pre-registered, the sequence invites
retrospective reinterpretation.

**Nothing rejected.** No finding rested on a factual misunderstanding of our design. The one
place we would add nuance is B03, above.

## What the review endorses keeping

Freezing CKA segmentation before causal outcomes; controlling smooth depth trends (it calls the
distance-and-position recognition "a major strength"); the held-out-model replication; honest
reporting of the sign inversion rather than averaging it away; full disclosure of the v1
saturation failure; per-model rather than universal boundaries.

## The claim we are entitled to

Not this:

> ~~The depth bands describe geometry; we find no evidence they partition computation.~~

But this:

> In the tested models, CKA-derived depth boundaries did not consistently predict additional
> damage from cross-layer residual-state substitution, beyond modeled depth and position
> effects. This is an inconclusive result about computational organization, not evidence
> against it.

## Proposed v3, from the review's own alternative

An **O(L) native-time causal-profile test** instead of the O(L^2) swap grid: perturb each
layer's own residual update at its normal execution time (calibrated attenuation or
matched noise), build a per-layer causal-effect profile over held-out outcomes, and test
whether the profile changes discontinuously at frozen CKA boundaries versus depth-matched
placebo cuts. This avoids the off-distribution transplant problem entirely, because nothing is
moved between depths, and it is cheaper than what we already ran.

Gates it must carry: dose-response monotonicity per model, and recovery of an imposed boundary
in a synthetic bottlenecked control. A predeclared equivalence margin. Model families as units.

## Caveat on the channel

One automated reviewer that saw a memo, not our data or code. It can find design and construct
errors, which it did; it cannot verify our numbers. It is a different lab and lineage from the
executing agent, so its failure modes are genuinely partly perpendicular to ours, but it is one
adversarial channel and not external validation.
