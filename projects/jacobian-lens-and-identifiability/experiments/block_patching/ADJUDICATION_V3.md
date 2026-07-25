# Adjudication of the v3 design audit (high effort)

Artifacts: `out/pro_review_v3/` (`gpt-5.6-sol`, `reasoning.mode=pro`, effort **high**,
completed, 101,210 tokens, $1.65). Prospective: no v3 data exists.

Verdict: **NOT READY TO FREEZE.** We accept it. v3 is not built, and the family campaign is
cancelled in favour of a different and much cheaper experiment the review proposed.

## The finding we missed, and it is the important one

**B03: the design cannot separate CKA information from its own mathematical inputs.**

For small attenuation, output KL is approximately a quadratic function of the residual update
and the **downstream logit Jacobian**. Our CKA boundaries are themselves derived from Jacobian
readout geometry. So a boundary-aligned jump in attenuation sensitivity could arise purely
because both quantities are functions of the same Jacobian, with no stage structure anywhere.

This is circularity, and we did not see it. It is worse than a power problem: it means a
**positive** v3 result would have been uninterpretable too. We would have run the experiment,
possibly found the effect, and published something confounded by construction.

The fix the review gives is right: the confirmatory question must be whether CKA adds
predictive value **beyond** cheap comparators (equal-depth cuts, update-norm-change cuts,
first-order Taylor-predicted-KL cuts, architecture-transition cuts), on prompts disjoint from
those used to fit the lens.

## The rest

**B01, attenuation still does not identify a stage boundary. ACCEPTED.** We flagged this
ourselves as our top worry and the review confirms it: attenuation measures how consequential a
block's update is, not whether computation is partitioned across a cut. A real stage boundary
can give a smooth profile; update magnitude, normalisation, or architecture can give a jump
with no stage. So both outcomes were overinterpretable. Consequence: any surviving claim must
be phrased as *block-update sensitivity*, never "computational stage".

**B02, the primary statistic is not executable as written. ACCEPTED.** A two-component outcome
with no scaling rule, "monotone spline or local regression" left undecided, and standardisation
by a possibly-unstable placebo SD. Different reasonable implementations could reverse the
result, which is a researcher-degrees-of-freedom hole.

**B04, both positive-control gates are misaligned. ACCEPTED.** Monotone dose response is not
required for a valid intervention (nonlinear trajectories exist, especially at `alpha = 1`),
and our inserted rank-`r` bottleneck tests sensitivity to gross off-distribution damage rather
than to subtle natural boundaries. We had asked specifically whether that control was fair; it
is not. The replacement is three separate gates: an **execution** gate (`alpha = 0` reproduces
clean logits, update scaled by exactly `1-alpha`), a **statistic-sensitivity** gate (blinded
synthetic profile spike-ins at the margin recovered at a prespecified rate), and a **construct**
control only if stage language is retained.

**B05, the powered design is infeasible at our scale. ACCEPTED, and decisive.** With
between-family SD ~1 and margin 0.5, equivalence needs roughly **35-44 families**, before
inflation for heterogeneity and attrition. We planned 8. Eight suffices only if between-family
SD is <= 0.48, which we have no reason to assume. So the cross-family *population* claim is out
of reach; the honest options are a finite named panel with no population claim, or a much
larger campaign we are not going to fund.

**I01-I05 accepted** (family unit construction, "on-distribution" overstatement, behavioural
scoring gaps, multiplicity rules, and a compute estimate that ignored access/storage/wall-time
for 35+ families).

## Nothing rejected

As with the previous audit, no finding rested on a misunderstanding of our design. Two
independent-lineage reviews have now told us the same underlying thing in different words: we
keep building instruments whose readings cannot be tied to the construct we name.

## Decision: cancel the family campaign, run the review's preliminary experiment instead

The review's closing suggestion is better than our design and it directly targets what we
actually care about:

> An **aligned-versus-random, equal-norm residual perturbation** study at CKA and matched cuts
> in two or three small models, testing whether the geometric directions predict causal
> transmission while separating geometry from native update magnitude. It should **replace**,
> not be added to, the family campaign.

Why this is the right pivot:

- It attacks the circularity in B03 head-on. Comparing **aligned** versus **random** directions
  of the **same norm** at the **same cut** cancels update magnitude and generic sensitivity,
  because both arms share them. What differs is only whether the perturbation lies in the
  Jacobian-geometry directions.
- It tests the thing our whole atlas is about, namely whether J-space geometry has causal
  meaning, rather than the much shakier "are the bands computational stages".
- Two or three small models, free on CPU or a few dollars of GPU, against a 35-44 family
  campaign we cannot afford.
- It is a preliminary that earns the right to a bigger claim, instead of a large campaign
  built on an unidentified construct.

## Consequences for what is already published

The Atlas note says we "intend to run" the O(L) native-time successor. That must be corrected:
v3 is not being run as specified. The published block-patching result stays as it is, since it
was already narrowed to the inconclusive statement after the first audit.

## Cost of the consult, and whether it was worth it

$0.68 (a failed high-effort run that hit the old output cap), $0.55 (medium, decisive), $1.65
(high, this one) = **$2.88 total**. It stopped us building a second instrument that could not
answer the question, caught a circularity that would have made a positive result meaningless,
and produced a cheaper experiment that is actually identifiable. The GPU campaign it replaced
would have cost far more and concluded nothing.
