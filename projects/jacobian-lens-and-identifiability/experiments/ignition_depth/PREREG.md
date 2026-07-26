# Pre-registration: does behavioural ignition depth track the fitted mid/late boundary?

**FROZEN before any run.** Written 2026-07-26. No ignition data existed at commit.

## Why this is a new strand, not a repair

Five designs have failed to show that CKA depth bands partition computation, and we have stopped
trying to show that. This asks a **different** question, about a **different** construct.

Rather than asking whether the bands are boundaries *of computation*, it asks whether the late
boundary coincides with the depth at which the model's answer becomes **readable**. That is a
behavioural landmark measured on the model's own output head, not a property of the lens, so it
does not inherit the circularity that killed the attenuation design: the ignition depth is
computed from logits, and the boundary from Jacobian readout geometry, and the two share no
estimator.

If they coincide, the late boundary has an interpretation we can state plainly: *this is where
the model has committed*. If they do not, the bands remain a description of readout geometry with
no behavioural landmark attached, which is where our note currently leaves them.

## Estimand

For a prompt whose correct next token is `t*`, define the **ignition depth** as the shallowest
layer `l` at which the *output head applied to that layer's residual stream* ranks `t*` first,
and it remains first at every deeper layer. This is a logit-lens readout, not a J-lens readout,
which is deliberate: it keeps the behavioural measure independent of the Jacobian.

    ignition(prompt) = min { l : rank_l(t*) = 1 and rank_m(t*) = 1 for all m > l }

Per model, the ignition statistic is the **median over prompts** that ignite at all, expressed as
relative depth. Prompts where `t*` is never rank 1 at any layer are excluded and their count
reported.

## Design

Ten frozen factual-completion prompts with unambiguous single-token answers, written into an
artifact before any run so the set cannot drift. For each model we compute ignition depth, and
compare its median relative depth to the fitted **late** boundary `b2 / L` from the cached
shared-vocabulary lens map.

Primary statistic per model: `gap = | median_ignition_reldepth - b2/L |`.

Null: the same gap computed against the *other* two boundaries of a random 3-segmentation with
the same block-size multiset, 1,000 draws.

## Frozen model set

The same six models the repaired Test B uses, so one pod session serves both and no model is
selected on the basis of its ignition result: `gpt2-small`, `gemma-3-270m`, `qwen3.5-0.8b`,
`qwen3-4b`, `gemma-2-9b`, `qwen3-1.7b`.

## Decision rules (frozen)

| outcome | verdict |
|---|---|
| pooled `gap` beats the null at `p < 0.05` **and** a strict majority of usable models individually beat their own null median | **IGNITION TRACKS THE LATE BOUNDARY** |
| pooled `p >= 0.05` | **NO ALIGNMENT**: the late boundary has no behavioural landmark in this measure |
| otherwise | **MIXED** |

A model is usable if at least 5 of its 10 prompts ignite. Fewer, and it is reported as failed.

## Prediction

We predict **NO ALIGNMENT**. Five prior designs have found nothing behavioural attached to these
boundaries, and Test B just found that they do not even track where representations reorganise.
Predicting otherwise here would be optimism rather than inference. We state it so a positive
would be a genuine surprise on the record, and so a null cannot be presented afterwards as
"expected all along" without this line to point at.

## Gates

- The output head must reproduce the model's actual next-token prediction at the final layer for
  at least 8 of 10 prompts, or that model's readout path is wrong and it is reported as failed.
- Ignition must be monotone by construction; we assert the "stays rank 1" condition rather than
  taking the first crossing.

## What this cannot establish

Coincidence of a geometric boundary with a behavioural landmark is an association across a
handful of models, not evidence that the boundary causes, marks, or partitions anything. Ten
prompts is a small behavioural sample, chosen to keep this cheap; a null here is weak evidence
and we will label it so.

## Cost

Forward passes only, six models, shared with the Test B pod session. Marginal cost **under $1**.
