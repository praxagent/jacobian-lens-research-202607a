# Results: what the invariant readout subspace encodes

Design frozen in [`PREREG.md`](PREREG.md) (commit `f563e28`) before any run, including the
adjusted-R² thresholds and the pre-declared predictions.

**Gate passed:** the pipeline reproduces our published correlation exactly, `-0.5908` against an
expected `-0.591`.

## Verdict: PARTLY SETTLED

Top direction, adjusted `R^2` = **0.572** from the seven frozen features. That falls in the
0.50 to 0.80 band we declared "partly settled" before running, and we said in advance that this
awkward middle was the likely outcome, so we report it as such rather than reaching for a
better adjective.

| direction | energy share | adj `R^2` (all 7 features) | model prior alone | unigram alone |
|---|---|---|---|---|
| 0 | 48.4% | **0.572** | 0.455 | 0.349 |
| 1 | 14.8% | 0.559 | 0.443 | 0.344 |
| 2 | 6.7% | 0.593 | 0.479 | 0.352 |
| 3 | 4.2% | **0.330** | **0.034** | 0.012 |

## The pre-declared prediction was right, and it matters

**The model's own output prior beats the corpus unigram count**, alone and by a clear margin:
`R^2` 0.455 against 0.349 on the top direction, and the same ordering on directions 1 and 2. In
the full regression `model_prior` also carries the largest standardised weight (`-0.301`) with
`log_unigram` second (`-0.170`).

That is the more interesting statement of the two. The invariant readout direction is better
described as **the model's own default output tendency** than as a property of the text it was
trained or fitted on. A frequency axis was the right neighbourhood; "the model's prior" is the
better address.

## An alternative we can now rule out

`is_unused` carries essentially **no** weight (`+0.021`, from 423 special or never-trained
slots). We were concerned that our published frequency correlation might partly be an
untrained-embedding artifact, since Gemma ships a large block of `<unused>` tokens sitting at
one pole of the axis. It is not. The frequency and prior structure is carried by real tokens.

Token length contributes modestly (`-0.114`) and embedding norm retains independent signal
beyond the prior (`+0.203`), so the axis is not purely a frequency object even within the part
we can explain.

## What is still not explained, and one clean surprise

**About 43% of the top direction's per-token variance is not accounted for** by any of the seven
features. We do not know what that is.

**Direction 3 is a different kind of object.** It carries 4.2% of the readout energy, and the
model prior explains almost none of it (`R^2` 0.034, against 0.455 for the top direction). Its
full-feature `R^2` of 0.330 is below our own "unexplained" threshold. Whatever the fourth
component of the invariant subspace encodes, it is **not** a prior or frequency axis, and none
of our named features describe it. Directions 1 and 2, by contrast, look like weaker copies of
direction 0.

So the subspace is not homogeneous: three prior-like components plus at least one that is
something else.

## Honest limits

One model, gemma-2-9b, which is where the flat readout was found in the first place. A high
`R^2` would have shown the direction is *describable* by these features, not that the model
*computes* it that way, and we are well short of a high `R^2` regardless. The prior was
estimated over 23,383 positions of WikiText, so it is itself a corpus-conditional quantity,
which given our corpus-dependence result is worth stating rather than glossing.

## Cost

One RTX 3090 for the forward passes, about **$0.10**; everything else local and free.
