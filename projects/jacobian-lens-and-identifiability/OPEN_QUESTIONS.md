# Open questions after the J-space campaign, and what we propose to do about each

Companion to the research note under review. We have been working this campaign for a long
stretch and want an outside check on **whether to stop**, before we spend more. Please be
willing to say "publish and stop" if that is the right answer; we are not looking for
permission to continue.

Context on the campaign's track record, because it should inform how much you trust our
judgement here: we published and then **withdrew** four claims of our own during this work (a
Gemma "one-geometry network" reading, a "concentration is a gemma-2 family trait" claim, a
`C = 0.9996` saturation artifact, and a "non-first-order component" reading of dose-dependence).
All four were caught by checks frozen before the numbers existed. Two full experimental designs
were killed by prospective adversarial review before producing data.

## 1. What does "the averaged lens captures ~5% of the achievable first-order effect" mean?

**Status.** Measured in 2 of 3 models (0.053, 0.045); the third is void because its gradient
arm saturates below our smallest perturbation.

**Why it is unresolved.** We have no reference point. 5% could be impressive (an average over a
whole corpus, with no knowledge of the input, recovering a twentieth of the input-optimal
direction) or damning (95% of the achievable effect is invisible to the lens). Nothing in the
J-lens literature we can find quantifies this.

**Proposal.** Sweep `C` against things it should depend on if it is meaningful: the corpus the
lens was fit on, model size, and depth within a model. If `C` sits near 5% everywhere, that is a
stable quantitative fact about what corpus-averaging costs. If it varies systematically with
depth or scale, that is more interesting still. Cheap: existing lenses, existing code, order
$1 of GPU.

## 2. Why does gemma-3-270m saturate so much earlier than the others?

**Status.** Its gradient arm is out of the first-order regime even at our smallest rung, in both
bf16 and float32, so `C` is void for it in every run.

**Why it might not be a nuisance.** In a separate analysis this same model had the most
concentrated readout in the whole 36-model zoo (top-16 readout energy share 0.202). "Extremely
concentrated readout" and "extremely perturbation-sensitive" may be the same underlying fact
seen twice, which would tie two of our findings together.

**Proposal.** Extend the dose ladder downward for this model alone until the gate passes, then
test whether readout concentration predicts saturation onset across all models we have. Free to
about $1.

## 3. Does the lens advantage change at CKA depth boundaries?

**Status.** Pre-registered as exploratory (P3), never analysed. The data is already on disk.

**Why it matters.** It is the only bridge between the geometry result (lens directions are
causally real) and the block question (are the bands computational). It is also the one place
we could still, cheaply, learn something about blocks.

**Proposal.** Run the pre-registered exploratory analysis on existing receipts. Zero additional
compute. **Reported as exploratory whatever it shows**, given the campaign's history of
boundary-effect results that did not replicate.

## 4. Every negative we have about blocks came from badly-segmented models

**Status.** Our block-causality nulls came from models whose fitted 3-segmentation is one large
block plus slivers (balance 0.15 to 0.56). The model with by far the cleanest structure, a 397B
MoE (balance 0.56, and the strongest band separation we have measured at +0.472), has never been
tested causally. We deliberately did **not** buy it, twice, on cost-discipline grounds.

**Why it is a real caveat rather than an excuse.** If depth blocks are computational anywhere,
they should be most detectable where the geometry is cleanest. Our claim is "inconclusive", and
inconclusive results from poorly-conditioned models are weak evidence.

**Proposal, and our uncertainty.** Cost is roughly $25 to $35, dominated by a ~800 GB download.
We are genuinely unsure this is worth it, because the two designs we would run on it were both
judged unidentified, so a bigger model may just buy a better-conditioned version of a test that
cannot answer the question. **We would value your view on whether this is worth buying at all,
or whether it only becomes worth buying once an identified design exists.**

## 5. Tier 2: do fitted boundaries move when the lens is fit on code instead of prose?

**Status.** Designed, never run; gated on funding approval since an earlier phase.

**Why it is interesting.** It asks whether the depth bands are a property of the *model* or of
the *estimation corpus*. If boundaries move with corpus, much of the atlas is a statement about
WikiText rather than about the network, which would be a significant caveat on our own published
map.

**Proposal.** Refit on a code corpus at the largest scale we can afford and compare fitted
boundaries. Note this is a **geometry** question, not a causal one, so none of the
identifiability problems above apply to it.

## The questions we actually want answered

1. Is the note publishable as it stands, or is something in it still overclaimed?
2. Of the five above, which are worth doing and which should we drop? We are prepared to drop
   all of them.
3. Is there a **stopping** argument we are not making, i.e. is the honest move to publish the
   geometry result plus the inconclusive block result and move to a different problem?
4. Given four self-withdrawn claims, are we now over-correcting, hedging results that deserve
   stronger statements?
