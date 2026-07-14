# Background — why replicate *Features as Rewards*

## The claim, in one line

If a model's internal representations track the factual validity of its own output
*and that readout is calibrated*, then a cheap probe on those representations can serve
as a dense reward for reducing hallucination — turning interpretability from a
monitoring tool into a *supervision* signal.

## What the paper actually establishes

The paper (Goodfire, arXiv 2602.10067) builds **RLFR** (RL from Feature Rewards): four
attention probes on residual-stream activations (localize an entity → classify it as
hallucinated → grade a retraction → grade a correction), trained to imitate a
Gemini-2.5-Pro-with-web-search grader on LongFact++, then used as the reward in an RL
loop on Gemma-3-12B-IT. Headline: a policy **58% less likely to hallucinate** with
best-of-32, ~90× cheaper than external evaluation, benchmarks preserved.

Two facts govern how much of this is worth our money:

1. **The 58% is mostly not a weight change.** The paper's own decomposition: 10% from
   the trained policy, 35% from inline interventions trickling through in-context, 13%
   from direct correction. The scientific core is upstream of the RL run.
2. **The load-bearing sub-claim is the probe calibration**, and it is cheap to test:
   Localize AUC .88 / Classify AUC .94 on held-out data. Everything the RL loop does
   rests on those probes being calibrated readouts of the model's belief.

So we replicate the calibration and the test-time (no-RL) result, and skip the
expensive RL run (Tier 3, excluded).

## Why it is worth our time

- It is a concrete, checkable instance of the broader thesis our campaign already
  circles: **a model's internal state carries information about the truth of its own
  output that its output does not always reveal.** Our workspace-under-pressure result
  is the same phenomenon in miniature — a logit/Jacobian-lens readout recovered a held
  "Moscow" while the model asserted "Kiev." *Features as Rewards* scales that from a
  single caught lie to a calibrated, usable signal over thousands of entities.
- It has a clean safety framing: grounding training in an interpretability signal, and
  a monitor that can flag its own uncertainty. Worth understanding whether the signal is
  real and *which instrument* reads it.

## The question the paper leaves open (our extension)

Despite Goodfire being an SAE company, this paper's "features" are **supervised
attention probes on raw activations — not SAE latents** ("SAE"/"dictionary" appear zero
times in the body; sparse autoencoders only in the reference list). That leaves an
obvious question unasked: **which interpretability object actually carries the
calibrated hallucination signal?**

We answer it by scoring the *same* held-out entity spans with four readers:

| Reader | Supervision | Cost | Whose instrument |
|---|---|---|---|
| Attention probe | supervised on gold labels | train a probe | the paper's |
| Logit lens | none (identity transport → unembed) | free | the mandated baseline; our lie-catch reader |
| Jacobian lens | none (fitted transport, unsupervised) | fit once | ours |
| SAE latent | none (pretrained dictionary feature) | pick a latent | Goodfire's own, unused here |

If an *unsupervised* reader (logit/J-lens) or a *sparse* one (SAE latent) matches the
supervised probe's AUROC, the calibrated signal is a property of the **features**, not
of the label-trained probe — a stronger and cheaper claim than the paper makes. If only
the supervised probe reads it, that is also a real, publishable result: the signal needs
supervision to surface. Either way the four-reader table is the finding, per the
"report the contrast, not the target value" rule.

## Falsifiable boundary (to be frozen in the protocol)

Under each pinned model, the LongFact++ prompt population, and the public gold-label
annotation set, a frozen readout distinguishes hallucinated from supported entity spans
at the reported out-of-sample AUROC/calibration. This does **not** claim: that the
signal is "the model's belief" in any mental-state sense; that it transfers to another
model, prompt distribution, or hallucination type; or anything about the RL policy (Tier
3, not run). See `PROPOSAL.md` for the tiered plan and `FEASIBILITY.md` for artifacts.

## Sources
- Features as Rewards — arXiv 2602.10067 (Goodfire).
- Real-Time Detection of Hallucinated Entities / LongFact++ — arXiv 2509.03531
  (Obeso, Arditi, Ferrando, Freeman, Holmes, Nanda), hallucination-probes.com.
- Gemma Scope 2 — `google/gemma-scope-2` (DeepMind), SAEs on the Gemma-3 family.
