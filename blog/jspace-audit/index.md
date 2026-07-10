---
title: "A Global Workspace, or a Training Artifact? Auditing Anthropic's J-Space Across the Open-Weight Lineup"
date: 2026-07-09
draft: true
authors: ["praxagent"]
tags: ["interpretability", "replication", "j-space", "global-workspace", "open-science"]
summary: "Anthropic found a 'global workspace' inside Claude with a new tool, the Jacobian lens. We ran it on every open-weight model we could load. The measurement is real and stable — but the workspace band is not universal: it emerges with scale, spans two-plus orders of magnitude between model families at matched size, shrinks under instruct-tuning, and in our cleanest natural experiment tracks one training choice — distillation pretraining."
---

<!-- DRAFT: numbers marked [PENDING-*] finalize when the uniform sweep completes
     (final measured N, 70B anchor, OLMo-32B uniform row, shared-vocab verdict,
     precision A/B, behavioral results, figure). Everything else is receipted. -->

Last week Anthropic published [*Verbalizable Representations Form a Global Workspace in
Language Models*](https://transformer-circuits.pub/2026/workspace/index.html) — the
"J-space" paper. Using a new tool called the Jacobian lens, they found a small,
privileged set of directions inside Claude that behaves strikingly like the *global
workspace* neuroscientists associate with conscious access: its contents are reportable,
causally steer downstream reasoning, are reused flexibly across tasks, and — when
ablated — selectively destroy multi-step reasoning while leaving fluent speech intact.

The paper drew a spectrum of reaction. The originators of Global *Neuronal* Workspace
theory (Dehaene & Naccache) called it
["a landmark in consciousness research."](https://www-cdn.anthropic.com/files/4zrzovbb/website/cc4be2488d65e54a6ed06492f8968398ddc18ebe.pdf)
Critics called it PR wearing a lab coat, a triviality of backprop, or bad philosophy.
A few people did run pieces of it again — Neel Nanda replicated core findings on one
Qwen model; Elie Bakouch charted the geometry across many. But nobody had
systematically stress-tested the claim the framing invites: that this is a property of
*language models*.

We did that. Anthropic released the [code](https://github.com/anthropics/jacobian-lens)
(Apache-2.0) and, with Neuronpedia, [pre-fitted lenses for 38 open-weight
models](https://huggingface.co/neuronpedia/jacobian-lens) — Pythia-70M to
Llama-3.3-70B, six model families. We measured a workspace signature on
**[PENDING-N] of them** (two lenses were unavailable/failed to load), added null
controls, re-fit our own lenses to check stability, and asked one question:

**Is the "workspace" a universal property of language models — or a contingent one?**

## TL;DR

1. **The measurement is real, and more solid than we expected.** We re-fit lenses from
   scratch and reproduced Neuronpedia's to CKA 0.999; re-fitting on disjoint *samples of
   the same corpus* changes essentially nothing (0.997–0.998, no layer below 0.976).
   Within-distribution, the J-space is a stable property of the *model*, not of how the
   lens is estimated.
2. **The workspace band is not universal.** It is absent below ~0.2B parameters, and at
   matched scale it spans **one-to-two-plus orders of magnitude between model families**
   — reaching ~300× at 12–14B (Qwen3-14B 0.21 vs Gemma-3-12B 0.0007). Strong across
   Qwen3/3.5, moderate in Llama/OLMo, near-zero across Gemma.
   [PENDING-SHARED: survives / attenuates under tokenizer-commensurable probes.]
3. **Our leading explanation is a training choice, part-tested.** In a natural
   experiment inside Gemma-2 — where Google's report says the 2B and 9B were pretrained
   by knowledge distillation but the 27B from scratch — the distilled models score
   0.002–0.007 regardless of scale while their from-scratch sibling scores ~7–23×
   higher, with architecture, normalization, and tokenizer held constant. We committed
   the prediction (9B ≤ 0.02) to git before the value resolved (commit `0412769`); it
   came in at 0.0019, under our ceiling. (A norm-scheme confound remains — see below.)
4. **Post-training weakens it:** every base→instruct pair we measured shows a *smaller*
   band after instruct-tuning.
5. **The transport is language-general deep, language-bound early:** a lens fit on
   Chinese Wikipedia agrees with an English-fit lens at 0.99 in the deepest layers but
   only ~0.73 in the earliest.
6. **The band has a functional correlate — strong, but it's mostly a family effect.** We
   ran Anthropic's own causal experiments across 23 open models. Whether an injected concept
   *resolves in the model's verbalizable workspace* tracks the geometric band at **Spearman
   ρ = +0.786** (n=23, p < 0.001), and steering-to-flip-output tracks it too (ρ = +0.761):
   every Gemma fails to route the concept into the readout (share ≈ 0.000), while
   Qwen/Llama/OLMo route it cleanly (0.95–0.99). But two committed-in-advance
   confound-breakers show the band itself isn't the causal variable: `qwen2.5-7b-it` has
   *no* band (0.003, Gemma-like) yet resolves fully (0.965), and the *banded* Gemma-4s
   (0.047–0.050) still resolve at 0.000. The clean tell is a matched-band pair — at the same
   band, `gemma-4-e4b` (0.050) resolves 0.000 while `qwen3-4b` (0.056) resolves 0.976. Same
   geometry, opposite behavior: the discriminating variable is *family*, not the band. So
   the band strongly predicts the behavior across families (because Gemma uniquely lacks
   both), but a matched comparison shows family — plausibly the same KD-pretraining that
   suppresses the band — is what's causal. We predicted both breakers would go the other
   way; they didn't, and we report the miss.

Everything — code, per-model CSVs, fitted lenses, this post — is in
[our repo](https://github.com/praxagent/research-and-replications), with a
[DATA.md](https://github.com/praxagent/research-and-replications/blob/main/projects/jacobian-lens-and-identifiability/DATA.md)
mapping every claim to its receipt.

## What the J-lens actually is (60 seconds)

For each layer *l*, the Jacobian lens is the **average causal Jacobian**
`J_l = E[∂h_final/∂h_l]` — over a text corpus, how much does nudging the residual
stream at layer *l* change what the model is about to say? Composing that with the
unembedding gives, per vocabulary token, the internal direction that most raises the
model's disposition to *say* it. The **J-space** is the sparse set of these directions
that are strongly active; Anthropic shows its contents are reportable, steerable, and
causally load-bearing in Claude.

Two properties shape our methods. The lens is *estimated* (averaged over a corpus), so
one must check the estimate is stable before trusting anything built on it. And it is
*linear* (so a lower bound on structure — the brain–LLM literature finds nonlinear maps
recover more) and *vocabulary-indexed* (so it sees only what single tokens can name — a
limitation Anthropic's invited commentators, Eleos/Butlin et al., stress).

## Our audit: strip it to the linear algebra, then scale it

We measured one structural signature: **the workspace band**. In Claude — and in the
[cross-model explorer](https://eliebak.com/viz/jspace-open) Bakouch built — the middle
layers' J-lens token geometries are highly similar to *each other* and distinct from
early/late layers: a contiguous block in the layer×layer CKA matrix. That block is the
"workspace" in its most theory-neutral form. Our statistic, `mid_sep`, measures how much
more self-similar the middle third of layers is than its neighbors. Three safeguards:

- **A null control.** Replace each layer's transport with a scale-matched random matrix:
  `mid_sep` collapses to ~0.00 on every model we ran the null on (11 models, 70M–27B).
  The shared unembedding alone produces only a flat, band-free CKA floor (0.62–0.91
  depending on model). Whatever the band is, it is not an artifact of the readout.
- **Estimation stability.** Our gpt2 lens matches Neuronpedia's at mean CKA 0.9992; two
  independent re-fits of qwen3-4b on disjoint wikitext samples each agree with the
  reference at 0.997–0.998 (min layer 0.976). The structures are properties of models,
  not fitting noise — within-distribution (all wikitext; cross-*distribution* stability
  is future work).
- **The tokenizer confound.** Anthropic's commentators note the J-space is defined by
  the token vocabulary, so cross-family comparisons could be confounded by different
  tokenizers. We re-ran the whole comparison using only token strings shared by *all*
  the tokenizers. [PENDING-SHARED: the family gap survives / attenuates by X%.]

A methods note in full honesty: mid-run, the 70B lens wouldn't fit in RAM under our
original fp32 pipeline, so the final numbers all come from one re-run that stores
intermediates in fp16 (computation stays fp32; the lenses ship in fp16 anyway). An A/B
against the original path across the finished models: max |Δ mid_sep| = [PENDING-AB].

## Finding 1: The band emerges with scale — but scale is not the story

{{< figure src="emergence_curve.png" caption="Mid-band separation vs. parameters (log scale), colored by family; gray = random-transport null floor. [PENDING-FIGURE]" >}}

Below ~0.2B parameters (Pythia-70M, GPT-2) there is no band: `mid_sep` ≈ 0.015 — small
next to every banded model (0.05–0.21), though above the random-transport null
(~0.000–0.002). By ~0.8B, bands appear — in some families. From there the curve does not
converge; it **fans out by family**:

| Family (best size measured) | mid_sep | Pretraining |
|---|---|---|
| Qwen3-14B | 0.21 | from scratch |
| Qwen3.5-27B | 0.20 | from scratch |
| OLMo-3-32B | [PENDING-OLMO ~0.094] | from scratch |
| Llama-3.1-8B | 0.087 | from scratch |
| gpt-oss-20b | 0.076 | "large-scale distillation and RL" (mechanism undisclosed) |
| Gemma-2-27B | 0.043 | **from scratch** |
| Gemma-3-27B | 0.025 | distilled |
| Gemma-2-9B | **0.0019** | **distilled** |
| Gemma-3-12B | **0.0007** | distilled |

The top of the range holds the pattern — and then breaks it open. **Llama-3.3-70B-Instruct
measures 0.148** (a solid band at 70B, instruct-tuned at that). And when we fit our own
lens on **Qwen3.5-397B-A17B** — 8×H200, the largest open MoE we could hold — the band came
in at **0.343** (n=24), 1.6× the strongest model in the entire 38-lens Neuronpedia sweep. The workspace band is not a small-model transient:
in the lineages that have it, **it grows all the way to 0.4T**.

At 12–14B the difference between Qwen and Gemma is ~**300×**. Both are residual-stream
transformers trained with backprop. Whatever produces the workspace band, it is not
"being a large language model." (One within-family wrinkle we don't hide: an older
qwen2.5-7b-it sits near the floor at 0.003 — the strong-band claim is specifically about
the Qwen3/3.5 *base* models we measured.)

## Finding 2: The leading suspect — distillation pretraining

The family table hides a natural experiment. Google's Gemma-2 report states the 2B and
9B were pretrained by **knowledge distillation** (learning to match a teacher's token
distributions) while the **27B was trained from scratch**. Same architecture, same
sandwich normalization, same tokenizer, same data lineage:

| Gemma-2 | Pretraining | mid_sep |
|---|---|---|
| 2B | distilled | 0.007 |
| 9B | distilled | 0.0019 |
| 27B | from scratch | 0.043 |

The distilled models sit at the floor *regardless of scale* — the 9B *below* the 2B —
while the from-scratch sibling scores ~7× (vs 2B) to ~23× (vs 9B) higher. We wrote the
prediction (9B ≤ 0.02) into git before the value resolved; it came in at 0.0019.
(Honesty note: the sweep had computed the number shortly before we looked — this is
predicted-before-seeing, not formal pre-registration; and the interval we sketched was
0.007–0.02, so it landed *under*, not inside.)

Why would distillation matter? There's prior art: distillation students demonstrably
[reorganize internal computation](https://arxiv.org/abs/2505.10822) rather than inherit
the teacher's, and distillation can [degrade a student's layerwise representational
geometry unless it is explicitly aligned](https://arxiv.org/abs/2606.05682). Intuition:
a student trained to mimic a teacher's output distribution needn't *develop its own*
internal serial workspace — the teacher already did that thinking.

**We call this a leading hypothesis, not a result**, and the honesty here matters.
Gemma uniquely combines distillation with *sandwich normalization*, so those two
explanations are confounded in this cohort; the within-Gemma-2 experiment holds norm
constant and is our best evidence, but it's one family. Gemma-3-4B (distilled) reaches
0.046, overlapping the from-scratch range. gpt-oss's "distillation and RL" is too
vaguely described to place. The competing hypotheses, per-layer maps, and falsification
tests are all in
[`hypotheses.md`](https://github.com/praxagent/research-and-replications/blob/main/projects/jacobian-lens-and-identifiability/hypotheses.md).

One hypothesis we *can* rule out in its mechanical form: attention *wiring*. Qwen3.5
runs 75% linear-attention layers yet posts bands as strong as any full-attention model's
(0.15–0.20 across every size); and Gemma's sliding windows don't even bind at the
128-token contexts the lenses are fit on. The band doesn't care how attention is wired.
(A subtler "training under tiny windows shapes the weights" version survives as a
weak secondary hypothesis.)

## Finding 3: Post-training and language

**The band predicts a behavior (geometry → function) — strongly, but it's mostly a family
effect.** We ran Anthropic's own causal experiments across 23 open models and correlated
them with the geometric band. The cleanest link: whether interpolating a concept into a
token makes that concept *resolve in the J-lens readout* (`share_span`) tracks `mid_sep` at
**ρ = +0.786 (n = 23, p < 0.001)**, and steering-to-flip-output at ρ = +0.761 — every Gemma
at 0.000 (the concept never reaches the verbalizable workspace), Qwen/Llama/OLMo at
0.95–0.99.

**But two committed-in-advance confound-breakers show the band itself isn't the causal
variable — and this is the most interesting result here.** We added `qwen2.5-7b-it` (a Qwen
with an almost non-existent band, 0.003) and the *banded* Gemma-4s (0.047–0.050) precisely
to separate "band drives behavior" from "family drives both." Both predictions missed, in
the same direction: the no-band Qwen resolves fully (0.965), and the banded Gemmas still
resolve at 0.000. The decisive datum is a matched-band pair — at essentially the same
geometric band, `gemma-4-e4b` (0.050) resolves **0.000** while `qwen3-4b` (0.056) resolves
**0.976**. Same geometry, opposite behavior: the discriminating variable is *family*, not
`mid_sep`. So the ρ≈0.78 correlation is real but confounded — Gemma uniquely lacks both the
band and the function (plausibly the same KD-pretraining that suppresses the band, from
Finding 2), and the cross-model correlation is largely that Gemma-vs-rest split.
"Band = workspace" is too strong; "band predicts the behavior across families, but a matched
comparison pins the cause on family" is the defensible claim. The smaller n=13 sample had
suggested a cleaner ρ=0.835; more data and the confound-breakers made it more honest. The
weakest leg is all-or-none "ignition" sharpness (ρ = 0.46, n = 15, borderline), so we do
*not* claim the workspace shows human-like ignition. One model, gemma-2-9b, is a separate
oddity: steerable (swap 0.82) yet zero concept-resolution and near-zero band. Full numbers
+ method:
[`experiments/behavioral/results.md`](https://github.com/praxagent/research-and-replications/blob/main/projects/jacobian-lens-and-identifiability/experiments/behavioral/results.md).

**Instruct-tuning shrinks the band in every pair we could measure.** gemma-3-4b:
0.046 → 0.015. gemma-2-2b: 0.007 → 0.001. llama-3.1-8b: 0.087 → 0.079. This is geometry,
not content — but it is at least in tension with a casual reading of "post-training
installs the Assistant into the workspace." (Anthropic's own §6.1 results show
post-training changes what the workspace *represents*; we add that its geometric
distinctness *decreases*.)

**The transport is language-bound early, language-general deep.** We fit a qwen3-4b lens
on Chinese Wikipedia and compared to the English-fit lens: agreement is ~0.73 in the
earliest layers, rising monotonically to 0.99 at the top — against an English-vs-English
baseline averaging 0.997 (never below 0.976). The English-anchoring that
[Wendler et al.](https://arxiv.org/abs/2402.10588) describe lives in the early
verbalization layers; the deep, workspace-adjacent geometry is close to
language-general. This *refines* Anthropic's multilingual claim and states where it
breaks.

## So: was "global workspace" oversold?

The calibrated version, claim by claim.

**What survives — and it's substantial.** The core measurement is excellent and
replicates in independent hands: ours across the open lineup, and DeepMind's Neel Nanda
on Qwen (who found the bands "notably less clean than the paper's" — consistent with our
family-dependence). The J-space is a stable, causally meaningful structure. Anthropic
also *hedged the strong claim themselves*: their thread explicitly claims a mechanism
for *conscious access* (what philosophers call access consciousness), **not** phenomenal
experience. The "it's all PR" line doesn't survive the fact that the theory's own
originators engaged it as serious, testable science.

**What we think doesn't survive: the implied universality.** "Language models develop a
global workspace" reads as a claim about language models. Our data says it is a claim
about *some* of them. The band is absent in small models, near-absent across an entire
major family, strongest in another, weakened by instruct-tuning, and — in our cleanest
experiment — tracks distillation pretraining. The workspace looks **contingent, not
constitutive**.

**And the deflationary critiques fail symmetrically.** "Trivially expected from backprop
and residual streams" (Trask's argument) and "every layered system has internals that
don't surface — should my .docx get a workspace?" (Barenholtz's reduction) both predict
the band should be *everywhere*. It isn't. Gemma-3-12B is a backprop-trained residual
transformer, and its band is 0.0007. **The hype and the anti-hype make the same mistake:
treating the phenomenon as necessary — as consciousness, or as triviality — when it is
contingent.** Contingent structure is the interesting kind: it demands an explanation
(we've proposed and partly tested one) rather than a label.

On the philosophy we defer to the experts and take no position on phenomenal
consciousness. The Eleos commentary's tiering is the rigorous frame: the evidence firmly
establishes a *privileged set* of accessible representations, only suggestively a unified
*stream*, and not the full workspace. Barenholtz's wedge cuts deepest — an LLM producing
"linguistic report … workspace and all, without any sensory grounding" arguably
*strengthens* the dissociation between reportability and experience. The better the
workspace result, the weaker the consciousness inference.

## Limitations, up front

- **Geometry vs. behavior — a correlation confounded by family.** `mid_sep` is our own CKA
  block-structure statistic, not Anthropic's full ablation battery. We tested the link
  directly (above): concept-resolution tracks the band at ρ = 0.786 (n = 23, p<0.001), real
  reassurance that a low `mid_sep` usually reflects a functional difference. But two
  confound-breakers (a no-band Qwen that resolves, banded Gemmas that don't) and a matched-
  band pair (gemma-4-e4b 0.050→0.000 vs qwen3-4b 0.056→0.976) show the causal variable is
  *family*, not the band itself — the two are entangled because Gemma uniquely lacks both.
  We don't claim `mid_sep` is a complete functional characterization of a "workspace"; it's
  a strong-but-confounded proxy.
- **Frontier scope — now measured, one lineage.** We fit our own lens on
  Qwen3.5-397B-A17B (n=24 prompts, converged per our calibration; held-out fidelity evals
  + an ignition run through the lens as sanity checks) and measured
  **mid_sep = 0.343** — the band survives, indeed peaks, at 0.4T. Scope
  honesty: that's ONE lineage (Qwen); we did not fit other frontier families, and Nanda's
  n=4 fit reported no geometry number to compare against. The lens is published for
  independent checking ([PENDING HF LINK]).
- **The tokenizer confound** is [PENDING-SHARED].
- **The distillation hypothesis is observational.** N(families) is small, teachers are
  undisclosed, and Gemma's sandwich-norm is a real, only-partially-broken confound. The
  decisive test — pretrain matched models with and without distillation — is beyond our
  budget.
- **One language pair, one model** for the language finding, so far.

## What's next

Frontier-scale fits (the transient question), Anthropic's behavioral battery across more
families (does Gemma *behave* workspace-less, or just measure that way?), the ignition
and capacity tests proposed by Dehaene & Naccache, and the bridge to brain-alignment —
does our band coincide with the layers that predict human brain activity? The
[project docs](https://github.com/praxagent/research-and-replications/tree/main/projects/jacobian-lens-and-identifiability)
track all of it.

## Credit

This audit exists because Anthropic released working code and, with Neuronpedia,
pre-fitted lenses — open science done right. The "middle layers are special" lineage
predates J-space: Schrimpf, Caucheteux & King, Jain & Huth, Goldstein, Toneva and the
brain–LLM alignment community, whose priority Jean-Rémi King has
[rightly emphasized](https://x.com/jeanremiking/status/2074500550947680368). The
identifiability theory framing when any of this is trustworthy is due to Zheng, Zhang
and colleagues. Commentary that sharpened this audit: Dehaene & Naccache, the Eleos
team, Neel Nanda, and the critics — Ravid, Trask, Barenholtz — whose objections we tried
to test rather than dismiss.

*All code, data, and fitted lenses: [github.com/praxagent/research-and-replications](https://github.com/praxagent/research-and-replications). Corrections welcome — that's what the receipts are for.*
