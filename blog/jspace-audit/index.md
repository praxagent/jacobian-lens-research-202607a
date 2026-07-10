---
title: "A Global Workspace, or a Training Artifact? Auditing Anthropic's J-Space Across the Open-Weight Lineup"
date: 2026-07-10
draft: true
authors: ["praxagent"]
tags: ["interpretability", "replication", "j-space", "global-workspace", "open-science"]
summary: "Anthropic found a 'global workspace' inside Claude with a new tool, the Jacobian lens. We ran it on every open-weight model we could load — then re-ran everything on tokenizer-commensurable probes, which overturned our own headline: the dramatic family gap was mostly measurement artifact. What actually varies with training: distillation-pretrained Gemma-2s have no band, instruct-tuning shrinks it everywhere, and — the sharpest finding — models with identical band geometry differ absolutely in whether concepts can functionally enter the workspace."
---

<!-- DRAFT: all numbers FINAL (uniform one-code-path sweep + shared-vocab re-sweep +
     precision A/B, 2026-07-10). Receipts: experiments/jacobian_lens/results.md +
     emergence*.csv + ab_report.txt; experiments/behavioral/results.md +
     behavioral_correlation*.csv; experiments/fit_our_own/results.md (397B). -->

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
**35 of them** (qwen3.6-27b's lens failed to load, qwen3-32b's lens stores its
weights in a layout our loader doesn't parse yet, and gemma-4-31b's unembedding
needs a ~50 GB model pull our CPU box couldn't hold), added null controls, re-fit
our own lenses to check stability, and asked one question:

**Is the "workspace" a universal property of language models — or a contingent one?**

## TL;DR

1. **The measurement is real, and more solid than we expected.** We re-fit lenses from
   scratch and reproduced Neuronpedia's to CKA 0.999; re-fitting on disjoint *samples of
   the same corpus* changes essentially nothing (0.997–0.998, no layer below 0.976).
   Within-distribution, the J-space is a stable property of the *model*, not of how the
   lens is estimated.
2. **The dramatic family gap was mostly a measurement artifact — and catching it is one
   of this audit's main results.** Probed naively (each model in its own vocabulary),
   the families fan out by orders of magnitude at matched size — Qwen3-14B 0.21 vs
   Gemma-3-12B 0.0007, ~300×. But the J-space is *vocabulary-indexed*, and Anthropic's
   own commentators flagged the confound. Re-probing every model on the 4096 token
   strings shared by **all** tokenizers collapses the gap: base-family means go from
   **6.2× apart to 1.4×** (1.14× excluding the two distilled Gemma-2s), the 12–14B
   "300×" becomes **1.8×** (0.206 vs 0.114), and Gemma-3-27B — 0.025 own-vocab — turns
   out to have the **strongest band in the entire sweep (0.298)**, ahead of every Qwen.
   Gemma's giant 262k vocabulary was diluting its probe set. Measured fairly, every
   big-enough base model we tested — **except the two KD-pretrained Gemma-2s** (see #3)
   — has a band; what varies with training recipe is
   *how much* (and see #6 for what varies absolutely). The sub-0.2B floor is real under
   both probings (Pythia-70M/GPT-2 ≈ 0.015).
3. **The cleanest training-choice result survives fair probing — narrower but
   stronger.** In a natural experiment inside Gemma-2 — Google's report says the 2B and
   9B were pretrained by knowledge distillation but the 27B from scratch — the distilled
   models stay at the floor under shared probes (0.007/0.005) while their from-scratch
   sibling rises to 0.113: **15–24×**, with architecture, normalization, tokenizer *and
   probe set* held constant. We committed the 9B prediction (≤ 0.02) to git before the
   value resolved (commit `0412769`). But the same shared probes broke our *cross*-family
   generalization: Gemma-3 is also distillation-trained (per its tech report), and posts
   the strongest band we measured. KD-suppression is a fact about Gemma-2's specific
   recipe, not about distillation per se — a scope-narrowing we report plainly.
4. **Post-training weakens it:** every base→instruct pair we measured shows a *smaller*
   band after instruct-tuning.
5. **The transport is language-general deep, language-bound early:** a lens fit on
   Chinese Wikipedia agrees with an English-fit lens at 0.99 in the deepest layers but
   only ~0.73 in the earliest.
6. **The sharpest finding: geometry and function dissociate.** We ran Anthropic's own
   causal experiments across 23 open models. Whether an injected concept *resolves in
   the model's verbalizable workspace* tracks the geometric band at **Spearman
   ρ = +0.739** own-vocab, and the correlation **survives the shared-probe correction
   attenuated (ρ = +0.534, n=23, p=0.009)**; steering-to-flip-output tracks it too
   (+0.741 → +0.524). But the shared probes turned our confound-breakers into something
   stronger. Under fair probing, several Gemmas have *solid* bands — and still never
   resolve: `gemma-2-27b` and `qwen3.5-2b` have the *same* band (0.113 vs 0.114) with
   share_span **0.000 vs 0.970**; `gemma-3-12b` vs `qwen3.5-0.8b` likewise (0.114 vs
   0.115 → 0.000 vs 0.948); and `qwen2.5-7b-it`, with almost no band (0.014), resolves
   fully (0.965). Identical geometry, absolute behavioral difference. Gemma doesn't
   uniquely lack the band — **it uniquely lacks the function**: all eight Gemmas measured
   sit at share_span 0.000, across bands spanning 0.005–0.114. The band is neither
   necessary nor sufficient for the workspace *behavior*; something family-bound
   (training recipe, plausibly, though Gemma-3's KD complicates the specific KD story)
   gates whether concepts can functionally enter. We predicted the confound-breakers
   would go the other way; they didn't, and we report the miss.

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
  tokenizers. We re-ran the whole comparison using only the 4096 token strings shared by
  *all* the tokenizers — and the confound turned out to be **real and material**: it was
  inflating the family gap by roughly 4× on family means (own-vocab 6.2× → shared 1.4×),
  with single-model swings as large as +0.27 (Gemma-3-27B). Every cross-family number in
  this post is therefore quoted in both probings, and cross-family *claims* rest on the
  shared-probe values. This is the audit auditing itself; we kept the correction.

A methods note in full honesty: mid-run, the 70B lens wouldn't fit in RAM under our
original fp32 pipeline, so the final numbers all come from one re-run that stores
intermediates in fp16 (computation stays fp32; the lenses ship in fp16 anyway). An A/B
against the original path across all 34 overlapping models: **max |Δ mid_sep| = 0.00000**
— the precision change had no effect on any number.

## Finding 1: The band emerges with scale — and the family story is smaller than it looks

{{< figure src="emergence_curve_shared.png" caption="Mid-band separation vs. parameters (log scale) under tokenizer-commensurable (shared) probes — the fair cross-family comparison. Solid = base models, dashed/open = instruct variants, gray = random-transport null floor (measured under own-vocab probes; a shared-probe null re-run is queued)." >}}

Below ~0.2B parameters (Pythia-70M, GPT-2) there is no band under either probing:
`mid_sep` ≈ 0.015 — small next to every banded model, though above the random-transport
null (~0.000–0.002). Above that, bands appear — and measured *fairly*, in more places
than the naive measurement shows. Both probings, per family at its best measured size:

| Family (best size) | own-vocab | shared-probe | Pretraining |
|---|---|---|---|
| Gemma-3-27B | 0.025 | **0.298** | distilled |
| Qwen3-14B | 0.211 | **0.206** | from scratch |
| Qwen3.5-27B | 0.197 | **0.196** | from scratch |
| OLMo-3-32B | 0.094 | **0.131** | from scratch |
| Llama-3.1-8B | 0.087 | **0.106** | from scratch |
| Gemma-2-27B | 0.043 | **0.113** | **from scratch** |
| gpt-oss-20b | 0.076 | 0.057 | "large-scale distillation and RL" (undisclosed) |
| Gemma-3-12B | 0.0007 | **0.114** | distilled |
| Gemma-2-9B | **0.0019** | **0.0047** | **distilled** |

Read the two columns against each other and the story reorganizes. Qwen barely moves
(max shift 0.05, most ≤0.02). Gemma moves enormously — Gemma-3-27B goes from "band-poor" (0.025) to the
**strongest band in the sweep** (0.298); Gemma-3-12B from 0.0007 (the number our earlier
draft called "essentially no band at 12B") to 0.114. The own-vocab "~300×" Qwen-vs-Gemma
contrast at 12–14B is **1.8×** under commensurable probes, and the base-family means are
1.4× apart (1.14× excluding the distilled Gemma-2s). Gemma's 262k-token vocabulary —
the largest in the study — was flooding its own probe set with tokens that dilute the
band statistic. The exceptions that stay at the floor under fair probing are precise:
the two KD-pretrained Gemma-2s (0.005–0.007) and, to a lesser degree, instruct variants
everywhere (dashed lines in the figure). One more surprise we don't hide: Gemma-3-270M
posts 0.134 under shared probes — a solid band at 0.27B, which muddies the clean
"emerges at ~0.8B" onset story our own-vocab data suggested. (And qwen2.5-7b-it sits
near the floor in both probings — 0.003/0.014 — an instruct model, but notably below
llama3.1-8b-it's 0.081.)

The top of the range holds. **Llama-3.3-70B-Instruct measures 0.148 own-vocab / 0.137
shared** (a solid band at 70B, instruct-tuned at that). And when we fit our own lens on
**Qwen3.5-397B-A17B** — 8×H200, the largest open MoE we could hold — the band came in at
**0.343** (n=24, own-vocab; Qwen models shift ≤0.05 under shared probes, so the
qualitative claim is probing-robust, though a shared-probe recompute of the 397B itself
is queued), 1.6× the strongest own-vocab band in the Neuronpedia sweep and above the
strongest shared-probe band (gemma-3-27b, 0.298). The workspace band is not a
small-model transient: **it grows all the way to 0.4T**.

## Finding 2: The leading suspect — Gemma-2's distillation recipe (not distillation per se)

The family table hides a natural experiment. Google's Gemma-2 report states the 2B and
9B were pretrained by **knowledge distillation** (learning to match a teacher's token
distributions) while the **27B was trained from scratch**. Same architecture, same
sandwich normalization, same tokenizer, same data lineage — and, under shared probes,
same probe set:

| Gemma-2 | Pretraining | mid_sep (own-vocab) | mid_sep (shared) |
|---|---|---|---|
| 2B | distilled | 0.007 | 0.007 |
| 9B | distilled | 0.0019 | 0.005 |
| 27B | from scratch | 0.043 | **0.113** |

The distilled models sit at the floor *regardless of scale* while the from-scratch
sibling separates — and the tokenizer correction, which shrank every *cross*-family gap,
made this *within*-family gap **larger**: 15× (vs 2B) to 24× (vs 9B) under shared
probes. It is the single most probing-robust contrast in our data. We wrote the
prediction (9B ≤ 0.02) into git before the value resolved; it came in at 0.0019.
(Honesty note: the sweep had computed the number shortly before we looked — this is
predicted-before-seeing, not formal pre-registration; and the interval we sketched was
0.007–0.02, so it landed *under*, not inside.)

**And the same correction broke our generalization of it.** Gemma-3 is *also*
distillation-pretrained (per its tech report, all sizes), and under shared probes
Gemma-3 models post strong bands — the 27B the strongest in the 35-model sweep (0.298).
On the own-vocab numbers we had read Gemma-3's low scores as more KD-suppression; that
reading was tokenizer artifact. So the defensible claim narrows: **something in
Gemma-2's specific distillation recipe suppresses the band; distillation per se does
not.** (Teacher identity, KD loss details, and data mix all differ between Gemma
generations and are undisclosed — we can't isolate which ingredient.)

Why would distillation matter? There's prior art: distillation students demonstrably
[reorganize internal computation](https://arxiv.org/abs/2505.10822) rather than inherit
the teacher's, and distillation can [degrade a student's layerwise representational
geometry unless it is explicitly aligned](https://arxiv.org/abs/2606.05682). Intuition:
a student trained to mimic a teacher's output distribution needn't *develop its own*
internal serial workspace — the teacher already did that thinking.

**We call this a leading hypothesis, not a result**, and after the shared-probe
correction its scope is explicitly narrow: one family generation, mechanism unknown.
Gemma uniquely combines distillation with *sandwich normalization*, so those two
explanations are confounded in this cohort; the within-Gemma-2 experiment holds norm
constant and is our best evidence, but it's one contrast in one family — and Gemma-3's
strong shared-probe bands rule out the simple "KD suppresses the workspace"
generalization outright. gpt-oss's "distillation and RL" is too vaguely described to
place. The competing hypotheses, per-layer maps, and falsification tests are all in
[`hypotheses.md`](https://github.com/praxagent/research-and-replications/blob/main/projects/jacobian-lens-and-identifiability/hypotheses.md).

One hypothesis we *can* rule out in its mechanical form: attention *wiring*. Qwen3.5
runs 75% linear-attention layers yet posts strong bands at every size (0.15–0.20
own-vocab, 0.11–0.20 shared — top-tier among base models either way); and Gemma's
sliding windows don't even bind at the
128-token contexts the lenses are fit on. The band doesn't care how attention is wired.
(A subtler "training under tiny windows shapes the weights" version survives as a
weak secondary hypothesis.)

## Finding 3: Geometry meets function — and dissociates (plus post-training and language)

**The band predicts a behavior (geometry → function) — and then, at matched geometry,
stops predicting it at all.** We ran Anthropic's own causal experiments across 23 open
models and correlated them with the geometric band. The cleanest link: whether
interpolating a concept into a token makes that concept *resolve in the J-lens readout*
(`share_span`) tracks `mid_sep` at **ρ = +0.739 own-vocab, +0.534 under shared probes
(n = 23, p = 0.009)** — the correlation survives the tokenizer correction, attenuated.
Steering-to-flip-output tracks it too (+0.741 → +0.524). Every Gemma resolves at 0.000
(the concept never reaches the verbalizable workspace); Qwen/Llama/OLMo at 0.95–0.99.
(A transparency note in the same spirit as the tokenizer fix: our verification pass
caught that an earlier draft's rank-correlation code mishandled ties and inflated these
ρ's — e.g. share_span read +0.786. All values here are tie-corrected and match scipy;
the receipts ledger documents the correction.)

**But matched-geometry comparisons show the band itself isn't the causal variable — and
this is the most interesting result here.** We added two committed-in-advance
confound-breakers — `qwen2.5-7b-it` (a Qwen with almost no band) and the *banded*
Gemma-4s — and both predictions missed in the same direction: the no-band Qwen resolves
fully (0.965), the banded Gemmas still resolve at 0.000. The shared-probe re-sweep then
upgraded this from two data points to a pattern. Under fair probing, Gemma models have
*solid* bands — and still never resolve. The matched pairs are now stark:
`gemma-2-27b` vs `qwen3.5-2b`: bands **0.113 vs 0.114**, share_span **0.000 vs 0.970**.
`gemma-3-12b` vs `qwen3.5-0.8b`: bands **0.114 vs 0.115**, share_span **0.000 vs 0.948**.
Identical geometry, absolute behavioral difference — eight Gemmas at exactly 0.000 across
bands from 0.005 to 0.114. The band is neither necessary (qwen2.5-7b-it) nor sufficient
(every banded Gemma) for the function. Something family-bound gates whether concepts can
*functionally enter* the workspace, and our CKA statistic doesn't see it. (Whether
that gate is Gemma's training recipe, and which ingredient, is exactly the open question
Finding 2 narrows.) "Band = workspace" is too strong; "the band predicts the behavior
across families, but matched comparisons dissociate them" is the defensible claim. The
smaller n=13 sample had suggested a cleaner ρ≈0.84; more data, the confound-breakers,
the tokenizer correction, and the tie-handling fix each made it more honest. The weakest
leg is all-or-none "ignition" sharpness (ρ = 0.45 own / 0.51 shared, n = 15, borderline
— not significant two-tailed), so we do *not*
claim the workspace shows human-like ignition. One model, gemma-2-9b, is a separate
oddity: steerable (swap 0.82) yet zero concept-resolution and near-zero band. Full
numbers + method:
[`experiments/behavioral/results.md`](https://github.com/praxagent/research-and-replications/blob/main/projects/jacobian-lens-and-identifiability/experiments/behavioral/results.md).

**Instruct-tuning shrinks the band in every pair we could measure — all eight, under
both probings, and the shared probes reveal the effect is much bigger than we first
reported.** Under commensurable probes: gemma-3-27b 0.298 → 0.104, gemma-3-12b
0.114 → 0.030, gemma-3-270m 0.134 → 0.025, gemma-3-1b 0.077 → 0.019, gemma-3-4b
0.061 → 0.013, llama-3.1-8b 0.106 → 0.081, and the two floor-level Gemma-2 pairs shrink
further still. This is geometry, not content — but it is at least in tension with a
casual reading of "post-training installs the Assistant into the workspace."
(Anthropic's own §6.1 results show post-training changes what the workspace
*represents*; we add that its geometric distinctness *decreases* — and that after the
tokenizer correction, instruct-tuning is a *larger* geometric effect than family.)

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
on Qwen (who found the bands "notably less clean than the paper's" — consistent with
band strength varying by training recipe and scale). The J-space is a stable, causally
meaningful structure. Anthropic
also *hedged the strong claim themselves*: their thread explicitly claims a mechanism
for *conscious access* (what philosophers call access consciousness), **not** phenomenal
experience. The "it's all PR" line doesn't survive the fact that the theory's own
originators engaged it as serious, testable science.

**What we think doesn't survive: the implied universality — though we must report that
it fares better than our own first measurement said.** "Language models develop a global
workspace" reads as a claim about language models. Our own-vocab sweep seemed to refute
it dramatically (a whole family near zero); the tokenizer correction took most of that
back, and we keep the correction: measured fairly, every big-enough *base* model we
tested has a band — except the two KD-pretrained Gemma-2s, which is axis (1) below.
What remains genuinely contingent — on three separate, probing-robust
axes — is: (1) the *geometry* at the training-recipe level: KD-pretrained Gemma-2s sit
at the floor at every scale; (2) the *geometry* under post-training: instruct-tuning
shrinks the band in all eight pairs, up to 5×; and (3) — the strongest — the *function*:
whether an injected concept can actually enter the verbalizable workspace splits by
family (eight Gemmas at exactly 0.000, every Qwen/Llama/OLMo at 0.95+, with gpt-oss-20b
a genuine intermediate at 0.52), completely unpredicted by
band geometry at matched values. The workspace *band* looks nearer to constitutive than
we first reported; the workspace *behavior* looks **contingent, not constitutive**.

**And the deflationary critiques still fail symmetrically.** "Trivially expected from
backprop and residual streams" (Trask's argument) and "every layered system has
internals that don't surface — should my .docx get a workspace?" (Barenholtz's
reduction) both predict the signatures should be *everywhere, uniformly*. They aren't.
Gemma-2-9B is a backprop-trained residual transformer whose band is 0.005 under the
fairest probing we have; and at *identical* band geometry (0.113 vs 0.114),
gemma-2-27b routes zero concepts into its readout while qwen3.5-2b's readout sweeps 97%
of the way (84% of its readouts fully resolve) — a
functional difference no "it's just backprop" account explains. **The hype and
the anti-hype make the same mistake: treating the phenomenon as necessary — as
consciousness, or as triviality — when it is contingent.** Contingent structure is the
interesting kind: it demands an explanation (we've proposed one, tested it, and narrowed
it when the data said so) rather than a label.

On the philosophy we defer to the experts and take no position on phenomenal
consciousness. The Eleos commentary's tiering is the rigorous frame: the evidence firmly
establishes a *privileged set* of accessible representations, only suggestively a unified
*stream*, and not the full workspace. Barenholtz's wedge cuts deepest — an LLM producing
"linguistic report … workspace and all, without any sensory grounding" arguably
*strengthens* the dissociation between reportability and experience. The better the
workspace result, the weaker the consciousness inference.

## Limitations, up front

- **Geometry vs. behavior — correlated across the lineup, dissociated at matched
  values.** `mid_sep` is our own CKA block-structure statistic, not Anthropic's full
  ablation battery. We tested the link directly (above): concept-resolution tracks the
  band at ρ = 0.739 own-vocab / 0.534 shared-probe (n = 23) — but matched-band pairs
  (gemma-2-27b 0.113 → share_span 0.000 vs qwen3.5-2b 0.114 → 0.970) show `mid_sep` does
  not measure whatever gates the *function*. We don't claim `mid_sep` is a functional
  characterization of a "workspace"; it's a geometric statistic whose functional
  correlate dissociates exactly where the comparisons are cleanest.
- **Frontier scope — now measured, one lineage.** We fit our own lens on
  Qwen3.5-397B-A17B (n=24 prompts, converged per our calibration; held-out fidelity evals
  + an ignition run through the lens as sanity checks) and measured
  **mid_sep = 0.343** — the band survives, indeed peaks, at 0.4T. And it's not just
  geometry: running the ignition battery *through our own 397B lens*, injected concepts
  resolve near-perfectly (share_span 0.988, 84% sharp) — the strongest workspace readout
  in our whole behavioral dataset, landing out-of-sample exactly where the
  geometry→function correlation predicts the biggest band should. Scope
  honesty: that's ONE lineage (Qwen); we did not fit other frontier families, and Nanda's
  n=4 fit reported no geometry number to compare against. The lens is published for
  independent checking
  ([praxagent/jacobian-lens-qwen3.5-397b-a17b](https://huggingface.co/praxagent/jacobian-lens-qwen3.5-397b-a17b))
  — and we ran that check ourselves before asking anyone else to: a fresh pod pulling
  only the HF artifacts reproduced the lens hash exactly and passed a pre-registered
  two-act behavioral trial (hidden two-hop entities read at median rank 43 of 248,320
  where identity/random transports read noise; causal steering flips 64% of outputs
  with 0/50 for both controls). Full protocol + receipts:
  `experiments/lens_demo/`.
- **The tokenizer confound was real, and correcting it rewrote our own headline** —
  own-vocab probing had inflated the family gap ~4×. We treat this as the audit's core
  methods lesson: *any* cross-family claim built on a vocabulary-indexed object (J-space
  included, ours included) needs commensurable probes before it's a claim. Details and
  full both-probings table: `experiments/jacobian_lens/results.md`.
- **The distillation hypothesis is observational.** N(families) is small, teachers are
  undisclosed, and Gemma's sandwich-norm is a real, only-partially-broken confound. The
  decisive test — pretrain matched models with and without distillation — is beyond our
  budget.
- **One language pair, one model** for the language finding, so far.

## What's next

A shared-probe recompute of the 397B band (the lens is public; the probe restriction is
cheap), the qwen3-32b lens-format gap, a *mechanistic* account of the family gate on
concept-resolution (what, physically, stops a concept entering Gemma's readout when the
geometry says it should?), the capacity tests proposed by Dehaene & Naccache, a
pretrain-matched KD-vs-from-scratch experiment (the decisive test our budget can't
reach), and the bridge to brain-alignment —
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
