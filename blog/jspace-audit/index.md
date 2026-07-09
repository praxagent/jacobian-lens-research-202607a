---
title: "A Global Workspace, or a Training Artifact? Auditing Anthropic's J-Space Across 38 Open Models"
date: 2026-07-09
draft: true
authors: ["praxagent"]
tags: ["interpretability", "replication", "j-space", "global-workspace", "open-science"]
summary: "We replicated Anthropic's Jacobian-lens 'global workspace' result across 38 open-weight models. The measurement is real and remarkably stable — but the workspace band is far from universal: it emerges with scale, varies ~20× between model families at matched size, shrinks under instruct-tuning, and our best-supported explanation is a training-recipe effect: distillation-pretrained models barely form one."
---

<!-- DRAFT BANNER: numbers marked [PENDING-*] finalize when the uniform sweep
     completes (70B anchor, shared-vocab confound verdict, precision A/B, final
     figure). Everything else is receipted in the repo. -->

Last week Anthropic published [*Verbalizable Representations Form a Global Workspace in
Language Models*](https://transformer-circuits.pub/2026/workspace/index.html) — the
"J-space" paper. Using a new tool called the Jacobian lens, they found a small,
privileged set of directions inside Claude that behaves strikingly like the *global
workspace* that neuroscientists associate with conscious access in humans: its contents
are reportable, causally steer downstream reasoning, are reused flexibly across tasks,
and — when ablated — selectively destroy multi-step reasoning while leaving fluent
speech intact.

The paper set off a familiar cycle. The originators of Global Workspace Theory called it
["a landmark in consciousness research."](https://www-cdn.anthropic.com/files/4zrzovbb/website/cc4be2488d65e54a6ed06492f8968398ddc18ebe.pdf)
Critics called it PR wearing a lab coat, a triviality of backprop, or bad philosophy.
Almost nobody ran the experiment again.

We did. Anthropic released the [code](https://github.com/anthropics/jacobian-lens)
(Apache-2.0) and, with Neuronpedia, [pre-fitted lenses for 38 open-weight
models](https://huggingface.co/neuronpedia/jacobian-lens) — from Pythia-70M to
Llama-3.3-70B, across six model families. That makes the central structural claim
*testable* by anyone. So we tested it, with a simple question the framing invites:

**Is the "workspace" a universal property of language models — or a contingent one?**

## TL;DR

1. **The measurement is real, and more solid than we expected.** We re-fit lenses from
   scratch and reproduced Neuronpedia's to CKA 0.999; re-fitting on disjoint corpora
   changes essentially nothing (0.997–0.998). The J-space is a stable property of the
   *model*, not an artifact of how the lens is estimated.
2. **The workspace band is not universal.** It is absent below ~0.2B parameters, and at
   matched scale it varies by roughly an order of magnitude *between model families* —
   strong in Qwen, moderate in Llama/OLMo, near-zero in Gemma (0.0007 at 12B).
   [PENDING-SHARED: survives/attenuates under tokenizer-commensurable probes.]
3. **The best-supported explanation is a training-recipe effect.** In a natural
   experiment inside Gemma-2 — where Google's own report says the 2B and 9B were
   pretrained by knowledge distillation but the 27B from scratch — the distilled models
   score 0.002–0.007 regardless of scale while their from-scratch sibling scores ~6–20×
   higher, with architecture, normalization, and tokenizer held constant. We predicted
   the 9B value before looking (commit `0412769`), and it landed inside our interval.
4. **Post-training matters too:** instruct-tuned models show consistently *weaker*
   bands than their base versions, in every pair we measured.
5. **The transport is language-general deep, language-bound early:** a lens fit on
   Chinese Wikipedia agrees with an English-fit lens at 0.99 in the deepest layers but
   only ~0.73 in early ones.

Everything — code, per-model CSVs, fitted lenses, this post — is in
[our repo](https://github.com/praxagent/research-and-replications). Every number below
has a receipt.

## What the J-lens actually is (60 seconds)

For each layer *l*, the Jacobian lens is the **average causal Jacobian**
`J_l = E[∂h_final/∂h_l]` — over a text corpus, how much does nudging the residual
stream at layer *l* change what the model is about to say? Composing that with the
unembedding gives, for every vocabulary token, the internal direction that most raises
the model's disposition to *say* it. The **J-space** is the sparse set of these
directions that are strongly active — Anthropic shows its contents are reportable,
steerable, and causally load-bearing for multi-step reasoning in Claude.

Two properties matter for what follows. The lens is *estimated* (averaged over a
corpus), so one must check the estimate is stable before trusting anything built on it.
And it is *linear* and *vocabulary-indexed*, so it is a lower bound on structure and
sees only what tokens can name — limitations flagged by Anthropic's own invited
commentators, which shape our methods below.

## Our audit: strip it to the linear algebra, then scale it

We measured one structural signature across all 38 models: **the workspace band**. In
Claude — and in the [cross-model explorer](https://eliebak.com/viz/jspace-open) Elie
Bakouch built — the middle layers' J-lens token geometries are highly similar to *each
other* and distinct from early/late layers: a contiguous block in the layer×layer CKA
matrix. That block structure is the "workspace" in its most theory-neutral form: a
distinguished mid-network stage where the model's verbalizable geometry is unified.

Our statistic, `mid_sep`, measures how much more self-similar the middle third of
layers is than its neighbors. Three safeguards:

- **A null control.** Replace each layer's transport with a scale-matched random
  matrix: `mid_sep` collapses to ~0.00 at every scale (the shared unembedding alone
  produces a flat ~0.66 CKA floor, no band). Whatever the band is, it is not an
  artifact of the readout construction.
- **Estimation stability.** We fit our own lenses with Anthropic's code: our gpt2 lens
  matches Neuronpedia's at mean CKA 0.9992; three qwen3-4b lenses fit on *disjoint*
  wikitext samples agree pairwise at 0.9971–0.9981. The structures we measure are
  properties of models, not fitting noise.
- **The tokenizer confound.** Anthropic's invited commentators (Butlin et al., Eleos)
  point out the J-space is defined by the token vocabulary, so cross-family comparisons
  could be confounded by tokenizers. We therefore re-ran the entire comparison using
  only token strings shared by *all 38 tokenizers*. [PENDING-SHARED: verdict —
  the family gap survives / attenuates by X%.]

One methods note in the interest of full honesty: mid-run we discovered the 70B lens
could not fit in RAM under our original fp32 pipeline. The final numbers all come from
a single re-run under one code path that stores intermediates in fp16 (computation
stays fp32; the lenses themselves are distributed in fp16); an A/B against the original
path across ~34 models showed max |Δ mid_sep| = [PENDING-AB].

## Finding 1: The band emerges with scale — but scale is not the story

{{< figure src="emergence_curve.png" caption="Mid-band separation vs. parameters (log scale), colored by model family; gray = random-transport null floor. [PENDING-FIGURE: final version]" >}}

Below ~0.2B parameters (Pythia-70M, GPT-2), there is no band: `mid_sep` ≈ 0.015,
statistically at the null floor. By ~0.8B, bands appear — in some families. From there
the curve does not converge to a universal value; it **fans out by family**:

| Family (best size measured) | mid_sep | Pretraining |
|---|---|---|
| Qwen3-14B | 0.21 | from scratch |
| Qwen3.5-27B | 0.20 | from scratch |
| OLMo-3-32B | 0.094 | from scratch |
| Llama-3.1-8B | 0.087 | from scratch |
| gpt-oss-20b | 0.076 | "large-scale distillation" (mechanism undisclosed) |
| Gemma-2-27B | 0.043 | **from scratch** |
| Gemma-3-27B | 0.025 | distilled |
| Gemma-2-9B | **0.0019** | **distilled** |
| Gemma-3-12B | **0.0007** | distilled |

[PENDING-70B: the Llama-3.3-70B anchor — does the top of the open-weights range
confirm or break the family pattern?]

At 12–14B, the difference between Qwen and Gemma is roughly **300×**. Both are
residual-stream transformers trained with backprop. Whatever produces the workspace
band, it is not "being a large language model."

## Finding 2: The mechanism — distillation suppresses the band

The family table hides a natural experiment. Google's Gemma-2 report states that the
2B and 9B models were pretrained by **knowledge distillation** (learning to match a
teacher's token distributions) while the **27B was trained from scratch**. Same
architecture, same sandwich normalization, same tokenizer, same data lineage:

| Gemma-2 | Pretraining | mid_sep |
|---|---|---|
| 2B | distilled | 0.007 |
| 9B | distilled | 0.0019 |
| 27B | from scratch | 0.043 |

The two distilled models sit at the floor *regardless of scale* — the 9B scores
*below* the 2B — while the from-scratch sibling scores ~6–20× higher. We wrote the
prediction (9B ≤ 0.02) into git before looking at the value; it came in at 0.0019.
(Honesty note: the sweep had computed the number minutes before we looked — this is
predicted-before-seeing, not formal pre-registration.)

The mechanism is plausible and has prior art: distillation students demonstrably
[reorganize internal computation](https://arxiv.org/abs/2505.10822) rather than inherit
the teacher's, and KL-only distillation [measurably *reduces* layerwise-CKA
structure](https://arxiv.org/abs/2606.05682). Intuitively: a student trained to mimic a
teacher's output distribution doesn't need to *develop its own* internal serial
workspace — the teacher already did that thinking.

We rank this a strong hypothesis, not a law. Gemma-3-4B (distilled) reaches 0.046,
overlapping the bottom of the from-scratch range; Gemma uniquely combines distillation
with sandwich-normalization (a residual confound the within-family experiment only
partially breaks); and gpt-oss's "distillation" is too vaguely described to score. The
alternative hypotheses, the per-layer attention maps, and the falsification tests are
all in [`hypotheses.md`](https://github.com/praxagent/research-and-replications/blob/main/projects/jacobian-lens-and-identifiability/hypotheses.md).

One hypothesis we can *rule out*: attention architecture. Qwen3.5 runs **75% linear-
attention layers** and posts the strongest bands in the cohort; Gemma's sliding windows
don't even bind at the 128-token contexts the lenses are fit on. The band does not care
how attention is wired; it seems to care how the model was *taught*.

## Finding 3: What post-training and language do to it

**Instruct-tuning shrinks the band, every time we can measure it.** gemma-3-4b:
0.046 → 0.015. gemma-2-2b: 0.007 → 0.001. llama-3.1-8b: 0.087 → 0.079. This is
geometry, not content — but it is at least in tension with a casual reading of
"post-training installs the Assistant into the workspace." (Anthropic's own §6.1/§9.3
results show post-training changes what the workspace *represents*; our result adds
that its geometric distinctness *decreases*.)

**The transport is language-bound early, language-general deep.** We fit a qwen3-4b
lens on Chinese Wikipedia and compared it to the English-fit lens: agreement is ~0.73
in the earliest layers, rising monotonically to 0.99 at the top — against an
English-vs-English baseline of 0.997 at every layer. The English-anchoring that
[Wendler et al.](https://arxiv.org/abs/2402.10588) describe lives in the early
verbalization layers; the deep, workspace-adjacent geometry is close to
language-general. This *refines* Anthropic's multilingual claim rather than refuting
it — and states precisely where it breaks.

## So: was "global workspace" oversold?

Here is the calibrated version, claim by claim.

**What survives — and it's substantial.** The core measurement is excellent and
replicates in independent hands: ours across 38 models, and DeepMind's Neel Nanda on
Qwen (who found the bands "notably less clean than the paper's" — consistent with our
family-dependence). The J-space is a stable, causally meaningful structure. Anthropic
also *hedged the consciousness claim themselves*: their thread explicitly claims a
mechanism for **access consciousness**, not phenomenal experience. The "it's all PR"
critique doesn't survive contact with the fact that the theory's own originators
engaged it as serious, testable science.

**What doesn't survive: the implied universality.** "Language models develop a global
workspace" reads as a claim about language models. Our data says it is a claim about
*some training recipes*. The band is absent in small models, near-absent in an entire
major model family, strongest in another, reduced by instruct-tuning, and — in the
cleanest test we could construct — suppressed by distillation pretraining. The
workspace is **contingent, not constitutive**.

**And the deflationary critiques fail symmetrically.** "It's trivially expected from
backprop and residual streams" (Trask) and "every layered system has internals that
don't surface — should my .docx get a workspace too?" (Barenholtz) both predict the
band should be *everywhere*. It isn't. Gemma-3-12B is a backprop-trained residual
transformer, and its band is 0.0007. **The hype and the anti-hype make the same
mistake: treating the phenomenon as necessary — either as consciousness or as
triviality — when it is contingent.** Contingent structure is the scientifically
interesting kind: it demands an explanation (we've proposed and part-tested one)
rather than a label.

On the philosophy, we defer to the experts on both flanks and take no position on
phenomenal consciousness. The Eleos commentary's tiering is the rigorous frame: the
evidence firmly establishes a *privileged set* of accessible representations, only
suggestively a unified *stream*, and not the full GWT *workspace*. Barenholtz's wedge
cuts deepest: an LLM producing "linguistic report, workspace and all, without sensory
grounding" arguably *strengthens* the dissociation between reportability and
experience. The better the workspace result, the weaker the consciousness inference.

## Limitations, up front

- **We measure geometry, not behavior.** `mid_sep` is our statistic — a CKA
  block-structure measure on the J-lens geometry. It is *not* Anthropic's causal
  battery (swaps, ablations, reportability). A family could conceivably pass causal
  workspace tests while scoring low on our band statistic (our own null and
  homogeneity checks partially, not fully, close this gap). Running Anthropic's
  released behavioral experiments per-family is the necessary next step.
- **Sub-frontier scope.** Our ceiling is [PENDING-70B: 70B]. Everything here could in
  principle be a sub-frontier transient — the exact overclaim-from-limited-scale error
  we're auditing. Nanda's scaling data (a ~400B MoE lens fit in ~1 hour on 8×H200s)
  makes the frontier check affordable; it's our top follow-up.
- **The tokenizer confound** is [PENDING-SHARED: handled — verdict here].
- **Distillation hypothesis is observational.** N(families) is small, teachers are
  undisclosed, and Gemma's norm scheme remains a partial confound. The decisive
  experiment — pretrain matched models with and without KD — is beyond our budget, but
  our per-layer and homogeneity-robust tests are running.
- **One language pair, one model** for the language finding, so far.

## What's next

Frontier-scale lens fits (the transient question), Anthropic's behavioral battery
per-family (does Gemma *behave* workspace-less, or just measure that way?), the
ignition and capacity tests proposed by Dehaene & Naccache, and the bridge to the
brain-alignment literature — does our band coincide with the layers that predict human
brain activity? The [full project docs](https://github.com/praxagent/research-and-replications/tree/main/projects/jacobian-lens-and-identifiability)
track all of it.

## Credit

This audit exists because Anthropic released working code and, with Neuronpedia,
pre-fitted lenses — open science done right, and we did our part on top of it. The
"middle layers are special" lineage long predates J-space: Schrimpf, Caucheteux & King,
Jain & Huth, Goldstein, Toneva and the brain–LLM alignment community, whose priority
Jean-Rémi King has [rightly emphasized](https://x.com/jeanremiking/status/2074500550947680368).
The identifiability theory that frames when any of this is trustworthy is due to Zheng,
Zhang and colleagues. Commentary that sharpened this audit: Dehaene & Naccache, the
Eleos team, Neel Nanda, and the critics — Ravid, Trask, Barenholtz — whose objections
we did our best to test rather than dismiss.

*All code, data, and fitted lenses: [github.com/praxagent/research-and-replications](https://github.com/praxagent/research-and-replications). Corrections welcome — that's what the receipts are for.*
