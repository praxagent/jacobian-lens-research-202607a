# Replication proposal — *Features as Rewards* (Goodfire, RLFR)

**Paper:** Prasad, Watts, Merullo, Gala, Lewis, McGrath, Lubana. *Features as Rewards:
Scalable Supervision for Open-Ended Tasks via Interpretability.* Goodfire AI. arXiv
**2602.10067v3**, 18 Feb 2026.

**Status:** proposal only — nothing run, no compute authorized. Reviewer: TJ.

---

## TL;DR / recommendation

**Do not attempt the full end-to-end RLFR replication first — and possibly not at all.**
The headline (a policy 58% less likely to hallucinate) is produced by an RL run the
paper prices at **~$3,818 of their own compute for 300 steps** (and notes the
ground-truth-graded comparison would have cost **~$344,064**). That is a large spend
for a number that, on their own decomposition, is mostly *not* a weight change (see
"the 58% is three effects" below).

Instead, replicate the **load-bearing, cheap, falsifiable sub-claims** — which are the
actual scientific content — and add a novel extension that is squarely in our lane:

1. **Tier 1 (cheap): the detection probes are calibrated.** Reproduce Localize AUC ≈
   .88 / Classify AUC ≈ .94 on Gemma-3-12B-IT. Near-free because the expensive
   labels are largely **public** (LongFact++ / hallucination-probes, Neel Nanda's
   group). This is the core "features encode calibrated belief" claim.
2. **Tier 2 (moderate, NO RL): probe-reward beats the model judging itself.** Reproduce
   Fig 7 — best-of-N with the feature-reward pipeline outperforms a Gemma-self-judge
   by ~15 points at n=256, *with no training*. This is the paper's own strongest
   evidence for the thesis and it needs no RL.
3. **Our extension (novel, cheap): does an UNSUPERVISED readout work as the reward?**
   The paper trains supervised attention probes on grader labels. We ask whether a
   **logit lens / Jacobian lens** readout of the entity span gives a comparably
   calibrated hallucination signal *without any label training* — a direct bridge to
   our [workspace-under-pressure](../jacobian-lens-and-identifiability) result, where
   an internal readout already caught a model asserting "Kiev" while holding "Moscow".

**Tier 3 (the full RL run) is optional and gated on Tiers 1–2 holding + an explicit
TJ decision on the ~$4k+ spend.** If we never do Tier 3, Tiers 1–2 + the extension are
already a complete, honest, publishable replication-plus-contribution.

---

## What the paper claims

**Method — RLFR (Reinforcement Learning from Feature Rewards).** A four-stage pipeline
that turns probes-on-model-features into a dense reward for an open-ended behavior; the
paper instantiates it on **hallucination reduction**:

1. **Detect.** Two attention-based probes read the model's activations: a **Localization**
   probe (is this token part of an *Entity* = a falsifiable claim) and a **Classification**
   probe (is that Entity hallucinated). Trained to imitate a grader (Gemini 2.5 Pro with
   web search) on LongFact++ completions.
2. **Act (intervene).** On a localized span the policy takes an action —
   *maintain / retract / correct* — and samples an intervention (rollout).
3. **Reward.** Two more probes (**Retraction**, **Correction**) grade the intervention,
   read on the **base model's** activations. Final reward is a *multiplicative rubric*:
   legibility × substantiveness × reward-probe score (legibility/substantiveness judged
   by the base policy as an LLM).
4. **RL.** ScaleRL + CISPO, 360 steps, a Lagrangian constraint pinning the
   correct:retract ratio to 60:40.

**Model / data.** Policy = **Gemma-3-12B-IT** (π_base = pretrained Gemma-3-12B). Grader =
**Gemini 2.5 Pro + web search**. Data = **LongFact++** (~20k longform prompts).

**⚠️ What kind of "features"? NOT SAE features.** Despite the title and Goodfire's
SAE-centric platform, every probe here is a supervised **attention probe trained on
raw residual-stream activations** of π_base (App B: noncausal attention probes,
1024–2048 dim, 8–32 heads). "SAE"/"dictionary" appear **zero** times in the body;
"sparse autoencoder" appears only twice, both in the reference list. So "features" =
internal representations read by a trained probe — the Obeso/Nanda hallucination-probe
lineage — **not** sparse-autoencoder latents. This matters for our extension below:
the paper's own house instrument (SAE latents) is conspicuously absent, which is an
opening, not just a gap.

**Headline results.**
- **Probes are well-calibrated:** Localize **AUC .88**, Classify **AUC .94**, Retract
  **.88**, Correct **.88** (selected on AUC-ROC; calibration plots in Fig 3).
- **RLFR + best-of-32 → 58% fewer hallucinations** vs the original model (with the
  monitoring harness). ~90× cheaper per rewarded intervention than external evaluation.
- **Test-time only (no training):** feature-reward best-of-N beats a Gemma-self-judge by
  ~15 pts at n=256 (Fig 7).
- **Benchmarks preserved:** RLFR ≈ base across HellaSwag/PIQA/ARC/WinoGrande/BBH/MMLU/
  MATH/GSM8K/GPQA (Table 1; small movements, e.g. MMLU 69.2→67.2).
- **Activation-source invariance (Fig 9c):** reward-probe ordering is invariant to
  whether probes run on base vs policy activations → you can host just the policy at
  test time.

**⚠️ The 58% is three effects, only one of which is the trained model.** The paper's own
decomposition (Fig 4): **10%** from the policy becoming less hallucinatory (the actual
weight change), **35%** from inline interventions trickling through in-context (ICL at
test time), **13%** from interventions directly correcting hallucinations. Removing
best-of-N and inline interventions (RLFR-NI) drops it to **31%**, and the pure
policy-reduction with no test-time compute is **~10%**. Any honest replication *must*
report this decomposition, not just "58%."

---

## Load-bearing claims, tiered by cost

| # | Claim | Needs | Cost class |
|---|---|---|---|
| C1 | Detection probes read calibrated hallucination belief (AUC .88/.94) | activation cache + probe training; **labels largely public** | **cheap** |
| C2 | Reward probes (retract/correct) grade interventions (AUC .88) | more probes on intervention activations | cheap–moderate |
| C3 | Probe-reward best-of-N > self-judge at test time (Fig 7), **no RL** | base-model sampling + probe scoring + eval grader on a subset | **moderate** |
| C4 | Activation-source invariance (Fig 9c) | probes run on two activation sources | cheap (rides on C1/C2) |
| C5 | Full RLFR training → 58% reduction (decomposed) | **RL on 12B, 360 steps + eval grader** | **expensive** |
| C6 | Benchmark preservation (Table 1) | the trained policy + LM-eval-harness | moderate (needs C5) |

C1–C4 are the scientific core and are cheap. C5 is the headline and is expensive. C6
depends on C5.

---

## Proposed scope

### Tier 0 — CPU / free (validate feasibility before any spend)
- Confirm **LongFact++ labels + Obeso hallucination-probes artifacts** are downloadable
  and licensed for our use; confirm **Gemma-3-12B-IT** weight access.
- Re-implement the two-probe detection head (attention probe over an entity span) and
  smoke it on **gpt2 / a tiny slice** to prove shapes, the localize→classify staging,
  and the AUC readout — no GPU.
- Freeze the Tier-1 pre-registration (metrics, layers/sites to probe, AUC gates,
  train/test split) per house rules *before* caching any real activations.

### Tier 1 — reproduce probe calibration (cheap GPU)
- Cache **Gemma-3-12B-IT** activations over a few thousand LongFact++ completions on the
  cheapest GPU that fits 12B (single A100/H100, hours).
- Train the localize + classify attention probes on the public labels; **gate:**
  Localize AUC **≥ .85**, Classify AUC **≥ .90** on held-out (reproduce within
  tolerance). Report calibration (reliability curve), not just AUC.
- **Deliverable:** a receipt with per-token probe scores + labels + AUC/calibration, and
  a figure with the same-statistic controls (a random-direction probe, a
  bag-of-embeddings baseline) so we know the AUC isn't trivially achievable.

### Tier 2 — reproduce "probe-reward > self-judge" at test time (moderate GPU, no RL)
- On a few hundred LongFact++ prompts, sample N interventions from the base model,
  score them with (a) the feature-reward pipeline and (b) a Gemma-self-judge; run
  best-of-N and measure fixed/retracted success as a function of N (Fig 7).
- **Gate:** the probe pipeline's success rate exceeds the self-judge's at n ≥ 64, with
  the margin widening in N (reproduce the qualitative crossover; exact 15-pt margin is
  a stretch goal). Eval labels on a subset via a grader (cost-capped) or the public
  LongFact++ ground truth.

### Extension — which interpretability object carries the calibrated signal? (novel, cheap; our lane)
Replace the paper's *supervised dense attention probe* with two other readers on the
**same held-out entity spans** and compare AUROC / calibration:
- **Unsupervised lens** — logit lens and our **Jacobian lens** at the entity span,
  scoring "does the model's internal distribution support the asserted entity?" No
  label training at all. Direct bridge to our workspace-under-pressure result (a
  logit/J-lens readout caught a held truth the output contradicted).
- **Sparse SAE latent** — an off-the-shelf SAE "unsupported-claim / fabrication /
  uncertainty" latent (Neuronpedia / Gemma Scope), i.e. the *sparse* interpretability
  object Goodfire's platform is built on but this paper did **not** use. We already
  have SAE-reader infrastructure from the `llm_selfref_pre` series.
This turns the extension into a clean three-way question the paper leaves open: is the
calibrated hallucination signal a property of a **supervised dense probe** (theirs), an
**unsupervised transport** (ours), or a **sparse dictionary feature** (Goodfire's own,
unused here)? All three ride cheaply on the Tier-1 activation cache.

### Tier 3 — full RLFR training (expensive, OPTIONAL, gated)
- Only if C1–C3 hold and TJ explicitly authorizes the ~$4k+ spend. Would need RL on
  12B (ScaleRL/CISPO, 360 steps) + an eval grader, then the Fig-4 decomposition and
  Table-1 benchmark preservation. Report the **decomposed** reduction, never a bare 58%.

---

## Why this fits our campaign

- The paper's central object — *a probe reading a model's calibrated belief about the
  factuality of its own output* — is the same object as our workspace-under-pressure
  divergence ("Moscow held while the mouth says Kiev"). We already have lens/readout
  and receipt infrastructure pointed at exactly this.
- Our extension (unsupervised readout as reward) is a genuine contribution, not a
  re-run, and it is cheap because it rides on the Tier-1 activation cache.
- It broadens the campaign from *reading* hidden belief to *using* it as supervision —
  a natural next step, and one with an obvious safety framing (grounding training in
  interpretability signals).

## Risks / confounds to pre-register against
- **The 58% is mostly test-time compute + ICL, not weight change** (10% policy).
  Headline honesty requires the Fig-4 decomposition.
- **Grader reproducibility.** Gemini 2.5 Pro + web search is non-deterministic and
  costs; lean on public LongFact++ labels and cap any grader spend.
- **Seed variance** — the paper flags large between-seed variation on Not-Supported
  metrics; budget ≥3 seeds for any headline.
- **Probe AUC depends on layer/site choices** (their App B). Freeze these before caching.
- **Label leakage** — the LongFact++ labels were themselves produced by an LLM+search
  grader; "reproducing AUC .94 against LLM labels" measures agreement with a grader,
  not ground truth. Say so.

## Rough cost estimate
| Tier | Compute | ~Cost |
|---|---|---|
| 0 | CPU only | free |
| 1 | 12B activation cache (hours, 1 GPU) + CPU probe train | ~$10–40 |
| 2 | 12B best-of-N sampling on a few hundred prompts + probe scoring + capped grader | ~$50–150 |
| Extension | readout-only on the Tier-1 cache | ~$0–10 |
| 3 (optional) | RL on 12B, 360 steps + eval grader | **~$4k+** |

**Tiers 0–2 + extension: order ~$100–200 total.** That buys the entire scientific core
plus a novel result, and defers the one expensive piece behind an explicit decision.

## Open questions for TJ
1. Green-light Tiers 0–2 + the extension (~$100–200), holding Tier 3 for a separate
   decision?
2. Is the **unsupervised-readout extension** the framing you want to lead with, or a
   straight replication of their probes first?
3. Any contact with the Goodfire / Obeso authors before we publish a replication?
