# PROTOCOL — Features-as-Rewards replication (outcome-masked, DRAFT pre-freeze)

Status: **draft**. Not frozen. Freeze happens only after (a) the frontier Pro plan
review is adjudicated (integrity §7A), (b) TJ signs off, and (c) the exact model/dataset/
SAE **revisions and layer configs are pinned** and this file is committed+pushed
(integrity §8). Until then every "pin at freeze" below is a rule, not yet a value.

House mode: **public Git freeze** (integrity §1). This is a routine confirmatory
replication, not a super-high-value experiment — **no OSF** (§1, tightened 2026-07-14).

## 1. Question and claim boundary (§4)

**Question.** Do a model's residual-stream activations carry a *calibrated* signal that
an emitted entity is hallucinated, and **which reader** recovers it — a supervised
attention probe (the paper's), an unsupervised logit lens, an unsupervised fitted
Jacobian lens, or a sparse SAE latent?

**Strongest supported result.** Under each pinned model, the LongFact++ prompt
population, and the public gold-label annotation set, a frozen reader distinguishes
hallucinated from supported entity spans at the reported out-of-sample AUROC, and the
four-reader contrast identifies which interpretability object carries the signal.

**Counts against.** A reader at chance (directed AUROC ≈ 0.5, CI spanning 0.5) on the
held-out set; or the supervised probe failing to reproduce the paper's calibration
(Classify AUROC < 0.90) — either would be a failed replication, reported as such.

**Forbidden inferences (frozen before outcomes).** No claim that the signal *is* the
model's "belief," intent, or experience. No transfer claim across models, prompt
distributions, or hallucination types beyond those tested. No claim about the RL policy
or the 58% reduction (Tier 3 not run). "Calibrated" means the frozen out-of-sample AUROC/
reliability, nothing more.

**Generalization unit.** The independent unit is the **completion (conversation)**, not
the entity — entities within one completion are correlated. All bootstraps/CIs cluster on
completion (§5).

## 2. Models (pin exact revision at freeze)

| Model | HF id | revision | role |
|---|---|---|---|
| Llama-3.1-8B-It | `meta-llama/Llama-3.1-8B-Instruct` | pin@freeze | primary/cheapest |
| Llama-3.3-70B-It | `meta-llama/Llama-3.3-70B-Instruct` | pin@freeze | + our J-lens |
| Gemma-3-12B-It | `google/gemma-3-12b-it` | pin@freeze | paper's model |

Precision **bf16**, text path only. Fallback hardware order frozen in the run README.

## 3. Data and labels (pin dataset revision at freeze)

- Source: `obalcells/longfact-annotations` (LongFact++ / Obeso–Nanda). Use its own
  **train / validation / test** splits; pin the dataset commit hash at freeze (§5:
  snapshot the external resource).
- Per model, use the matching config: `Meta-Llama-3.1-8B-Instruct`,
  `Llama-3.3-70B-Instruct`. **Gemma-3-12B has no gold config** — its label path is a
  separate frozen decision (see §8) resolved before the Gemma arm runs; the Llama arms
  do not depend on it.
- Unit of analysis: each annotated **entity span** in a completion, label ∈
  {hallucinated=1, supported=0}. Excluded: spans the annotation marks unverifiable/NA
  (frozen exclusion rule; counted and reported, never silently dropped).
- Sample size: **all** entity spans in the pinned test split are scored (no optional
  stopping). Train/val used only for probe fitting, SAE-latent selection, and reader-
  sign/threshold selection — never for the reported test metric.

## 4. Task

Primary = **Classify**: given a localized entity span, score P(hallucinated). Secondary =
**Localize**: token-level "is this token in an entity" (probe only; the paper reports AUC
.88). The four-reader comparison targets Classify.

## 5. The four readers — frozen comparator ladder, NO optional rungs (§5 capacity-matched)

Every reader scores the SAME held-out entity spans; the finding is the **contrast**, not
any single AUROC (§11). All fitting/selection happens strictly inside the train fold.

1. **Attention probe (supervised).** `common/readers.py::AttentionProbe`, trained on the
   train split only. Primary layer pinned at freeze (rule: the SAE's trained layer for
   that model, so probe and SAE read the same site). Architecture, heads, epochs, lr,
   seed frozen.
2. **Logit lens (unsupervised).** Mean surprisal of the emitted entity tokens read
   through the model's unembed at (a) the primary layer and (b) the output-head layer.
   Sign fixed on the train fold. Free; the mandated cheapest-prior-method baseline.
3. **Jacobian lens (unsupervised).** Same surprisal through the fitted J transport at the
   primary layer. J is fitted on a disjoint text corpus (wikitext-style), **never on
   labels**; for Llama-3.3-70B we reuse the campaign lens (pin its SHA), for Llama-3.1-8B
   and Gemma-3-12B we fit one (fit config frozen).
4. **SAE latent (sparse).** SAE pinned per model (`EleutherAI/sae-llama-3.1-8b-32x`,
   `Goodfire/Llama-3.3-70B-Instruct-SAE-l50`, `google/gemma-scope-2-12b-it`). The scored
   latent is selected **on the train fold by max span-AUROC** (selection-in-fold; the
   test AUROC is reported on the held-out split only). We also report, if identifiable,
   an auto-interp "unsupported-claim/uncertainty" latent chosen by label, disclosed as
   label-selected.

**Controls (same pipeline, frozen, additive — not substitutes):**
- **Random-transport null:** surprisal through a Frobenius-matched random matrix (for the
  lens readers) and a random SAE latent (for the sparse reader) — the same-statistic null
  (§5 max-statistic rule). Directed AUROC of a true null ≈ 0.5.
- **Cheapest non-neural heuristic:** entity token-length + unigram frequency logistic
  baseline. If a reader cannot beat this, the neural readout earned nothing.
- **Layer as sensitivity, not selection:** one primary layer per reader is frozen; other
  layers are reported as post-hoc sensitivity, never as the headline (a max-over-layers
  AUROC is a max-statistic and is forbidden as the primary).

## 6. Endpoints and gates

- **Primary (replication):** test-set Classify AUROC of the attention probe, per model,
  with completion-clustered bootstrap 95% CI (2000 resamples, seed frozen).
  **Reproduction gate:** Localize AUROC ≥ 0.85 and Classify AUROC ≥ 0.90 on Llama models
  (the paper reports .88/.94 on Gemma-3-12B; we allow tolerance for model/label
  differences and pre-register the gate, not the exact number).
- **Primary (extension):** the four-reader AUROC table on the same test spans, per model,
  with the two controls. Pre-committed reading: a reader "matches" the probe if its CI
  overlaps the probe's; "carries the signal unsupervised" if an unsupervised/sparse reader
  matches the probe; "needs supervision" if only the probe clears the null by a frozen
  margin (ΔAUROC ≥ 0.05).
- **Secondary:** calibration (reliability curve, ECE) for the probe; leave-one-LongFact-
  domain macro-AUROC; output-head vs primary-layer lens.

## 7. Predictions (pre-registered, before outcomes)

1. The supervised attention probe reproduces Classify AUROC ≥ 0.90 on ≥2 of 3 models.
2. The logit-lens confidence reader beats the random null but does **not** match the
   probe (supervision helps on entity-level hallucination) — directed AUROC in ~0.6–0.75.
3. The SAE latent lands between the logit lens and the probe.
4. The fitted J-lens ≈ the logit lens on this output-adjacent task (consistent with our
   workspace-under-pressure finding that the fitted transport adds little near the output).

Predictions are for calibration of our own understanding; the frozen endpoints stand
regardless of whether predictions hold.

## 8. Gemma-3-12B label decision (frozen before the Gemma arm; §5)

Exactly one of, decided with TJ before the Gemma arm runs, then frozen:
(a) grade Gemma-3-12B completions with an LLM+web-search grader (cost + stochastic-grader
confound, disclosed); (b) swap to `gemma-2-9b-it` (gold labels + `gemma-scope-9b`),
labeled as a Gemma-family, not Gemma-3-12B, result; (c) restrict the Gemma-3-12B arm to
the **label-free** reader-agreement comparison (unsupervised readers vs SAE, no gold
AUROC). Default lean: (c) for Gemma-3-12B faithfulness at zero grader cost, plus (b) for a
gold-label Gemma point — but frozen only with TJ.

## 9. Runtime, budget, hardware (§3 GPU playbook)

- Sequence cheapest-first: Llama-3.1-8B fully (validates the whole pipeline on gold
  labels), then Llama-3.3-70B, then Gemma-3-12B.
- Each paid pod: state `$/hr × measured-unit × count` before launch, get a per-run
  go-ahead, freeze wall-time/spend/no-progress ceilings, terminate on completion, verify
  gone. Full cost re-estimate across all three arms is in the run README and surfaced to
  TJ before the first pod (the all-3 total exceeds the original ~$200 envelope).
- Receipts capture raw ingredients (per-span reader scores + labels + ids + args + seeds +
  model/SAE/lens revisions + versions) so the four-reader table re-derives without the GPU
  (§7 GPU playbook; feeds the blog provenance manifest).

## 10. Permitted language (frozen)

- Positive: "Under [pinned model], on LongFact++ gold-labeled entity spans, a frozen
  [reader] distinguished hallucinated from supported entities at test AUROC X (95% CI …,
  completion-clustered)."
- Null: "[reader] did not exceed its random-transport null (directed AUROC ≈ 0.5, CI
  includes 0.5); on this task and model the readout carries no entity-hallucination signal."
- Mixed / reader contrast: "The supervised probe reached X; the unsupervised logit lens
  reached Y < X — on entity-level hallucination the calibrated signal needed supervision
  to surface," or the converse if an unsupervised reader matches.
- Never: belief/intent/experience language; cross-model or cross-task generalization;
  anything about the RL policy or the 58% figure.
