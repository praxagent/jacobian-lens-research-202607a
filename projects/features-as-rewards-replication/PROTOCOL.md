# PROTOCOL v2 — reader benchmark for hallucinated-entity detection (outcome-masked, pre-freeze)

Supersedes the v1 draft after the frontier Pro review (`ADJUDICATION.md`, 10 blocking + 7
important findings, all accepted). **Not frozen.** Freeze requires: every B-row resolved
here; the control-gate suite (§7) + two-stage pipeline (§9) implemented and green on CPU;
per-arm manifests (§9) + the cost table (§10) filled with pinned revisions; and TJ sign-off
on the frozen plan **and** the whole-arm-set budget. House mode: **public Git freeze**, no
OSF (routine confirmatory benchmark).

## 1. What this is, and is not (B01, B02, B03)

A **benchmark of readers for hallucinated-entity detection** on labeled entity spans — it is
**not** a replication of *Features as Rewards*' Gemma-3 number (their exact completions/
grader/probe code are not public). We compare how well different readers of the same
residual-stream states **discriminate** hallucinated from supported entity spans.

- **Discrimination, not calibration.** Primary construct is out-of-sample **AUROC**. We do
  not call any raw score a probability. A separate frozen **calibration** sub-analysis
  (§8) applies to the supervised probe only.
- **Teacher-forced, observational.** We re-run pinned models over archived completions and
  read residual states; these are not necessarily the emitting model's original states, and
  we make **no causal/intervention claim** and no mental-state ("belief") claim.
- **Independent unit = the completion.** All uncertainty clusters on completion (§8).
- **Labels operationalize agreement with an annotation rubric, not ground truth.** The
  public arms use the Obeso LongFact++ rubric; the Gemma-3 arm uses our own grader (§11).

## 2. Arms (pin exact revision at freeze; B07)

**Gold-label four-reader arms** (public labels): `meta-llama/Llama-3.1-8B-Instruct`
(**primary confirmatory model**), `meta-llama/Llama-3.3-70B-Instruct`, `google/gemma-2-9b-it`.
**Own-graded labeled arm** (paper's exact model): `google/gemma-3-12b-it`, labels from our
cheaper grader (§11). Precision **bf16**, text path. Fallback hardware order in the execution
manifest. The arm set is **fixed at freeze**; no arm is added or dropped based on outcomes.

## 3. Data, splits, leakage bars (B09)

- Public labels: `obalcells/longfact-annotations`, pin the dataset commit hash. Use its
  `train / validation / test` splits, matched config per model.
- **Split roles (resolves the v1 contradiction):** **train** = fit probe parameters and the
  heuristic; **validation** = ALL choices (epoch, sign, SAE latent identity+sign, calibration
  map, thresholds, layer confirmation); **test** = ONE final evaluation, scored once.
- **Two-stage pipeline:** stage-1 *score generation* runs with **no access to test labels**
  and emits per-span scores keyed by frozen span IDs; stage-2 *locked evaluator* joins scores
  to labels by ID. `annotations`, `canary`, `verification_note`, and any label field are
  **barred from reader inputs**. Duplicate-completion-ID and cross-split leakage checks run
  before scoring; a hit stops the run.
- Sample = all label∈{Supported→0, Not Supported→1} spans in the pinned test split (no optional
  stopping). Excluded: unverifiable/NA labels; entity spans whose first token is position 0
  (no preceding state, B03) — both counted and reported.

## 4. Task and reader timing (B03)

Primary = **Classify**: score P̂(hallucinated) for a localized entity span. Every entity token
x_t is scored from the **immediately preceding causal residual** h_{t-1}; all readers consume
the **same** pre-entity states; readers may not see completion tokens after the span. The
frozen token-alignment table (per-arm manifest, §9) pins: prefix available at scoring time,
residual tensor + array index per token, pre-token convention, causal mask, span aggregation
(first/last/mean over span tokens — frozen rule), BOS/chat-template offsets, truncation.

## 5. Readers — frozen, taxonomy corrected (B05)

Scored on the SAME held-out spans; the finding is the **contrast** (§8).

- **Supervised:** attention probe (`common/readers.py::AttentionProbe`, source commit pinned),
  fit on **train**, all choices on **validation**, **3 fixed seeds**, primary = mean of the 3
  independently trained probes (never best-seed; B/I06).
- **Label-free:** (a) **logit lens** — pre-token surprisal via the model's unembed at the
  frozen primary layer; (b) **fitted Jacobian lens** — same surprisal through J (fit on a
  disjoint corpus, label-free; fit recipe + corpus + matrix hash pinned, LongFact excluded;
  I07). Sign fixed on **validation** by a mechanical formula.
- **Label-selected sparse:** **SAE latent** — dictionary fixed; the scored coordinate + sign
  selected on **validation** by max span-AUROC over the dictionary (this uses labels → *not*
  unsupervised; B05). Reported as label-selected.
- **Native output-token surprisal** (I02): the model's native next-token NLL of the entity
  (final norm + real output head). A reader **and** the §7 positive control — not "another lens".
- **Non-neural heuristic** (I04): logistic on entity token-length + unigram log-frequency;
  frequency corpus+revision, normalization, tokenizer, OOV smoothing, regularization all
  pinned; fit on train/val IDs only.

## 6. Frozen sites (no max-over-layers headline)

One **primary layer** per arm, pinned at freeze (rule: the SAE's trained residual site, so
probe/lens/SAE share a site; `resid_post`, layer index and hook named in the manifest — B04).
Other layers are **descriptive sensitivity only**.

## 7. Control-gate suite (B08) — distinguishes a scientific null from a pipeline bug

Frozen fixtures in `fixtures/`, all must pass before any arm's scores are trusted:
- **Alignment fixtures:** known text/spans → exact token indices + causal score positions.
- **Native-logit positive control:** the output-head lens reproduces the model's native token
  NLL within a frozen tolerance.
- **J-lens identity control:** identity transport == logit-lens implementation.
- **SAE compatibility:** dims + a known encode/decode fixture + firing/reconstruction diagnostics.
- **Leakage negative control:** a shuffled-label probe is at chance on held-out under the same
  fitting pipeline.
- **Procedure-matched SAE null:** random/permuted-latent controls undergo the **same candidate
  count + validation selection** as the label-selected SAE (a single random latent is not the
  right null).
- **Mechanical sign rule:** dev-only formula; never flip direction from test.
A failed gate ⇒ logged **outcome-masked implementation amendment** + rerun affected arms — never
a report/no-report choice.

## 8. Endpoints and statistics (B06, B02, I01, I03)

- **Primary (Llama-3.1-8B, confirmatory):** probe test AUROC + a family of **paired
  reader−probe AUROC differences** (logit lens, J-lens, SAE) under **completion-clustered
  paired bootstrap** (2000 resamples, frozen seed, shared resamples across readers, single-class
  resamples handled by a frozen rule). AUROC is **span-weighted**; CI is completion-clustered.
- **Equivalence margin (frozen):** a reader **matches** the probe iff its paired 95% CI ⊂
  [−0.05, +0.05]; **noninferior** iff lower bound > −0.05; **worse** iff upper bound < −0.05.
  "Above chance" is a separate one-sided CI rule vs 0.5, distinct from beating the heuristic.
- **Multiplicity:** **Holm** across the primary family; all other models/readers/layers are
  secondary/descriptive.
- **Precision calc (design stage):** from the pinned completion/class counts and a plausible
  within-completion correlation, confirm the expected CI can resolve the 0.05 margin; if not,
  the claim is **descriptive estimation**, not "matches".
- **Calibration sub-analysis (probe only, B02):** logistic map fit on validation; report test
  Brier, log-loss, ECE (frozen bins/binning/empty-bin rule). Never for raw lens/SAE scores.
- **Domain robustness (I03, descriptive):** macro-average of per-domain test AUROCs, min
  class-count rule, undefined domains dropped. Not confirmatory.
- **Reporting counts:** per split/model — completions, spans, supported, hallucinated, domains,
  clusters-with-each-class.

## 9. Execution + per-arm manifests (B04, B09)

- **Per-arm manifest** (`manifests/<arm>.json`): model+tokenizer revision, chat-template hash,
  SAE repo revision/file/width/L0/layer/hook/normalization/encoder-decoder convention, probe
  source commit + full architecture/optimizer/aggregation, logit-lens final-norm+unembed
  convention, J-lens source/target hooks + fit config + matrix hash, exact span aggregation.
- **Execution manifest:** immutable commit/tag + allowed output-branch lineage; clean-worktree
  check + exact commands; env lock (CUDA/PyTorch/Transformers, attention backend, dtype, device
  map, batching, deterministic flags, tolerances); the two-stage pipeline; failure handling
  (truncation/tokenizer-mismatch/NaN/OOM/classless/retry/hardware-fallback); reruns supplement,
  not replace; fixed seeds (train, bootstrap, controls, generation, J-fit). bf16 is not claimed
  bitwise-deterministic; tolerances are reported.
- Receipts store **per-span sufficient statistics** (scores per reader + label + ids + score_pos
  + args + seeds + revisions + versions), NOT raw 70B activations (B10) — re-analyzable without
  the GPU, feeds the blog provenance manifest.

## 10. Cost, storage, budget (B10) — one upfront approval for the whole arm set

A per-arm × per-stage table (completion/token counts, GPU type, measured-throughput assumption,
expected+max hours, $/hr, J-fit cost, extraction cost, grader cost, receipt volume, total) is
filled at freeze and approved as a whole (not outcome-contingent per model). Working estimate:
gold arms ~$50–120 + Gemma-3 own-graded arm ~$70–250 = **~$120–370 total**. Technical stop rules
only (hard spend, wall-time, no-progress); any unrun registered arm is reported. The campaign
Llama-3.3-70B J-lens is archived or shipped with a deterministic refit recipe (third-party
reproducibility).

## 11. Gemma-3-12B own-graded arm (cheaper grader; B01/B07, I05)

Generate Gemma-3-12B completions on a frozen LongFact prompt sample (source + generation params
+ seeds pinned), then annotate hallucinated entities with a **cheaper LLM+web-search grader**
(model+version pinned; the open Obeso annotation pipeline where possible), producing gold-ish
token-level labels → the full four-reader comparison + Gemma Scope 2 SAE. Stated explicitly as
**agreement with our grader's rubric, not ground truth**; the grader is a disclosed confound and
non-determinism source. This arm is **labeled** (has its own gold-ish AUROC), not a
score-agreement-only arm.

## 12. Predictions (pre-registered) and permitted language

Predictions: probe AUROC ≥ 0.85 on the primary model; logit lens beats chance but is worse than
the probe by > 0.05 (supervision helps on entity-level detection); SAE (label-selected) between;
J-lens ≈ logit lens (output-adjacent task). Predictions calibrate our understanding; endpoints
stand regardless.

Permitted language — positive: "Under [pinned model], on [rubric] labels, [reader] discriminated
hallucinated from supported entity spans at test AUROC X (95% CI …, completion-clustered,
span-weighted)." Null (I01): "did not detect discrimination above chance at the registered
precision (estimate, CI)"; "practically no signal" only after a passed equivalence test around
0.5. Contrast: "[reader] was worse than / matched / noninferior to the probe by the frozen
0.05 margin." Never: calibration/probability language for raw scores; belief/intent/experience;
cross-model/task generalization; RL-policy or 58% claims; "reproduced the paper's .88/.94".
