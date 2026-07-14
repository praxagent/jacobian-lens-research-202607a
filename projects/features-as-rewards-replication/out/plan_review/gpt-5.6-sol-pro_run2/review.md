# Verdict

The plan has several unusually strong foundations: it restricts mental-state and RL claims; recognizes the completion rather than the entity as the independent sampling unit; proposes a genuine held-out test split; intends to score readers on the same spans; forbids headline layer selection; and plans to retain raw per-span receipts. Those choices should be preserved.

It is nevertheless not yet a freezeable replication. The central issue is not merely missing revision hashes. As written, the study alternates among three different targets:

1. replication of the *Features as Rewards* Gemma-3-12B probe result;
2. benchmarking hallucination readers on public LongFact++ annotations from a different cited project and mostly different models; and
3. a label-free agreement study on Gemma-3-12B.

Only the second is currently supported by the supplied artifacts, and even that target lacks the temporal alignment, hook definitions, calibration procedure, comparator semantics, statistical contrast, controls, and execution manifest needed to make the result unambiguous. In particular, AUROC is incorrectly treated as calibration; a gold-label-selected SAE latent is described as unsupervised; CI overlap is used as evidence of matching; and the position from which each reader scores an already-emitted entity is not defined. These defects could support mutually incompatible interpretations after outcomes are seen.

The smallest defensible repair is to reframe the frozen core as a held-out **discrimination benchmark of specified readers under teacher forcing on archived LongFact++ conversations**, make one public-label model the confirmatory primary, explicitly classify the max-AUROC SAE as label-supervised feature selection, define exact causal token alignment and hooks, and replace CI-overlap language with paired completion-clustered contrasts using a frozen noninferiority/equivalence margin. A true paper replication would instead require the paper’s exact model, outputs/labels or equivalent annotation process, probe implementation, activation site, and evaluation protocol; that evidence is not present in the packet.

NOT READY TO FREEZE

# Blocking findings

## B01 — The claimed replication is not comparable to the cited experiment

- **Severity:** Blocking — definite claim/design mismatch, plus missing evidence about the prior implementation.
- **Plan section or excerpt:** Title: “Features-as-Rewards replication”; §6: “the paper reports .88/.94 on Gemma-3-12B”; §8: Gemma-3-12B has no gold labels and is run only for label-free agreement. The bounded background says the cited paper used probes trained to imitate a Gemini-2.5-Pro-with-web-search grader on Gemma-3-12B-IT, while this plan uses public annotations from the separate LongFact++/Obeso–Nanda resource, primarily on Llama models and gemma-2-9b-it.
- **Why it matters:** A result on different models, outputs, label provenance, and possibly a different probe implementation cannot reproduce the cited Gemma-3-12B AUROCs. The proposed thresholds of 0.85/0.90 on Llama models are tolerances chosen for a different setting, not a replication criterion. The packet also does not include the cited paper, probe code, exact paper dataset, label rubric, or hook specification, so exact comparability has not been demonstrated. Calling the study a replication would overstate either a positive or a negative outcome. A failure on Llama/public labels would not falsify the paper’s Gemma/grader result.
- **Concrete minimum fix:** Choose and freeze one of two claims:
  1. **Minimal and feasible:** Rename the study a “LongFact++ held-out reader benchmark motivated by *Features as Rewards*.” Remove the paper-reproduction gate and present the prior .88/.94 only as external context, not a numerical success criterion.
  2. **True replication:** Obtain and pin the exact Gemma-3-12B outputs, labels or grader protocol, probe code/architecture, chat template, activation location, split construction, and metric implementation used for the cited result. Run that exact arm as the replication; treat Llama and SAE/J-lens comparisons as extensions.
  
  Also state that the labels operationalize agreement with the public annotation rubric, not factual truth without annotation error.
- **Claim affected:** “Routine confirmatory replication,” “failed replication,” reproduction gate, and any comparison to the cited .88/.94.

## B02 — “Calibrated” is conflated with discrimination

- **Severity:** Blocking — definite construct-validity defect.
- **Plan section or excerpt:** §1: “‘Calibrated’ means the frozen out-of-sample AUROC/reliability”; §4: “score P(hallucinated)”; §6 makes AUROC primary and reliability/ECE secondary; §1 says a reader at chance AUROC counts against calibrated signal.
- **Why it matters:** AUROC measures rank discrimination and is invariant to monotone transformations. It cannot show that a score is a calibrated probability. A raw surprisal, SAE activation, or signed lens score is not \(P(\text{hallucinated})\). A reader can have high AUROC and arbitrarily bad calibration, or be well calibrated at the base rate while having little discrimination. “AUROC/reliability” also leaves open which outcome will justify the word after results are known.
- **Concrete minimum fix:** Either:
  - replace “calibrated signal” with “discriminative signal” throughout and report raw-score AUROC as the primary construct; or
  - define a frozen calibration procedure using development data only, such as a logistic calibration map fit on validation, and evaluate test log loss, Brier score, calibration intercept/slope, and a fully specified ECE/reliability calculation. Keep AUROC as a separate discrimination endpoint. Pin ECE bins, binning method, treatment of empty bins, and whether calibration is assessed only for the probe or for every reader.
  
  Do not call a raw lens or SAE score a probability unless a mapping is actually fit and tested.
- **Claim affected:** Every use of “calibrated,” “P(hallucinated),” and the background assertion that probe calibration is being replicated.

## B03 — Reader timing is undefined and may leak the entity or future text

- **Severity:** Blocking — missing specification with a high risk of a definite temporal construct error.
- **Plan section or excerpt:** §5: “Mean surprisal of the emitted entity tokens read through the model’s unembed”; §1: “activations carry a signal that an emitted entity is hallucinated”; no definition is given for whether residual state \(h_t\) scores token \(x_t\) or \(x_{t+1}\), whether the probe sees the full answer, or how cached states are produced.
- **Why it matters:** In a causal LM, the residual at the position containing token \(x_t\) normally predicts \(x_{t+1}\); the state used to score \(x_t\) is at the preceding causal position. Using a post-token state to “predict” that same token leaks its identity through the residual stream. Allowing an attention reader to access later answer tokens, retractions, corrections, citations, or the annotation span boundary would turn an online self-monitoring claim into retrospective text classification. Conversely, pre-token confidence and post-entity factuality are different constructs. Reusing KV caches across conversations or failing to reset text state can create cross-example carryover. Finally, if archived completions were generated by an unknown model revision or template, teacher-forced activations from a newly pinned revision are not necessarily the states of the model that emitted the text.
- **Concrete minimum fix:** Add a token-level alignment table that freezes, for every reader:
  - the exact prefix available at scoring time;
  - the residual tensor and array index used to score each token;
  - whether scoring is pre-token, post-token, or post-span;
  - the causal attention mask;
  - span aggregation, including first/last token and punctuation treatment;
  - BOS/chat-template offsets, padding, truncation, and multimodal/text-path tokens;
  - whether future completion tokens are prohibited;
  - KV-cache reset and batching behavior.
  
  The smallest common design is to score every entity token from the **immediately preceding causal residual state**, aggregate those token scores by one frozen rule, and give all readers only those same states. If the paper used post-entity states, that should instead be reproduced explicitly and the lens score redefined accordingly. Reconstruct each archived transcript from frozen text and chat templates; verify token identity round trips. If the dataset does not identify the generating model revision/template, describe the outcome as a teacher-forced readout of the pinned model, not the emitter’s original internal state. State explicitly that the study is observational and makes no causal intervention claim.
- **Claim affected:** “A model’s residual-stream activations carry a signal,” online/self-monitoring interpretations, reader comparability, and any connection to a deployable reward.

## B04 — Hook, layer, tokenizer, and reader semantics are not freezeable from the plan

- **Severity:** Blocking — missing implementation evidence and undefined choices.
- **Plan section or excerpt:** §5 ties the primary layer to “the SAE’s trained layer” but does not pin exact SAE releases/configs or hook names; `common/readers.py::AttentionProbe` is referenced but its contents are not supplied; “output-head layer,” “same surprisal through the fitted J transport,” and SAE span scores are not defined.
- **Why it matters:** `resid_pre`, `resid_mid`, `resid_post`, attention output, and MLP output are not interchangeable. Layer numbering can be zero- or one-based. SAEs may expect a particular residual site, normalization, model revision, tokenizer, context length, dtype, and activation scaling. A Jacobian transport needs an exact source hook, target, centering/intercept convention, normalization, vocabulary/unembed convention, and token alignment. “Output-head layer” could mean final block residual before final norm, after final norm, or native logits. The SAE score could be mean, maximum, final-token activation, firing indicator, or reconstruction-derived quantity. Any of these choices can materially change AUROC.
- **Concrete minimum fix:** Before freeze, create a machine-readable per-arm manifest containing:
  - model and tokenizer revision;
  - chat template hash;
  - SAE repository revision, exact SAE file/config, width/L0, layer index, hook name, expected model, normalization, and decoder/encoder conventions;
  - probe source-code commit, complete architecture, inputs, masks, initialization, optimizer, epochs, batch construction, class weighting, early-stopping rule, and entity aggregation;
  - logit-lens source state, final normalization and unembed convention;
  - J-lens source/target hooks, matrix/intercept hashes, fit corpus/config, and scoring equation;
  - exact span-to-token alignment and score aggregation for all readers.
  
  Add small fixture tests showing that all readers receive the intended same token states, that native output-head logits are reproduced to a frozen tolerance, and that each SAE can encode the pinned activation tensor with expected dimensions and reconstruction statistics. The referenced code and run README must be included in the freeze artifact or pinned by immutable commit; they were not supplied here and should not be treated as reviewed.
- **Claim affected:** The four-reader comparison, “same site,” SAE/Jacobian-lens compatibility, reproducibility, and paper comparability.

## B05 — The SAE reader is label-supervised, and reader performance cannot locate where the signal “resides”

- **Severity:** Blocking — definite supervision-labeling error and overinterpretation.
- **Plan section or excerpt:** §5: “The scored latent is selected on the train fold by max span-AUROC”; §6: an “unsupervised/sparse reader” can show that the signal “carries” unsupervised; background: matching would show that the calibrated signal is “a property of the features, not of the label-trained probe”; §5 calls the ladder “capacity-matched.”
- **Why it matters:** Selecting one SAE latent by maximum gold-label AUROC is supervised feature selection, even though the SAE dictionary itself was trained without hallucination labels. Searching a large latent dictionary can use substantial label information. It therefore cannot support the conclusion that an unsupervised reader recovered the signal. A single scalar surprisal, a label-selected SAE coordinate, and a trainable attention probe are also not capacity-matched. Failure of these particular unsupervised formulas cannot show that the representation “needs supervision”; success does not establish that the signal resides in one interpretability object rather than being exposed by a particular readout.
- **Concrete minimum fix:** Make the following distinction explicit:
  - **Label-free readers:** logit lens, J-lens, and only an SAE latent selected without hallucination labels under a completely frozen semantic or external criterion.
  - **Label-selected sparse reader:** SAE dictionary fixed, coordinate and sign selected using development labels.
  
  If no defensible label-free latent-selection rule is available, retain only the label-selected SAE arm and remove it from all “unsupervised” conclusions. Drop “capacity-matched,” “identifies which object carries the signal,” and global “needs supervision” language. The valid claim is narrower: “Among these specified scoring rules, reader A had higher/lower held-out discrimination than reader B.” A negative result may say that the tested label-free readers were not noninferior, not that no unsupervised readout exists.
- **Claim affected:** The central four-reader interpretation, SAE prediction, “property of the features,” and “needs supervision.”

## B06 — The statistical decision rules do not establish matching, and the design lacks a frozen precision/multiplicity plan

- **Severity:** Blocking — definite inferential defect plus missing evidence about adequacy.
- **Plan section or excerpt:** §6: a reader “matches” the probe if its CI overlaps the probe’s; “needs supervision” if only the probe clears the null by a margin; §3 uses all test spans but gives no completion/class counts or power calculation; there are multiple readers, models, layers, controls, and secondary endpoints.
- **Why it matters:** Overlap of two marginal confidence intervals is neither a paired test nor evidence of equivalence. Wide intervals almost guarantee overlap and can make an underpowered study declare a match. The readers score the same spans, so the relevant quantity is the paired AUROC difference under completion-cluster resampling. “Clears the null by ΔAUROC ≥ 0.05” is ambiguous about whether it means AUROC ≥ 0.55, a difference from a random control, or a confidence-bound criterion. No primary model or family of contrasts is identified, and no multiplicity rule is frozen. “Use all available test spans” prevents optional stopping but does not demonstrate enough independent completions or hallucinated entities to resolve the proposed 0.05 margin.
- **Concrete minimum fix:** Before outcomes:
  1. Report per split/model the number of completions, spans, supported spans, hallucinated spans, domains, and clusters with each class.
  2. Choose one confirmatory primary model and a small primary family, for example probe AUROC plus three paired reader-minus-probe contrasts.
  3. Define “match” using a paired completion-clustered noninferiority or equivalence interval with a frozen margin. If the intended tolerance is 0.05, specify whether matching means the entire paired CI lies within \([-0.05, 0.05]\) or merely that the reader is no worse than the probe by more than 0.05.
  4. Define “signal above chance” as a confidence-bound or adjusted-test rule, separately from comparison with the heuristic.
  5. Freeze multiplicity handling, such as Holm adjustment across the primary reader contrasts, or designate all but one contrast secondary.
  6. Conduct a design-stage precision calculation using the known cluster counts and plausible within-completion correlation. If the expected CI cannot resolve the frozen margin, change the claim to descriptive estimation rather than “matches.”
  7. State whether AUROC is span-weighted or completion-weighted. Completion-cluster bootstrapping fixes uncertainty dependence but does not change the estimand’s weighting.
  8. Specify how bootstrap resamples containing only one class are handled and use paired resamples shared across readers.
- **Claim affected:** “Matches the probe,” “needs supervision,” “beats the null,” the four-reader finding, and any ≥2-of-3-model conclusion.

## B07 — The Gemma arms are internally inconsistent and incompletely specified

- **Severity:** Blocking — definite contradictions plus missing artifacts.
- **Plan section or excerpt:** §2 says “The four gold-AUROC arms are” followed by three models; §8 adds a full four-reader gemma-2-9b-it arm; §5 lists J-lens fitting only for Llama-3.1-8B and Gemma-3-12B and does not define a gemma-2-9b J-lens; §9 sequences “all three arms” despite listing four models. Gemma-3-12B is to use “the same entity spans,” but no Gemma-3 conversations, prompts, or entity-localization procedure are specified.
- **Why it matters:** A label-free agreement analysis still requires a defined text population and entity spans. The public resource reportedly has no Gemma-3-12B config, so there is no supplied source for those spans. Generating new completions and running a NER/localization procedure would create an additional sampling and selection design that is absent. The full gemma-2-9b arm requires an exact model revision, compatible SAE, primary hook, probe configuration, and J-lens plan. The feasibility artifact reports a label config and mentions a Gemma Scope SAE for gemma-2-9b, but it does not document model download/access or a J-lens for that arm. These omissions make both the arm count and the budget indeterminate.
- **Concrete minimum fix:** The smallest repair is to remove Gemma-3-12B agreement from the confirmatory protocol and place it in a separate exploratory amendment. Keep gemma-2-9b only after its exact model/SAE/J-lens/hook artifacts and cost are verified and pinned. If Gemma-3 agreement is retained, freeze:
  - prompt source and sampling;
  - generation parameters and seeds;
  - entity extraction/localization method;
  - which entities are included;
  - score direction and agreement metrics;
  - the inferential boundary that agreement is not factual validity.
  
  Correct the model/arm counts and sequence everywhere.
- **Claim affected:** Exact-model paper connection, “same entity spans,” per-model four-reader table, predictions across three models, feasibility, and budget.

## B08 — Controls and gates are not procedure-matched or sufficient to detect implementation failures

- **Severity:** Blocking — incomplete falsification logic.
- **Plan section or excerpt:** §5 proposes one Frobenius-matched random transport and “a random SAE latent”; signs are fixed on train; §6 refers to clearing a null; no positive-control or pipeline manipulation gate is specified.
- **Why it matters:** A single random latent is not the same-statistic null for an SAE coordinate chosen by maximizing AUROC over a large dictionary. A random transport can also differ from the fitted transport in rank, spectrum, centering, and output scale despite matching the Frobenius norm. Directional AUROC can be silently test-flipped unless the sign rule is completely mechanical. Most importantly, chance-like results could arise from an off-by-one token shift, wrong hook, failed J transport, incorrect final normalization, broken SAE loading, or span alignment error. Conversely, a positive result could arise from leakage. The current controls do not discriminate scientific nulls from pipeline failures.
- **Concrete minimum fix:** Freeze a compact gate suite:
  - **Alignment fixtures:** known text/spans with exact token indices and causal score positions.
  - **Native-logit positive control:** the designated output-head lens must reproduce the model’s native token logits or token NLL within a frozen numerical tolerance.
  - **J-lens identity control:** an identity transport at the same source/target convention must reproduce the logit-lens implementation.
  - **SAE compatibility control:** dimensionality, expected firing/reconstruction diagnostics, and a known encode/decode fixture.
  - **Leakage negative control:** a shuffled-label probe must be at chance on held-out labels under the same fitting pipeline.
  - **Procedure-matched SAE null:** random or permuted-latent controls must undergo the same candidate count, sign choice, and development-set selection procedure as the reported label-selected SAE.
  - **Sign rule:** write the exact development-only formula; never flip direction from test results.
  
  State what happens if a gate fails: repair under a logged, outcome-blind implementation amendment and rerun all affected arms, rather than choosing whether to report based on the scientific scores. No causal manipulation is necessary for this observational readout question, but the protocol must explicitly decline causal claims.
- **Claim affected:** Null interpretation, SAE evidence, sign-directed AUROCs, and whether a failure is scientific rather than an implementation error.

## B09 — Split roles, leakage prevention, deterministic execution, and failure handling are not frozen

- **Severity:** Blocking — definite contradiction and missing execution controls.
- **Plan section or excerpt:** §3 says “Train/val used only for probe fitting, SAE-latent selection, and reader-sign/threshold selection”; §5 says “all fitting/selection happens strictly inside the train fold.” Architecture, seeds, revisions, hardware order, and budget are repeatedly described as “pin at freeze,” but no values or complete run manifest are included.
- **Why it matters:** Train/validation role ambiguity allows hyperparameters, latent identity, sign, calibration, or thresholds to move between splits after development results are visible. The public test labels are accessible, so a procedural barrier is needed even if intentional leakage is not expected. BF16 execution across different GPU types, kernels, attention implementations, batching, and software versions is not bitwise deterministic merely because a seed is set. The protocol does not define treatment of tokenizer mismatches, truncation, missing hooks, OOMs, NaNs, classless completions, failed pods, retries, partially cached conversations, or reruns with changed hardware. These choices can change the analyzed sample or permit outcome-dependent execution.
- **Concrete minimum fix:** Freeze one execution manifest that specifies:
  - immutable repository commit/tag and allowed output branch lineage;
  - clean-worktree check and exact command lines;
  - container/environment lock, CUDA/PyTorch/Transformers versions, attention backend, dtype, device map, batching, deterministic flags, and acceptable numerical tolerances;
  - exact split roles, e.g. train for parameter fitting, validation for all model/epoch/sign/latent/calibration choices, and test for one final evaluation;
  - a two-stage pipeline in which score generation cannot read test labels, followed by a locked evaluator that joins scores to labels by frozen IDs;
  - checks for duplicate completion IDs or leakage across splits and a rule that `annotations`, `canary`, verification fields, and test labels cannot enter reader inputs;
  - common-analysis-set rules and all exclusions;
  - handling of truncation, tokenization failures, nonfinite scores, missing SAE/J outputs, OOMs, interrupted runs, retry count, and hardware fallback;
  - whether reruns replace or supplement prior receipts and how discrepancies are reconciled;
  - fixed seeds for training, bootstrap, random controls, generation where applicable, and J-lens fitting.
  
  Resolve the train/validation contradiction in the protocol text before freeze.
- **Claim affected:** Out-of-sample status, reproducibility, common-span comparability, deterministic execution, and the validity of all reported CIs.

## B10 — Cost, storage, artifact availability, and unconditional execution are not established

- **Severity:** Blocking — missing feasibility evidence for an expensive multi-model run.
- **Plan section or excerpt:** §9 says the total exceeds the original ~$200 envelope, but the estimate is only in an unsupplied run README; each pod requires a later go-ahead. The Llama-3.3 J-lens is described as a campaign asset to be pinned later. Exact model/dataset/SAE revisions remain unset.
- **Why it matters:** The supplied packet does not give completion/token counts, GPU-hour estimates, hardware rates, J-lens fitting cost, SAE/probe extraction cost, storage requirements, or the actual total. “Per-run go-ahead” after cheaper-arm outcomes are known can turn the later model set into an outcome-dependent sample. The existing J-lens may not be publicly downloadable or regenerable; the packet only says it is owned by the campaign. Gated models also limit third-party reproduction unless access requirements and immutable derived artifacts are documented. Retaining full 70B residual activations can be very expensive in storage, while streaming reductions require a different implementation that must be frozen.
- **Concrete minimum fix:** Before freeze:
  - include the actual cost/storage table by arm and stage, with token counts, GPU type, measured throughput assumption, expected and maximum hours, hourly price, J-fit cost, extraction cost, and receipt/storage volume;
  - secure approval for the entire frozen arm set, not outcome-contingent per-model approval;
  - define only technical stop rules such as hard spend, wall-time, and no-progress limits, with mandatory reporting of any unrun registered arm;
  - pin every model/dataset/SAE/lens revision and archive the J-lens matrix/config where licensing permits;
  - state third-party access requirements and provide a deterministic lens-refit recipe if the campaign asset cannot be distributed;
  - stream or retain only the frozen per-span sufficient statistics unless raw activations are scientifically required.
- **Claim affected:** Feasibility, completeness of the registered model family, third-party reproduction, and protection against selective execution.

# Important non-blocking findings

## I01 — The null reporting language overstates absence of evidence

- **Severity:** Important — definite interpretation problem, but easily repaired.
- **Plan section or excerpt:** §10: if a CI includes 0.5, “the readout carries no entity-hallucination signal.”
- **Why it matters:** Failure to exclude chance does not establish no signal, especially with an unreported number of independent completions and possibly wide intervals. The same issue applies to §1’s “counts against” language. A result may be inconclusive rather than a scientific null.
- **Concrete minimum fix:** Use: “The study did not detect discrimination above chance at the registered precision,” followed by the estimate and CI. Reserve an affirmative “practically no signal” conclusion for a valid equivalence test showing the entire interval lies inside a frozen negligible-effect region around 0.5.
- **Claim affected:** Null reader conclusions and failed-replication language.

## I02 — “Output-head layer” should be treated as native likelihood, not as another internal lens location

- **Severity:** Important — semantic clarity.
- **Plan section or excerpt:** §5 reports logit lens at the primary layer and “the output-head layer”; §6 lists output-head versus primary-layer lens as secondary.
- **Why it matters:** Once final normalization and the actual output head are applied, this is the model’s native next-token likelihood baseline, not merely another approximate logit lens. Naming it loosely could obscure that it is a strong text-probability baseline and a crucial implementation check.
- **Concrete minimum fix:** Rename it “native output-token surprisal,” define the exact causal shift, and use it both as a reader and as the positive-control check against the model’s forward-pass logits.
- **Claim affected:** Reader taxonomy and interpretation of internal versus output-level information.

## I03 — The leave-one-domain endpoint is undefined

- **Severity:** Important — missing endpoint specification.
- **Plan section or excerpt:** §6: “leave-one-LongFact-domain macro-AUROC.”
- **Why it matters:** This could mean merely calculating AUROC separately by test domain, training while excluding each domain, or tuning on all but one domain and testing on the held-out domain. Those are different estimands. Some domains may also contain only one class, making AUROC undefined.
- **Concrete minimum fix:** Either delete the endpoint or specify domain labels, training/selection exclusions, macro-weighting, minimum class counts, treatment of undefined domains, and whether it is confirmatory or descriptive. The smallest option is a descriptive macro-average of per-domain test AUROCs with an explicit class-count rule.
- **Claim affected:** Domain robustness/generalization.

## I04 — The non-neural heuristic is not reproducible as written

- **Severity:** Important — missing baseline details.
- **Plan section or excerpt:** §5: “entity token-length + unigram frequency logistic baseline.”
- **Why it matters:** Token length depends on tokenizer and span boundaries; unigram frequency depends on corpus, case folding, normalization, subword versus string counts, smoothing, and handling of unseen entities. The logistic model also has regularization and class-weight choices.
- **Concrete minimum fix:** Pin the frequency corpus/revision, text normalization, tokenizer, frequency definition, OOV smoothing, feature transformations, model regularization, fitting split, and score direction. Fit it using exactly the same development/test IDs as the neural readers.
- **Claim affected:** “Earned nothing beyond a cheap heuristic.”

## I05 — Agreement on the unlabeled Gemma arm would not validate hallucination detection

- **Severity:** Important — construct limitation if the arm is retained.
- **Plan section or excerpt:** §8: label-free inter-reader agreement and rank correlation “still tests ‘do the readers agree on which entities are suspect.’”
- **Why it matters:** Correlated readers can agree because they share token likelihood, entity rarity, length, or preprocessing artifacts. Agreement has no direction toward factual validity without labels. Multiple readers derived from the same residuals are not independent witnesses.
- **Concrete minimum fix:** Label the endpoint purely “score agreement,” report scatterplots and correlations without validity language, include the length/frequency baseline in the agreement matrix, and do not use agreement to support the paper’s hallucination-detection claim.
- **Claim affected:** Interpretation of the exact Gemma-3-12B arm.

## I06 — Probe-training variability is not represented by a single frozen seed

- **Severity:** Important — judgment call about robustness, not automatically fatal.
- **Plan section or excerpt:** §5 says probe seed will be frozen; §6 reports bootstrap uncertainty over completions.
- **Why it matters:** Completion bootstrap uncertainty does not include optimization/initialization variability. A single seed is reproducible but may be atypically favorable or unfavorable, especially for a small attention probe.
- **Concrete minimum fix:** Use a small fixed set of training seeds, such as three, and define whether the primary score is from a predesignated seed or the mean of independently trained probes. Never select the best seed. If budget forbids repeats, keep one seed but explicitly state that the CI conditions on that fitted probe and does not represent training variability.
- **Claim affected:** Stability of the supervised-probe benchmark.

## I07 — The J-lens fit corpus and fitting estimand need a leakage and lineage statement

- **Severity:** Important — missing evidence, although label-free fitting is a strong choice.
- **Plan section or excerpt:** §5: J is fitted on a “wikitext-style” disjoint corpus and the campaign lens is reused.
- **Why it matters:** “Wikitext-style” is not an identifiable resource. The exact corpus, sequence construction, model revision, source/target activations, fit objective, regularization, sample count, and possible overlap with LongFact prompts are unspecified. Lack of label use prevents direct supervised leakage but does not establish reproducibility.
- **Concrete minimum fix:** Pin corpus revision and IDs, sampling seed, preprocessing, context length, objective, regularization, number of tokens, source and target hooks, held-out fit diagnostics, and matrix hash. State whether LongFact conversations/prompts are excluded.
- **Claim affected:** “Unsupervised fitted Jacobian lens” and third-party regeneration.

# What should remain unchanged

- **The explicit forbidden inferences.** The prohibition on belief, intent, experience, broad transfer, RL-policy, and 58% reduction claims is appropriately strict.
- **The independent-unit declaration.** Treating completions/conversations as clusters rather than treating every entity as independent is a strong and necessary design choice.
- **Use of a genuine held-out test split.** Reader fitting and selection should remain confined to development data, once train and validation roles are resolved.
- **Scoring readers on the same spans.** Paired comparison is substantially stronger and more efficient than separate reader samples. Preserve it and use paired cluster resampling.
- **All-test-set evaluation with no optional sample-size stopping.** Keep the rule that every independently eligible test span is scored and that no scientific stopping decision depends on interim AUROC.
- **Counted exclusions.** Unverifiable/NA annotations, tokenization failures, truncations, and other exclusions should all be enumerated rather than silently dropped.
- **Layer sensitivity not used for headline selection.** The prohibition on primary max-over-layers reporting is an excellent defense against researcher degrees of freedom.
- **Raw score receipts.** Per-span scores, labels, completion IDs, arguments, seeds, revisions, and software versions are sufficient statistics for independent table and CI reconstruction. Keep this requirement.
- **No stochastic web grader in the current budget.** Avoiding a new grader confound is sensible. The consequence is that Gemma-3-12B cannot be a gold-label replication arm, which should be stated rather than worked around.
- **Disclosing label-selected SAE analysis.** The plan already acknowledges that at least one latent would be label-selected. Preserve that disclosure, but classify the primary max-AUROC latent the same way.
- **Cheapest-first engineering validation.** It is reasonable to validate code on the 8B arm first, provided later scientific arms have already been funded and their execution is not conditioned on the 8B outcome.
- **Public immutable freeze.** OSF is not inherently required. A public signed tag/release with immutable manifests and descendant-lineage checks can be adequate if implemented as specified.
- **Reporting contrasts rather than celebrating a single target value.** This is a good organizing principle once “contrast” is defined as a paired estimand rather than CI overlap.

# Minimal revised design

1. **Narrow and name the study correctly.**
   - Frozen core claim: “On archived, public LongFact++ conversations for a pinned model, specified readers applied to causally aligned teacher-forced residual states discriminate public-annotation hallucinated from supported entity spans at held-out test AUROC.”
   - Do not call this a replication of the cited Gemma-3-12B paper result unless the exact paper data, labels, probe, hook, and protocol are obtained.
   - Use “discrimination,” not “calibration,” for the reader table.

2. **Choose one confirmatory primary arm.**
   - Use Llama-3.1-8B-Instruct as the primary because it has public labels, compatible artifacts, and the lowest cost.
   - Treat Llama-3.3-70B and gemma-2-9b as fixed secondary external-replication arms only after their complete artifact manifests and total budget are approved.
   - Move unlabeled Gemma-3-12B agreement to a separate exploratory amendment. It is not required for the decisive core result.

3. **Freeze the exact data estimand.**
   - Snapshot dataset revision and all split IDs.
   - Report completion and class counts before execution.
   - Analyze all independently eligible test spans, with eligibility defined without reader outcomes.
   - Define the primary AUROC as span-weighted AUROC with completion-clustered uncertainty, or choose a completion-weighted alternative and state it explicitly.
   - Preserve public annotation labels as the operational target and avoid claims of annotation-error-free factual truth.

4. **Use one common causal alignment.**
   - Reconstruct each archived conversation using the pinned tokenizer and chat template.
   - For each entity token \(x_t\), use only the residual state from the causal position that predicts \(x_t\); future answer text must be inaccessible.
   - Reset caches between completions.
   - Freeze punctuation, whitespace, multi-token entity, BOS, padding, and truncation rules.
   - Aggregate token scores to an entity score by one predeclared rule, such as the arithmetic mean, for every compatible scalar reader.
   - If the attention probe needs a sequence of token states, give it exactly the same causally aligned states and mask.

5. **Freeze exact reader definitions.**
   - **Probe:** exact code commit, architecture, layer/hook, fitting and validation roles, class weights, optimizer, epochs, initialization, mask, and entity aggregation.
   - **Native surprisal/logit lens:** exact final normalization, unembed, causal shift, and sign.
   - **J-lens:** exact source/target hooks, matrix/intercept, fitting corpus and objective, and score equation.
   - **SAE:** exact file, hook, scaling, latent statistic, and aggregation.
   - Classify the max-development-AUROC SAE coordinate as **label-selected sparse readout**, not unsupervised.
   - Omit an “unsupervised SAE” claim unless latent identity and sign are selected without hallucination labels.

6. **Use validation consistently.**
   - Train: fit probe parameters.
   - Validation: choose epoch/hyperparameters from a frozen candidate set, reader signs, SAE coordinate, any threshold, and probability calibration map.
   - Test: join frozen scores to labels once and evaluate.
   - If development sample size makes this allocation inefficient, define a fixed internal split or cross-fitting procedure before test scoring; do not alternate informally between train and validation.

7. **Separate discrimination from calibration.**
   - Primary reader table: raw-score test AUROC.
   - Optional calibrated-probability endpoint: fit a frozen logistic calibration map on validation and report test Brier score, log loss, calibration intercept/slope, and fixed-bin ECE.
   - Do not infer calibration from AUROC or call raw SAE/lens scores probabilities.

8. **Replace CI overlap with paired contrasts.**
   - Compute each reader’s AUROC and each reader-minus-probe AUROC difference using the same completion bootstrap resamples.
   - Freeze a practical margin, preferably the already contemplated 0.05.
   - Define whether the target is equivalence or noninferiority.
   - Use a small multiplicity correction across the primary reader contrasts, or designate one contrast primary and the others secondary.
   - Treat intervals wider than the margin as inconclusive, not as matching.
   - Report comparison with the heuristic using a paired difference, not point estimates alone.

9. **Add only decisive controls.**
   - Native-output-logit equality fixture.
   - Token/span alignment fixture.
   - SAE shape and reconstruction/firing fixture.
   - Identity-J/logit-lens equality fixture.
   - Shuffled-label probe negative control.
   - Procedure-matched random-latent control for the label-selected SAE.
   - Frozen sign formulas based only on validation.
   - A failed implementation gate triggers a logged repair and complete affected rerun, not selective interpretation.

10. **Freeze execution and cost before any scientific score is evaluated.**
    - Signed public release/tag, exact environment, commands, hardware, determinism settings, seeds, failure rules, and receipt schema.
    - Score-generation process cannot load test labels.
    - Test evaluator accepts only frozen score files and ID manifests.
    - Approve the whole registered model set and cost ceiling before seeing the 8B scientific outcome.
    - If only the 8B arm is funded, freeze only that arm rather than registering later arms conditionally.

This revised core remains comparatively small: one model, four clearly classified reader implementations, two simple baselines, one held-out test set, paired completion-clustered inference, and implementation fixtures. It answers a defensible reader-comparison question without requiring an RL run, a new web grader, a causal intervention study, or a large cross-model campaign.

# Freeze checklist

## Claim and prior-experiment boundary

- [ ] Final title says either “benchmark/extension” or documents the exact assets that make it a true replication.
- [ ] The exact target claim is one sentence and uses “discrimination” unless probability calibration is actually tested.
- [ ] The label construct is stated as agreement with the frozen public annotation rubric.
- [ ] Prior .88/.94 values are contextual unless the exact model/data/probe evaluation is reproduced.
- [ ] “Failed replication” language is removed for noncomparable Llama/gemma-2 arms.
- [ ] “Carries the signal,” “property of the features,” and “needs supervision” are narrowed to performance of the specified readers.
- [ ] No causal, belief, intent, experience, transfer, RL, or 58% reduction inference is permitted.

## Models, data, and artifacts

- [ ] Primary model is identified.
- [ ] Every secondary model is explicitly confirmatory, secondary, or exploratory.
- [ ] Exact model and tokenizer revisions are pinned.
- [ ] Dataset revision and split-ID hashes are pinned.
- [ ] Chat template content/hash is pinned.
- [ ] Generation revision/template provenance is known, or the claim says “teacher-forced pinned-model states.”
- [ ] Exact SAE repository revision and individual SAE file/config are pinned per model.
- [ ] Exact J-lens matrix/intercept/config hashes are pinned per model.
- [ ] Existing campaign J-lens is distributable, or a complete refit recipe is supplied.
- [ ] Gemma-3-12B is removed from the gold-label analysis.
- [ ] Any retained unlabeled Gemma