# Verdict

The question is worthwhile, the campaign is prospective, and the move from cross-depth transplantation to native-time intervention is a substantial improvement. However, the study can support only a narrow claim about **abrupt changes in layer-wise output sensitivity under attenuation**. It cannot identify computational stage boundaries as ordinarily understood.

**Single strongest reason this design will also fail:** a computational stage boundary does not imply an adjacent jump in the final-output importance of individual layers. Attenuation measures how consequential each block’s native update is, not whether computation is partitioned across a cut. Thus a genuine stage boundary may yield a smooth profile, while update magnitude, normalization, architecture, or shared Jacobian geometry may create a jump without a stage. Positive and null results would therefore both be overinterpretable.

Historical dispositions:

- Prior B01 is closed for transplantation, but not for the broader stage-boundary construct.
- Prior B02 is not closed merely by naming `delta = 0.5`; the margin still lacks a decision-based justification.
- Prior B03 is partly closed by using families as units, but family aggregation, independence, and power remain unspecified.
- Prior B04 is addressed in intent, but neither proposed gate is a valid end-to-end positive control for the target construct.

I reviewed only the supplied brief; implementation, boundary files, prompts, and model artifacts were not available for inspection.

**NOT READY TO FREEZE**

# Blocking findings

## B01 — Attenuation does not identify a computational stage boundary

- **Severity:** Blocking
- **Plan section:** “Construct and claim boundary”; “Intervention”; research question 1.
- **Basis:** Construct-validity judgment with a definite claim-boundary consequence.
- **Why it matters:** Fractional attenuation estimates the effect of changing one block’s update on final outputs. A stage boundary is a property of organization or transmission across a cut. There is no necessary correspondence between those properties. A positive result establishes co-location with a sensitivity jump; a null does not refute staged computation.
- **Concrete minimum fix:** Freeze the claim as: “CKA cuts predict abrupt changes in block-update sensitivity under a specified attenuation intervention.” Remove “computational stage,” “partition,” and equivalent interpretations. If a stage claim is essential, first define a necessary cut-level property—such as altered transmission of norm-matched perturbations across an interface—and validate that property in a small end-to-end-trained modular model before launching the family campaign.
- **Claim affected:** Any interpretation that CKA boundaries identify computational stages. The narrower causal-sensitivity claim remains potentially supportable.

## B02 — The primary outcome and cut statistic are not executable as written

- **Severity:** Blocking
- **Plan section:** “Outcomes”; “Primary statistic.”
- **Basis:** Definite specification defect.
- **Why it matters:** `e_l` is a two-component vector, but no scaling, norm, covariance treatment, or multivariate residual definition is given. `1 - Acc_l` is behavioral error, not the causal change from clean accuracy. “Monotone spline or local regression” leaves the model class and tuning undecided. The placebo set is also unclear and may contain only a few correlated cuts when segment-length multisets are merely permuted, making its SD unstable. Different reasonable implementations can reverse the result.
- **Concrete minimum fix:** Use one scalar confirmatory outcome—preferably held-out-prompt KL sensitivity—with behavioral accuracy change as a secondary outcome. Freeze one cut score, bandwidth, transformation, and placebo rule. A simpler option is a prespecified local left-versus-right contrast over two layers on each side, ranked against nonboundary cuts matched on normalized depth and architecture status. Avoid division by a small, model-specific placebo SD.
- **Claim affected:** Both positive and equivalence conclusions about discontinuity.

## B03 — The design does not distinguish CKA information from its strongest cheap alternative

- **Severity:** Blocking
- **Plan section:** “Primary statistic”; eligibility based on a published Jacobian lens.
- **Basis:** Missing required controls.
- **Why it matters:** For small attenuation, output KL is approximately a quadratic function of the residual update and the downstream logit Jacobian. CKA boundaries are themselves derived from Jacobian readout geometry. An association can therefore arise from shared mathematical inputs, update norms, normalization changes, or architectural transitions rather than from a distinctive stage boundary.
- **Concrete minimum fix:** Use disjoint prompts for CKA segmentation and causal evaluation, and prespecify at least these cheap comparators: equal-depth cuts, cuts selected by changes in residual-update norm, cuts selected by first-order Taylor-predicted KL or Jacobian norm, and architecture-transition cuts. The confirmatory question should be whether CKA adds predictive value beyond these controls, not merely whether it beats arbitrary cuts.
- **Claim affected:** The claim that CKA-derived boundaries have distinctive causal meaning beyond generic depth or sensitivity structure.

## B04 — Both positive-control gates are misaligned

- **Severity:** Blocking
- **Plan section:** “Positive controls.”
- **Basis:** Definite control-logic defect.
- **Why it matters:** Exact monotonic KL across three large doses is not required for a valid intervention; nonlinear trajectories can be nonmonotone, especially at `alpha = 1`. Conversely, mechanical scaling can be correct even if output KL is small. The post-hoc low-rank projection creates an unusually severe, off-distribution bottleneck and is not itself one of the attenuated layer updates. Detecting it would show sensitivity to gross damage, not sensitivity to natural stage boundaries or subtle profile discontinuities.
- **Concrete minimum fix:** Separate three controls:
  1. **Execution gate:** `alpha = 0` reproduces clean activations/logits within frozen tolerance, and the targeted update is numerically scaled by exactly `1-alpha`.
  2. **Statistic-sensitivity gate:** blinded synthetic profile spike-ins at the chosen equivalence margin are recovered at a prespecified rate.
  3. **Construct control:** only if retaining a stage interpretation, use a small end-to-end-trained modular/composed model with a hard information interface, acceptable clean performance, and a boundary hidden from the analysis pipeline.
  
  Keep dose response and the inserted projection as diagnostics, not automatic validity gates.
- **Claim affected:** Interpretability of null results and any claim that the instrument detects genuine stages.

## B05 — The equivalence margin, family claim, and target N are not justified

- **Severity:** Blocking
- **Plan section:** “Equivalence margin”; “Eligibility and units.”
- **Basis:** Definite missing inferential design.
- **Why it matters:** Half of a model-specific placebo SD has no demonstrated behavioral or scientific meaning, and that SD may vary with depth, cut count, and profile roughness. A population mean also does not establish “consistency” across families. With an illustrative between-family SD of 1, a two-one-sided-test equivalence design with bounds ±0.5 requires approximately 35 families for 80% power and 44 for 90% power under favorable known-variance assumptions; Student-\(t\), heterogeneity, and attrition increase those numbers. Eight families would achieve 80% equivalence power only if between-family SD were roughly 0.48 or smaller.
- **Concrete minimum fix:** Put the effect on an interpretable scale, such as the probability that a CKA cut has a larger local contrast than a matched nonboundary cut. Justify the margin as a minimum useful discrimination level. Then either:
  - restrict inference to eight named families and make no population-equivalence or broad-consistency claim; or
  - obtain an external or label-blinded estimate of between-family variance, preregister sample-size re-estimation and a maximum N, and recruit the resulting number of genuinely independent families.
  
  Define positive as a CI wholly above the positive meaningful margin, equivalence via a 90% CI wholly inside the equivalence region, anti-association separately, and everything else as inconclusive.
- **Claim affected:** Cross-family generalization, consistency, and evidence for absence.

# Important non-blocking findings

## I01 — The family-level unit is stated but not constructed

- **Severity:** Important
- **Plan section:** “Checkpoints from one lineage … clustered as one family”; “one `Z_model` per family.”
- **Basis:** Missing evidence/specification.
- **Why it matters:** It is unclear whether one checkpoint is selected per family or several checkpoint scores are averaged, modeled hierarchically, or otherwise weighted. Shared architectures, training corpora, and ancestral checkpoints may also violate the intended independence.
- **Concrete minimum fix:** Prefer one frozen representative checkpoint per prespecified family. Otherwise freeze a family aggregation rule and define independence by training ancestry, not merely by model branding.
- **Claim affected:** Cross-family uncertainty and effective N.

## I02 — Native-time attenuation is better, but it is not on-distribution by construction

- **Severity:** Important
- **Plan section:** “the network is only ever asked to consume states it produced itself.”
- **Basis:** Definite conceptual overstatement plus local verification requirement.
- **Why it matters:** The attenuated residual state is intervention-created and may be unlike training states, particularly at `alpha = 1`. Transformer blocks may also contain multiple serial or parallel residual additions, making “the layer’s update” architecture-dependent.
- **Concrete minimum fix:** Retain the intervention but describe it as lower-distance, not on-distribution. Use the lowest dose as primary or justify `alpha = 0.5`; treat `alpha = 1` as a stress test. Locally verify whether attenuation is applied to the total block map or to individual residual branches for every eligible architecture.
- **Claim affected:** Interpretation of dose effects and comparability across architectures.

## I03 — Behavioral scoring, prompt precision, and leakage rules are incomplete

- **Severity:** Important
- **Plan section:** “held-out prompt set”; exact-match accuracy.
- **Basis:** Missing evidence.
- **Why it matters:** The brief does not specify prompt counts, generation-time intervention, answer normalization, multiple valid answers, missing generations, or overlap with Jacobian-lens fitting data. Behavioral improvements are also possible, so `1-Acc` should not be described as harm.
- **Concrete minimum fix:** Freeze disjoint discovery/evaluation prompt IDs, prompt-sampling precision targets, intervention at every autoregressive generation step, deterministic answer normalization, missing-output treatment, and signed `Acc_perturbed - Acc_clean`. Keep behavior secondary unless separately powered.
- **Claim affected:** Behavioral corroboration and leakage-free evaluation.

## I04 — Multiplicity, stopping, and invalid-run handling need prospective rules

- **Severity:** Important
- **Plan section:** Three doses, two outcomes, multiple models and gates.
- **Basis:** Missing specification.
- **Why it matters:** Without one primary dose/outcome/statistic and rules for failed checkpoints, replacements, and interim additions, the study can be reinterpreted after results are visible.
- **Concrete minimum fix:** Declare one confirmatory endpoint; label all other doses, behavioral outcomes, baselines, and heterogeneity analyses secondary or diagnostic. Prohibit outcome-based checkpoint replacement and unblinded stopping. Report all eligible failures, with study-level invalidity determined by frozen control rules.
- **Claim affected:** Confirmatory error control and credibility of mixed outcomes.

## I05 — The compute estimate omits the main reproduction burden

- **Severity:** Important
- **Plan section:** “Cost.”
- **Basis:** Missing feasibility evidence.
- **Why it matters:** Forward-pass cost may be small, but storage, download time, access restrictions, architecture-specific hooks, and obtaining 35 or more independent families could dominate. “Small models run free on CPU” is not a complete resource estimate.
- **Concrete minimum fix:** Before scaling beyond a named eight-family panel, inventory checkpoint availability, licenses, compressed and resident storage, wall-clock time, and architecture compatibility. Keep the packet compact; verify hashes and execution locally rather than adding manifests or logs here.
- **Claim affected:** Feasibility of the powered cross-family design.

# What should remain unchanged

- The prospective, outcome-free design review should remain in place.
- Cross-layer state substitution should remain abandoned.
- Native-time, in-place attenuation with an explicit `alpha = 0` identity condition is a strong design choice.
- CKA boundaries must remain frozen before any causal outcomes are examined.
- Semantic claims about sensory, workspace, or motor stages should remain excluded.
- Training lineages should remain clustered as families; checkpoint count must not be treated as inferential N.
- The removal of the balanced-block eligibility requirement should remain.
- KL direction should remain frozen rather than selected after seeing results.
- The \(O(L)\) intervention schedule is proportionate and should not be expanded into an expensive perturb-everything campaign.
- The four-way outcome logic—positive, null/equivalent, inconclusive, invalid—is valuable and should remain after its thresholds and gates are corrected.

# Minimal revised design

1. **Freeze the narrow claim.**  
   “Across the prespecified target families, frozen CKA cuts predict larger local changes in block-update sensitivity under attenuation than depth- and architecture-matched nonboundary cuts.” Do not call this computational-stage identification.

2. **Use one scalar primary sensitivity measure.**  
   Use held-out-prompt KL as primary. Prefer a low-dose estimate, such as a frozen summary of `alpha = 0.25` and `0.5`; reserve `alpha = 1` for stress testing. Use signed clean-to-perturbed accuracy change as secondary corroboration.

3. **Replace spline-plus-placebo-SD standardization with a simple cut comparison.**  
   For every eligible cut, compute a frozen local contrast using the same number of layers immediately on either side. For each CKA boundary, compare that score with nonboundary cuts matched on normalized depth, edge distance, and architectural-transition status. Convert it to a within-family percentile or probability-of-superiority score, then average the two boundary scores within the family.

4. **Test CKA against cheap explanations.**  
   Run the same frozen scoring procedure for equal-depth cuts, update-norm-change cuts, first-order predicted-sensitivity cuts, and architecture transitions. CKA-specific evidence requires improvement over these controls on independent evaluation prompts.

5. **Use valid control gates.**
   - Verify `alpha = 0` identity and exact numerical scaling locally.
   - Demonstrate recovery of blinded profile jumps at and around the chosen meaningful margin.
   - If stage language is ever restored, first pass a separate trained modular-model construct control. Do not use the post-hoc rank bottleneck as the sole gate.

6. **Choose one inferential scope.**
   - **Smallest campaign:** eight named families, one representative checkpoint each, all-family effects reported, with conclusions restricted to that finite panel.
   - **Population campaign:** justify a meaningful margin, estimate between-family variance without unblinding the true-boundary effect, and increase N accordingly. Under SD ≈ 1 and margin 0.5, plan on roughly 35–44 families before inflation rather than eight.

7. **Freeze decisions before outcomes.**
   - Meaningful positive: CI wholly above the positive margin.
   - Equivalence: 90% CI wholly within the equivalence bounds.
   - Meaningful anti-association: CI wholly below the negative margin.
   - Otherwise: inconclusive.
   - Invalid only for prespecified execution or positive-control failures, not for nonmonotone scientific results.

If the actual scientific objective is causal meaning of **Jacobian geometry**, a better preliminary experiment is an aligned-versus-random, equal-norm residual perturbation study at CKA and matched cuts in two or three small models. That directly tests whether the geometric directions predict causal transmission while separating geometry from native update magnitude. It should replace, not be added to, the family campaign if stage interpretation remains the priority.

# Freeze checklist

- [ ] Claim text is limited to attenuation sensitivity, or a separately validated operational definition of “stage boundary” is supplied.
- [ ] One primary scalar outcome, dose summary, cut score, bandwidth, transformation, and family score are fully executable from the preregistration.
- [ ] The equivalence margin is tied to a minimum useful discrimination or raw effect, not a conventional placebo SD alone.
- [ ] Target population is either eight named families or a defined family universe with a formal power calculation.
- [ ] One checkpoint per family is frozen, or the within-family aggregation rule is frozen.
- [ ] Family independence is checked locally from training ancestry and checkpoint provenance.
- [ ] CKA discovery prompts and causal evaluation prompts are disjoint; IDs and hashes are retained locally.
- [ ] Equal-depth, update-norm, first-order sensitivity, and architecture-transition comparators are frozen.
- [ ] Local source inspection confirms the exact residual location being scaled for each architecture.
- [ ] `alpha = 0` clean-logit identity and exact update-scaling tests pass within frozen tolerances.
- [ ] Autoregressive intervention timing, KL precision, answer normalization, missing generations, and judging rules are frozen.
- [ ] Prompt count is justified by measurement precision; family remains the confirmatory inferential unit.
- [ ] Execution, statistic-sensitivity, and any construct-control gates are separated and frozen.
- [ ] Nonmonotone dose response is treated as a result, not automatic evidence of implementation failure.
- [ ] Multiplicity, interim access, checkpoint replacement, stopping, exclusions, and failed-family handling are preregistered.
- [ ] The CI rules for positive, equivalent, anti-associated, inconclusive, and invalid outcomes are internally consistent.
- [ ] Checkpoint access, licenses, storage, wall time, and reproduction artifacts are confirmed locally before committing to the powered N.
