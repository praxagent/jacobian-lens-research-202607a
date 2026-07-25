# Verdict

The current study does **not** support the broad claim that depth bands fail to partition computation. Cross-layer state substitution measures whether a residual state remains consumable at another depth; computational stages need not imply such “state typing,” especially in additive residual streams. A genuine staged computation could therefore produce the reported nulls or inversions.

The strongest reason the conclusion could be wrong is **construct mismatch**, not merely low power. The current intervention also skips or repeats computation and creates off-distribution states, so boundary effects are not uniquely identified even when distance and position are controlled.

The existing results can support only a narrow descriptive statement:

> In the tested models, CKA-derived boundaries did not consistently predict additional damage from cross-layer residual-state substitution beyond modeled depth and position effects.

They cannot establish equivalence, generalize across models, or negate computational partitioning broadly. As a paper, this is presently an informative failed-probe sequence or inconclusive result, not a decisive negative result.

The cheaper better test is an **O(L) native-time causal-profile test**: perturb each layer’s own residual update at its native location, construct a predeclared causal-effect profile, and test whether adjacent profiles change discontinuously at CKA boundaries relative to depth-matched placebo cuts. This avoids the O(L²) swap grid and the principal state-compatibility confound.

**NOT READY TO FREEZE**

# Blocking findings

### B01 — The estimand does not identify the stated construct

- **Severity:** Blocking
- **Plan section or excerpt:** “Is ‘can layer `j` consume layer `i`’s representation’ the right operationalization of ‘blocks are computational units’?”
- **Why it matters:** No. Cross-layer consumability is neither necessary nor sufficient for computational partitioning. Residual states are running sums rather than explicitly stage-typed interfaces. Swapping between depths additionally skips some transformations, repeats others, changes activation statistics, and presents downstream layers with states they were not trained to consume. A boundary penalty could reflect distribution shift; its absence could coexist with real staged computation.
- **Concrete minimum fix:** Either narrow the claim to cross-layer substitution compatibility, or replace the primary estimand with a native-time intervention. The smallest useful replacement is a boundary discontinuity in adjacent layers’ causal-effect profiles under standardized perturbations to each layer’s own residual update.
- **Claim affected:** “We find no evidence [depth bands] partition computation.”

### B02 — A non-significant test is being treated as evidence of absence without an equivalence target

- **Severity:** Blocking
- **Plan section or excerpt:** “`beta = +0.220`, p = 0.18. Null”; “Are we entitled to any negative claim at n = 2 balanced models?”
- **Why it matters:** Failure to reject a point null does not show the effect is absent or practically small. The fourfold variation in null SDs, opposite-signed model effects, and two balanced models leave both meaningful positive effects and heterogeneous effects plausible. The study currently cannot distinguish “small,” “variable,” and “poorly measured.”
- **Concrete minimum fix:** Define a smallest effect of scientific interest, `δ`, on a standardized model-level estimand before collecting new outcomes. Power the study for equivalence or report a confidence interval against `±δ`. If the interval is wider than that range, the required disposition is “inconclusive,” not “negative.”
- **Claim affected:** Any negative or absence claim, including the proposed headline.

### B03 — The segmentation permutation does not provide population-level inference

- **Severity:** Blocking
- **Plan section or excerpt:** “Significance comes from a permutation null of random 3-segmentations with the same block-size multiset.”
- **Why it matters:** This tests whether the fitted boundary placement is unusual relative to chosen placebo placements within one model, conditional on assumptions about exchangeability across depth. It does not make models independent replicates or support a claim about models generally. Layer position is highly structured, and regression adjustment does not by itself establish that segmentations are exchangeable. The packet also does not specify the permutation support; with fixed contiguous segment sizes, it may be very small.
- **Concrete minimum fix:** Use each independently trained model family as the inferential unit. Compute one predeclared boundary-versus-matched-cut contrast per model, use placebo cuts only for within-model calibration, and perform the confirmatory inference across model families. Before freeze, locally verify the exact permutation generator, support size, and exchangeability rationale.
- **Claim affected:** Generalization from the tested boundaries to depth bands across language models.

### B04 — There is no positive-control gate showing that the instrument can detect computational staging

- **Severity:** Blocking
- **Plan section or excerpt:** Controls are distance and mean-position dummies; v1 saturated and v2 null SDs vary fourfold.
- **Why it matters:** Distance and position controls address confounding but do not establish sensitivity or construct validity. Saturation in v1 and heterogeneous noise in v2 make a null compatible with an insensitive or unstable instrument. Without a known-positive case, positive, null, and invalid outcomes cannot be separated.
- **Concrete minimum fix:** Before testing natural-model boundaries, require both:  
  1. a dose-responsive output effect from the native-time perturbation in every eligible model; and  
  2. recovery of a known imposed stage boundary in a small synthetic or explicitly bottlenecked positive-control model using the same frozen analysis.  
  Failure of either gate makes the natural-model test invalid, not negative.
- **Claim affected:** Whether a null can be interpreted as evidence against computational organization rather than instrument failure.

# Important non-blocking findings

### I01 — Model eligibility creates a strong and potentially selective claim boundary

- **Severity:** Important
- **Plan section or excerpt:** “Only models >= 27B have balanced fitted blocks”; “one affordable model with balanced blocks.”
- **Why it matters:** Selecting models because their segmentation gives favorable crossing counts can entangle architecture, scale, affordability, and block balance. Small-model results are not intrinsically uninterpretable, but the current pairwise-crossing estimand gives them poor leverage.
- **Concrete minimum fix:** Predeclare eligibility independently of observed causal effects. Prefer the adjacent-boundary causal-profile statistic, which does not require balanced block lengths. Treat checkpoints from one lineage as clustered, not independent models.
- **Claim affected:** External validity across model sizes and families.

### I02 — The campaign history requires a single protected confirmatory endpoint

- **Severity:** Important
- **Plan section or excerpt:** v2’s first-boundary effect was selected exploratorily and then tested in v2.1; the second boundary became salient after that result.
- **Why it matters:** The held-out preregistered replication is a strong choice, but repeated movement among global, first-boundary, second-boundary, model-specific, and sign-specific summaries permits retrospective reinterpretation of a mixed campaign.
- **Concrete minimum fix:** Freeze one pooled boundary statistic, one direction or two-sided equivalence criterion, one model eligibility rule, and one analysis population. Label all boundary-specific and architecture-specific results exploratory, regardless of significance.
- **Claim affected:** Confirmatory status and resistance to selective interpretation.

### I03 — Cross-model effect scaling and outcome validity are not yet specified

- **Severity:** Important
- **Plan section or excerpt:** “Measure `KL(patched || clean)`”; “per-model null SDs vary 4x.”
- **Why it matters:** Raw KL effects can vary with model calibration, vocabulary entropy, and perturbation magnitude. This can obscure a common effect and make model-level aggregation arbitrary. KL direction also changes interpretation.
- **Concrete minimum fix:** Predeclare KL direction, token support, perturbation normalization, and a model-level standardized effect based only on blinded or control variation. Include a task-relevant outcome alongside distributional change if the claim concerns computation serving behavior.
- **Claim affected:** Comparability and substantive interpretation of effects across models.

# What should remain unchanged

- Keep each model’s CKA segmentation fixed before causal outcomes are examined; do not hand-adjust boundaries to fit intervention results.
- Preserve explicit control for smooth depth trends. The recognition that distance and absolute position can mimic a boundary effect is a major strength.
- Preserve the held-out-model replication principle and the honest reporting of the sign inversion. Do not average it away or relabel the second boundary as confirmatory.
- Preserve full disclosure of the v1 saturation failure and the exploratory-to-confirmatory history.
- Continue using model-specific boundaries rather than importing one universal layer partition.
- Keep raw records, output dumps, and source trees outside the director packet; they should be checked locally under a frozen audit protocol.

# Minimal revised design

1. **Narrow the confirmatory claim.**  
   Use:

   > CKA-derived depth boundaries do not mark a practically meaningful discontinuity in native-time causal influence profiles, within the prespecified model population and intervention class.

   Do not claim that depth bands fail to partition computation in every meaningful sense.

2. **Use one native-time estimand.**  
   At each layer `l`, apply a standardized perturbation to that layer’s own residual update at its normal execution time—such as a calibrated attenuation or matched-noise intervention. Measure a fixed vector of effects on held-out behavioral and output-distribution outcomes. Define the adjacent causal-profile discontinuity between `l` and `l+1`.

3. **Define one boundary contrast.**  
   For each model, compare the mean discontinuity at its two frozen CKA boundaries with the expected discontinuity at depth-matched placebo cuts. Fit the depth trend without using boundary labels. Pool the two boundaries in the primary endpoint; analyze them separately only as exploratory outcomes.

4. **Use models, not layer pairs, as independent units.**  
   Sample independently trained model families under frozen eligibility criteria. Cluster related checkpoints and variants by family. Choose the number of families from a blinded variance exercise and the prespecified equivalence margin; do not freeze a negative-result study until it has adequate probability of placing the confidence interval inside `[-δ, +δ]` when the true effect is negligible.

5. **Add positive-control gates.**
   - Confirm that intervention dose produces a stable, monotonic response in each model.
   - Demonstrate recovery of an imposed boundary in a small staged or bottlenecked control model.
   - If either fails, classify the natural-model result as invalid.

6. **Freeze interpretation rules.**
   - Confidence interval wholly within `[-δ, +δ]`: evidence against a practically meaningful discontinuity for the narrowed construct.
   - Interval excludes zero but not the equivalence region: positive or directional evidence as specified.
   - Interval overlaps effects larger than `δ`: inconclusive/underpowered.
   - Positive-control or intervention-fidelity failure: invalid.
   - Strong cross-family heterogeneity: report heterogeneity; do not replace it with a universal null.

This design is cheaper than the cross-layer swap matrix because it requires approximately one intervention series per layer rather than all source-target layer pairs.

# Freeze checklist

- [ ] Replace “partition computation” with an exact, intervention-linked construct and claim boundary.
- [ ] Specify the primary native-time perturbation, dose, residual-stream location, and whether it occurs before or after normalization and residual addition.
- [ ] Define the causal-effect profile, primary outcome vector, KL direction, and model-level standardization.
- [ ] Set a scientifically justified equivalence margin `δ`.
- [ ] Complete a blinded power/precision calculation and freeze the number of independent model families.
- [ ] Freeze model eligibility and lineage-clustering rules independently of causal outcomes.
- [ ] Freeze the CKA fitting data and boundaries; use disjoint held-out prompts/tasks for causal evaluation.
- [ ] Specify one pooled boundary statistic and depth-matched placebo-cut procedure.
- [ ] Locally verify the permutation support and code path; do not rely on the current verbal description.
- [ ] Predefine positive-control gates and the “invalid” disposition if they fail.
- [ ] Freeze multiplicity, stopping, exclusions, missing-output handling, and judge/blinding rules.
- [ ] Verify locally that no prompt selection, task selection, or intervention calibration used confirmatory outcomes.
- [ ] Preserve v1, v2, and v2.1 as disclosed exploratory/prior evidence rather than pooling them into the new confirmatory test.
- [ ] If no new decisive study is run, publish the current campaign only as a narrowly stated, inconclusive intervention result—not as evidence that depth bands do not partition computation.
