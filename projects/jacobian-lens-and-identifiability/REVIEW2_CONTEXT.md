# What changed since your last review, and what to check

You reviewed an earlier draft of this note and returned NOT READY FOR READER REVIEW with four
accuracy/claim findings, three statistical/scope findings, a reference audit, and two
figure-consistency findings. We accepted all of them. This is the revised note plus one new
experiment. Please review the note as it now stands.

## What we changed in response to you

- **B01 (conflicting numbers).** Added a "Reading the numbers" table defining fixed vs fitted,
  own vs shared probe, public vs independent fit, and stating that `fitted_sep >= mid_sep` by
  construction so it is not independent validation. Corrected the scale span (we said 500x; it
  is 5,700-fold) and the 36/35/34/38 denominators. Reconciled the concentration ranking: gpt-oss-20b
  at 0.351 is the most concentrated, and our own context file had wrongly said gemma-3-270m.
- **B02 (vocabulary mechanism).** Now reported as probe dependence. The shared probe is described
  as equalising which tokens are compared, explicitly not the null floor. Crowding is named as an
  unisolated hypothesis.
- **B03 (geometry vs mechanism).** "Same weights with and without tuning" removed as false.
  The unigram-prior identification downgraded to what was measured (0.48 correlation with
  embedding norm, function-word pole) with a note that the label needs a frequency test we
  did not run.
- **B04 (estimand).** Full estimand added: perturbation construction, equal-norm verification,
  8 random draws, aggregation, bootstrap over prompts, per-model intervals. We also corrected
  our own "ceiling" label: the comparator maximises one token's log-probability, not KL, so it
  is a reference direction and the fraction is relative to it.
- **S01/S02.** Zoo results recast as descriptive of the available lens collection.
- **R01/R02.** Bibliography split into prior-work claims vs software/data artifacts with what
  each warrants; Anthropic report marked provisional; jlens and Neuronpedia pinned to
  commit/snapshot; Gao et al. added; PrimeIntellect removed; Kornblith's claim narrowed;
  literature claim narrowed to "we are not aware of, and did not run a systematic search".
  Added a section stating exactly what "pre-registered" means here (git commits before data,
  not an external registry).
- **F01/F02.** Own-vocabulary figure captions relabelled as superseded rather than asserting a
  family split; the fixed-vs-fitted plot no longer reads as validation; the CKA-to-percentage
  conversion removed.
- **Structure.** The failed block-causality campaign moved to an appendix, which opens with a
  table of all four claims we published and then withdrew. Findings now lead.

## The new experiment (not in the draft you saw)

We ran the corpus-dependence test you recommended as a separate preregistered study, but as
part of this note because the editor wanted one self-contained picture. Nine fits, three models,
identical recipe: WikiText seed 0, WikiText seed 1, and code. The two WikiText fits give a seed
null. Result: boundaries are identical across seeds in all three models, and move by up to
fifteen layers on code, with map distance 87x to 292x the null. We moved the resulting scope
condition into the summary and the results rather than the caveats.

## What we want from this pass

1. Is anything **still** overclaimed, especially in the new corpus section?
2. Did our revisions introduce **new** errors or internal inconsistencies? We are poor judges of
   our own edits; last time our own review packet contradicted the note.
3. Is the appendix the right home for the failed campaign, or does moving it there make the
   note read as less honest than the previous draft?
4. Is the note now publishable, or is there a specific remaining blocker?
