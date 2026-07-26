# Pre-registration: does the fitted map depend on the fitting budget?

**FROZEN before any fit.** Written 2026-07-26; no fit-budget data existed at commit.

## Why

Every cross-lens comparison in our note assumes lenses fitted with different prompt budgets are
comparable. They are not obviously so: the public Neuronpedia lenses use a much larger budget
than our own fits (100 prompts), and our released 397B lens used 24. An external review flagged
this as an unmeasured confound underlying the entire 36-model zoo.

## Design

Refit **gpt2-small** and **gemma-3-270m** on WikiText seed 0 at budgets
**25, 50, 200, 400** prompts, identical recipe otherwise. The existing 100-prompt fit from the
corpus experiment is the reference. Compare each budget's map to the 100-prompt reference with
the same measures used there: boundary shift, map distance (`1 - CKA`), band-statistic shift.

**The reference scale is the seed null already measured** in the corpus experiment (map distance
0.0002 for gpt2-small, 0.0005 for gemma-3-270m). A budget effect only matters if it exceeds
ordinary fitting variation.

## Hypotheses

- **P1.** Map distance to the 100-prompt reference **decreases monotonically** as budget rises
  toward 100 and stays near the seed null at 200 and 400, i.e. the fit has converged by ~100.
- **P2.** The 25-prompt map is materially different, which would mean our own 24-prompt 397B
  lens is in the unconverged regime.

## Decision rules (frozen)

| outcome | verdict, and consequence for the note |
|---|---|
| distance at 200 and 400 within ~2x the seed null | **CONVERGED**: budget is not a confound above ~100 prompts, and the fit-heterogeneity caveat is discharged for that range |
| distance still falling at 400 | **NOT CONVERGED**: cross-lens comparisons in the zoo are budget-confounded, and we must say so as a scope condition alongside corpus dependence |
| 25-prompt map far from the rest | the released 397B lens (24 prompts) is separately caveated regardless of the above |

A NOT CONVERGED outcome weakens our own zoo comparisons and we commit to reporting it with the
same prominence as the corpus result.

## Cost

Eight fits on one RTX 4090, estimated **under $3** against remaining budget. Fitting cost scales
with prompt count, so the 400-prompt fits dominate.
