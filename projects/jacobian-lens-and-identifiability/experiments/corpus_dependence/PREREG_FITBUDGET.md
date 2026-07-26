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

---

## Amendment 1 (2026-07-26): extend the sweep to 1,000 prompts

**Frozen before the 1,000-prompt fits ran. Written after the 25/50/200/400 results were fully
inspected, and that is disclosed here rather than hidden.**

### Trigger

The original sweep returned CONVERGED, which discharged the fit-heterogeneity caveat between 25
and 400 prompts. It does not reach the budget the **public Neuronpedia lenses actually use**,
which is on the order of a thousand WikiText prompts, and those lenses are 35 of the 36 rows in
our zoo. Discharging a caveat over a range that excludes the population it was raised about is
not discharging it. TJ asked for the range to be closed properly.

### What had been inspected at the time of writing

Everything in `results_fitbudget.json` and the ledger entry in `results.md`: all eight
map distances, the ratios to the seed null, the zero boundary movement, and the failure of the
original P1 monotonicity prediction. This amendment is therefore **not blind**, and its result
should be read as confirmatory of a pre-existing threshold rather than as an independent test.

### Design

Two additional fits, `gpt2-small` and `gemma-3-270m` on WikiText seed 0 at **1,000 prompts**,
identical recipe and identical reference (each model's 100-prompt `wiki_a` fit). Same three
measures, same seed-null reference scale, same analyzer, unchanged.

### Decision rule (unchanged, deliberately)

The bar stays exactly what it was: **map distance within 2x the seed null** counts as converged.
We are not renegotiating the threshold to accommodate a new point. If either 1,000-prompt fit
exceeds it, the verdict for the whole sweep changes to NOT CONVERGED at the high end, the note's
discharge of the fit-heterogeneity caveat is withdrawn, and we report that the public lenses sit
in a regime our own fits do not reach. That outcome is worse for us than the alternative and we
commit to it in the same words either way.

### Prediction

Distances at 1,000 land within 2x the null, like every other budget. We state it so that a miss
is on the record. We do **not** predict monotonicity, having learned from P1 that ordering among
converged points is noise.

### Cost

Two fits, roughly 2.5x the wall-clock of the 400-prompt pair. One RTX 4090 at $0.69/hr for about
1.5 hours including model download, estimated **under $1.50**.
