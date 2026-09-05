# Pre-registration B3: Test B restricted to models whose fitted boundaries are identified

Frozen 2026-09-05, before the restricted analysis is run. **Disclosure of what had already been
seen:** the per-model boundary-identifiability table (the spread of near-optimal segmentations on
the shared maps, computed while auditing the own-vocab/shared-probe defect) had been inspected
before this rule was written. The restricted Test B analysis, and which models it admits, had not
been run. The Test B verdict on all usable models (`results_B2.json`, shared probe) was known.

## Why

The fitted three-segmentation maximises mean within-block CKA. For a map whose objective is nearly
flat, many segmentations are almost as good as the optimum, and "the boundary" is then a nearly
arbitrary pick among them. Test B compares lens boundaries to activation boundaries; when either
is poorly identified, agreement is compared against noise, which biases the test toward its null.
The pre-registered Test B did not gate on this. This amendment does.

## Identifiability measure (frozen)

For a map `M` (L x L), score every legal segmentation `(b1, b2)` with the atlas objective
(`atlas_stage_a.fitted_seg`: sum over the three blocks of mean off-diagonal within-block CKA).
Let `best` and `worst` be the maximum and minimum scores. The **near-optimal set** is every
segmentation with score `>= best - 0.05 * (best - worst)`. The **spread** of each boundary is
`(max - min)` of that boundary over the near-optimal set, divided by `L`.

A boundary pair is **identified** if both spreads are `<= 0.25`. The 5% and 0.25 thresholds are
chosen before the restricted analysis and are not tuned afterwards; the sensitivity to 0.15 and
0.35 is reported alongside but does not change the verdict rule.

## Restricted analysis (frozen)

Apply the gate to the **lens** map (shared probe) **and** to the **activation** map (raw, and
standardised) of each Test B model. A model enters a cell only if it passes the cell's existing
usability gate **and** both its lens boundaries and that cell's activation boundaries are
identified. Everything else in `analyze_B2.py` is unchanged: same nulls, same pooled p, same
`MIN_USABLE = 4`, same four cells, same decision rule:

- pooled `p < 0.05` and a strict majority beat their own null median -> BANDS TRACK REPRESENTATIONS
- pooled `p >= 0.05` -> BANDS ARE A READOUT PROPERTY
- otherwise -> MIXED
- fewer than 4 admitted models in a cell -> NO VERDICT for that cell (reported, not pooled away)

## Prediction

Consistent with every earlier result, we predict BANDS ARE A READOUT PROPERTY in every cell that
reaches a verdict. We state plainly that this is the analysis in which a real boundary
correspondence would have its best chance to show, because the noise models are removed, so a
positive here would carry more weight than the unrestricted null carries, and we would report it
with the same prominence.

## What this is and is not

An exploratory-labelled robustness analysis of a pre-registered test, frozen after the
identifiability spreads were seen and before the restricted verdicts were computed. It cannot be
called a blind pre-registration and the note will not call it one.
