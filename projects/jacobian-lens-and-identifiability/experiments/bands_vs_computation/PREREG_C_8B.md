# Pre-registration: Test C at 8B (llama3.1-8b, 400-prompt lenses)

Frozen 2026-09-08 before any damage matrix for llama3.1-8b exists. Same design, statistic, gates and
decision rules as Test C in `PREREG.md`; one model instead of three.

## Why now

Test C asked whether the fitting corpus changes which boundary predicts damage. It was reported as
**moot** for the three sub-1B models because, under the corrected map statistic, their WikiText and
code boundaries differ by at most one layer, so the crossed design had almost nothing to cross. The
8B second attempt (`../corpus_dependence/PREREG_8B_v2.md`, 2026-09-08) supplies the contrast the
design needs: on converged 400-prompt fits, llama3.1-8b's fitted boundaries are **(13, 17)** on
WikiText (both seeds, and the public lens) and **(6, 17)** on code, both well identified on their own
maps. This is the one model where Test C's premise holds, so we run it there and nowhere else.

## What had been seen when this was written

All of the 8B second attempt (boundaries, map distances, the post-hoc identifiability check) and all
of the original Test C results. No damage matrix for llama3.1-8b exists; none has been looked at.

## Design (unchanged)

`swap_damage.py` measures `D(i, j)`, the KL damage from swapping layer `i`'s captured clean output
into layer `j` on the same prompt, on the frozen prompt artifact (`prompts_frozen.json`: 48 prose and
48 code prompts, 64 target tokens), float32, on one GPU. Then, holding `D` fixed within each cell:

    D(i,j) ~ dummies(|i-j|) + dummies(mean position) + beta * crosses(i,j)

with boundaries from the WikiText lens (13, 17) and from the code lens (6, 17), crossed with the
damage corpus (prose, code), exactly the 2x2 of `PREREG.md`. Boundaries come from
`../corpus_dependence/results_8b_n400_raw.json` (slug `llama3.1-8b-n400`), the 400-prompt fits; the
lens-layer-to-block index convention is the one the original Test C used for its three models,
unchanged. The random-3-segmentation null (1,000 draws, same block-size multiset) is unchanged.

- **C1**: the mean of the two matched-cell `beta`s (damage on prose with WikiText boundaries, damage on
  code with code boundaries) is above its null at one-sided `p < 0.05`, where the null is the
  elementwise mean of the two cells' random-segmentation null draws (1,000 each, same block-size
  multisets as the frozen Test C).
- **C2**: the mean over the two damage corpora of `beta(matched) - beta(mismatched)` is positive at
  one-sided `p < 0.05`, where each cell's null is the elementwise difference of the matched and the
  mismatched segmentations' null draws and the pooled null is their mean. The original `analyze_C.py`
  reported per-arm two-sided p-values and pooled means only; these two pooled one-sided tests are
  added to it now (`C1_pooled_p`, `C2_pooled_p`), before any 8B damage matrix exists, and re-running
  it on the three original models leaves every previously recorded field unchanged.

The mechanical control (gpt2-small, identical segmentations, C2 exactly 0) is not re-run; its
recorded result stands as the check on the analyzer.

## Gates (unchanged)

`D(i, i) = 0` exactly; median `D` over `|i-j| >= 2` inside `[0.05, 5.0]` nats per cell, else that
cell is out of dynamic range and reported as such.

## Decision table (unchanged)

| C1 | C2 | verdict |
|---|---|---|
| significant | significant | BOUNDARIES PREDICT DAMAGE, AND THE CORPUS MATTERS |
| significant | not | BOUNDARIES PREDICT DAMAGE, CORPUS-INVARIANTLY (needs replication before any claim) |
| not | significant | INCOHERENT: report as such and claim nothing |
| not | not | NO PURCHASE, at the one scale where the premise holds |

## Prediction

**NO PURCHASE.** The block campaign's position is that the bands describe first-order readout
geometry; a boundary that relocates by seven layers with the fitting corpus is, if anything, more
evidence that the boundary is a property of the estimation distribution than of the computation.
C2 is the arm most likely to move, because the contrast is now real. Stated in advance; a
significant C2 here would be the first positive of the campaign and would be reported as exactly that,
single model, needing replication.

## Power, stated before running

One model. The random-segmentation null's spread is set by segmentation variability, so 992 pairs
per cell buy nothing; a null is weak evidence and will be reported as weak. C2 is a within-model paired
contrast with model, layers, positions, distances and prompts identical across arms, which is the
best-identified comparison the campaign has.

## Cost

Two damage matrices (prose, code) of 32 x 32 patched forward passes at batch 48 x 64 tokens, float32,
on one RTX A6000 (8B in float32 is 32 GB; fits). Roughly 2,000 forward passes at a few seconds each:
about 2 to 3 h including download, **about $1.50 at $0.53/hr**. Receipts carry per-prompt KL; `DONE`
marker; scheduled fetch and terminate from the work box; no API key on the pod.
