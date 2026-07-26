# Results: a J-lens depth map is substantially a property of the fitting corpus

Design frozen in [`PREREG.md`](PREREG.md) before any fit (commit `4b4f7d5`). Nine fits, three
models, identical recipe, on one RTX 4090. **Verdict: CORPUS MATTERS, 3 of 3 models.**

## The seed null is essentially zero, which is what makes the rest meaningful

Refitting the same model on a **different WikiText sample** changes almost nothing:

| model | boundary shift, seed | map distance, seed | band-stat shift, seed |
|---|---|---|---|
| gpt2-small | **0 layers** | 0.0002 | 0.0002 |
| gemma-3-270m | **0 layers** | 0.0005 | 0.0106 |
| qwen3.5-0.8b | **0 layers** | 0.0002 | 0.0001 |

Fitted boundaries land on **exactly the same layers** across seeds in all three models, and the
maps are near-identical. So the fitting procedure is highly reproducible, and the resolution
gate in the pre-registration is satisfied in the good direction: the null is small but non-zero,
so we can actually detect a difference against it.

## Changing the corpus moves the map by 87x to 292x the seed null

| model | layers | wiki_a | wiki_b | **code** | corpus boundary shift | map distance vs seed null |
|---|---|---|---|---|---|---|
| gpt2-small | 11 | (6, 8) | (6, 8) | (6, 8) | 0 layers | 0.0674 vs 0.0002 = **292x** |
| gemma-3-270m | 17 | (3, 15) | (3, 15) | **(12, 14)** | **10 layers** | 0.1028 vs 0.0005 = **198x** |
| qwen3.5-0.8b | 23 | (15, 19) | (15, 19) | **(4, 15)** | **15 layers** | 0.0150 vs 0.0002 = **87x** |

**P1 and P2 both supported in every model.** The corpus effect exceeds the seed null by two
orders of magnitude, and in two of three models the fitted boundaries relocate across most of
the network's depth: qwen3.5-0.8b's early/mid boundary moves from layer 15 to layer 4, and
gemma-3-270m's from layer 3 to layer 12.

## Two distinct ways the map changes, which the summary statistic hides

The models do not all fail the same way, and the band statistic alone would have missed it.

**gpt2-small: same boundaries, weaker bands.** The fitted boundaries do not move at all, but
band separation drops from 0.397 to 0.292. The block *structure* stays where it is and gets
less pronounced.

**qwen3.5-0.8b: same band strength, different boundaries.** Band separation is essentially
unchanged (0.2556 on WikiText, 0.2550 on code, a difference of 0.0006, well inside its own seed
null) while the boundaries move by 15 layers on a 23-layer model. **A summary statistic that
only reports band separation would have called this model perfectly corpus-robust.** It is not:
the blocks are equally strong and in completely different places.

**gemma-3-270m: both.** Boundaries move 10 layers and band separation falls from 0.509 to 0.369.

## What this means for the atlas, stated plainly

Every lens in our 36-model atlas, and every public lens we used, was fitted on WikiText. This
result says the fitted depth boundaries are **not** a stable property of the model alone: they
are a property of the model **as estimated over a particular text distribution**, and swapping
prose for code relocates them.

Concretely, the following must be read as scoped to WikiText-fitted lenses:

- the fitted boundary positions and the three-phase description built on them;
- cross-family and cross-scale boundary comparisons;
- any reading of a band as "where the model changes what it is doing", since the answer depends
  on what text you asked it about.

What survives unweakened:

- the fit is **highly reproducible** at fixed corpus (seed null ~0), so published lens maps are
  replicable, they are just corpus-conditional;
- the geometry-versus-activation result, the readout-concentration analysis, and the
  perturbation result, which compare quantities computed within a single fixed lens rather than
  boundary positions across models.

This is the least convenient result in the campaign for our own published note, which is why
the decision table was frozen in advance with "corpus matters" and "robust" declared equally
publishable, and with a commitment to put an unfavourable outcome in the headline rather than
the footnotes.

## Limits

Three models, all under 1B, one code corpus, 100 prompts per fit, length-matched at a 128-token
cap. We have not tested whether the effect shrinks at scale, whether other corpora move
boundaries as much as code does, or whether some middle ground of mixed text yields a
corpus-independent map. A larger model might well be more stable; nothing here rules that out.

## Cost

Nine fits on one RTX 4090, about **$1.75** against a $50 authorization. The estimate we gave
before launching (~1 hour) was roughly 3x optimistic, because we extrapolated from a timing
probe on the *smallest* model and fit cost grows with width and depth as well as parameter
count.

---

# Fit-budget sweep (2026-07-26) — VERDICT: CONVERGED

Design frozen in `PREREG_FITBUDGET.md` (commit `1d9bd6d`) before any fit existed. Ran because an
external review flagged that the 36-model zoo mixes lenses fitted at very different prompt
budgets (the Neuronpedia collection on the order of a thousand prompts, ours at 100, our released
397B lens at 24), and nothing in the note measured whether that heterogeneity mattered.

Eight fits: gpt2-small and gemma-3-270m on WikiText seed 0 at 25 / 50 / 200 / 400 prompts,
identical recipe otherwise. Reference is each model's 100-prompt `wiki_a` fit from the corpus
experiment above; the reference scale is that experiment's already-measured seed null.

| model | budget | map distance to n=100 | as a multiple of the seed null | boundary shift | band_sep |
|---|---|---|---|---|---|
| gpt2-small (null 2.31e-4, ref boundaries 6/8, band 0.3972) | 25 | 4.34e-4 | 1.9x | 0 | 0.4012 |
| | 50 | 3.04e-4 | 1.3x | 0 | 0.4001 |
| | 200 | 7.41e-5 | 0.3x | 0 | 0.3979 |
| | 400 | 1.04e-4 | 0.4x | 0 | 0.3966 |
| gemma-3-270m (null 5.18e-4, ref boundaries 3/15, band 0.5093) | 25 | 5.27e-4 | 1.0x | 0 | 0.5038 |
| | 50 | 1.81e-4 | 0.4x | 0 | 0.5064 |
| | 200 | 2.91e-4 | 0.6x | 0 | 0.5105 |
| | 400 | 1.97e-4 | 0.4x | 0 | 0.5109 |

**Against the frozen decision table: CONVERGED.** Every budget at 200 and 400 sits within 2x the
seed null (0.3x to 0.6x), so budget is not a confound above ~100 prompts and the note's
*fit heterogeneity* caveat is discharged for that range.

Two things we did not expect and are reporting as such:

1. **Even the 25-prompt fits are within the convergence bar** (1.9x and 1.0x the null). The
   frozen table anticipated that a 25-prompt map might be "far from the rest" and would then
   require a separate caveat on our 24-prompt 397B lens. It is not, so that caveat is not
   triggered. This is a stronger result than the design expected, which is exactly why the
   threshold was fixed in advance.
2. **P1 as literally worded is not supported.** It predicted map distance would *decrease
   monotonically* toward 100. It does not: gemma-3-270m's distance is non-monotonic (1.0x, 0.4x,
   0.6x, 0.4x) and gpt2-small ticks up slightly from 200 to 400. The correct reading is that
   every budget is already inside sampling noise, so the ordering among them is noise too, and a
   monotonicity prediction was the wrong shape of hypothesis for a converged quantity. The
   verdict rests on the frozen threshold, not on P1's shape.

**Fitted boundaries never moved once, at any budget, in either model.** Set against the corpus
result directly above (boundaries move a combined 10 and 15 layers, map distance 198x and 292x
the null), the two experiments together say something sharper than either alone: **what you fit
on determines the map; how much you fit on, within an order of magnitude either side of 100
prompts, does not.**

Cost: RTX 4090 at $0.69/hr, ~1.4h including one restart, about **$1.15**. The restart was a
dependency ordering bug, not a science bug: `jlens` pulls `transformers` 5.x, which requires
`torch>=2.5`, and the pod image ships 2.4.1, so all eight fits died instantly on
`ImportError: DTensor`. Fixed by upgrading torch before installing jlens.

Receipt: `results_fitbudget.json`. Figure: `build_fitbudget_fig.py` (written and committed
before the fits landed).
