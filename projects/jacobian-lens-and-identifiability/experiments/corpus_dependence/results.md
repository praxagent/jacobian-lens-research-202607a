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
