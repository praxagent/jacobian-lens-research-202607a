# Pre-registration: corpus dependence at 8B (llama3.1-8b)

Frozen 2026-09-05 before any 8B fit exists. Extends `PREREG.md` (three models under 1B) to one
mid-size model after the 2026-09-05 statistic correction, to test whether the corrected picture
(the map moves with corpus, the boundaries barely do) holds at 8B. Same design, same recipe.

## Design (unchanged from PREREG.md)

Model `meta-llama/Llama-3.1-8B` (in the zoo, public lens boundaries (13, 17) on the shared probe,
**identified** by the B3 rule at spreads 0.13 / 0.13). Three lenses with `fit_our_own/fit_lens.py`,
100 prompts each, `--max-seq-len 128 --match-length --dim-batch 8`:
`wiki_a` (WikiText seed 0), `wiki_b` (WikiText seed 1), `code` (codeparrot-clean seed 0).
Measures on the shared-vocabulary probe with the corrected statistic (`analyze.py` linear CKA of
readout geometry, identity-checked against `common.cka.linear_cka`): boundary shift, map distance
(`1 - CKA` between off-diagonal map profiles), band shift; seed null = `wiki_a` vs `wiki_b`.

**Unembedding path.** Llama 3.1 is untied. The probe rows must come from `lm_head.weight`, as
the atlas's `unembed.py` does, not from the input embedding; the helper
`run_geometry_causality.probe_rows_inline` matches embedding-tensor names only and must not be
used for this model (flagged by the atlas-tool build on 2026-09-05). The analysis run will assert
that the tensor used is `lm_head.weight`.

## Predictions

- **P1** (map): corpus map distance exceeds the seed null by more than 10x. Predicted to hold.
- **P2** (boundaries): the corpus boundary shift is at most 2 layers, and at most the larger of
  2 and the seed boundary shift. Predicted to hold (this is the corrected small-model picture).
- **P3** (descriptive): band separation on code differs from WikiText by more than the seed band
  shift. No verdict attached.
- **Anchor**: our `wiki_a` fit's shared-probe map must lie within 0.01 map distance of the public
  Neuronpedia llama3.1-8b shared map (the small models landed at 0.001 to 0.003), or the fit
  recipe is not comparable and the run is reported as failed rather than interpreted.

## Decision table

| outcome | statement in the note |
|---|---|
| P1 and P2 hold | the corrected corpus picture replicates at 8B: map moves, boundaries do not |
| P1 holds, P2 fails | at 8B the corpus does move the boundaries; the small-model result was scale-limited |
| P1 fails | the corpus effect on the map shrinks with scale; reported as such |
| anchor fails | run reported as failed; no corpus statement at 8B |

## Cost and gating

One RTX A6000 (48 GB, $0.33/hr), shared with the qwen3.5-0.8b float32 perturbation re-run. The
three fits are gated on a 2-prompt timing probe: if three 100-prompt fits extrapolate to more than
`MAX_FIT_HOURS` (8) they are skipped and the probe time is reported instead. Worst case about $3.
