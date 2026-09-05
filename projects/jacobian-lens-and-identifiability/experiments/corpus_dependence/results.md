# Results: a J-lens depth map is substantially a property of the fitting corpus

Design frozen in [`PREREG.md`](PREREG.md) before any fit (commit `4b4f7d5`). Nine fits, three
models, identical recipe, on one RTX 4090. **Verdict: CORPUS MATTERS, 3 of 3 models** (on the
map-distance measure; see the correction below for what changed about the boundaries).

> **CORRECTION, 2026-09-05. The map statistic `analyze.py` used before this date was not CKA.**
> It built each layer's d x d readout covariance `G_l = J_l^T M J_l` and scored a layer pair with
> `<G_i, G_j>_F / (|G_i| |G_j|)`, the cosine between two self-covariances. Linear CKA of the readout
> geometries `D_l = U_c J_l` is `|J_j^T M J_i|_F^2 / (|J_i^T M J_i|_F |J_j^T M J_j|_F)`, which needs
> the cross-gram. `PREREG.md` defines map distance as `1 - CKA` between the "layer-by-layer
> readout-geometry maps", i.e. the atlas's maps, so the implementation deviated from the
> pre-registered statistic. The two statistics give very different maps from the same lens (our
> gpt2-small `wiki_a` fit: off-diagonal range [0.942, 0.997] under CKA, [0.027, 0.820] under the
> old formula), and every fitted boundary reported in the superseded sections below was fitted on
> the wrong map. The analyzer now computes the pre-registered statistic (`cka_from_readout`),
> asserts at run time that it equals `common.cka.linear_cka` on explicit geometries (agreement to
> 1e-6 on all three models), and keeps the old formula behind `--legacy`; the superseded output is
> preserved as `results_legacy_selfgram.json`. Caught by a fresh reviewer noticing that the public
> Neuronpedia fit and our own fit of the same model disagreed by a map distance of 0.45 under the
> old statistic while their per-layer geometries agree at CKA 0.9992 (`fit_our_own/results.md`);
> under CKA the two fits agree to 0.003, i.e. seed-null scale.

## Corrected results (2026-09-05, linear CKA, shared probe, same nine fits)

| model | layers | wiki_a | wiki_b | code | seed shift | corpus shift | map distance, seed | map distance, corpus | ratio |
|---|---|---|---|---|---|---|---|---|---|
| gpt2-small | 11 | (2, 4) | (2, 4) | (2, 4) | 0 | 0 | 0.0028 | 0.2390 | **84x** |
| gemma-3-270m | 17 | (3, 15) | (3, 12) | (3, 15) | **3** | 0 | 0.0001 | 0.1193 | **888x** |
| qwen3.5-0.8b | 23 | (6, 15) | (6, 15) | (5, 15) | 0 | 1 | 0.0005 | 0.0440 | **86x** |

Band separation (fixed thirds, CKA map), wiki_a / wiki_b / code: gpt2-small 0.0149 / 0.0152 /
0.0050; gemma-3-270m 0.1343 / 0.1379 / 0.0241; qwen3.5-0.8b 0.1130 / 0.1112 / 0.1232.

**What survives.** P1, the map-distance test: the corpus effect exceeds the seed null in 3 of 3
models, by 84x to 888x, so the frozen verdict **CORPUS MATTERS** stands on the pre-registered
measure. The map really does move when the lens is fitted on code, and for gpt2-small it moves
more than the superseded numbers said.

**What does not survive.** P2 and the whole boundary story. Under the pre-registered statistic the
fitted boundaries move by 0, 0 and 1 layers when the corpus changes, and by 0, 3 and 0 layers when
only the WikiText sample changes. The "combined 10 and 15 layers" relocation, the "qwen3.5-0.8b
keeps its band strength and moves its blocks" reading, the "summary statistic would have hidden
this" section, and the claim that a seed resample leaves boundaries "on exactly the same layers"
were all artifacts of fitting boundaries on the wrong map. The corrected picture is less dramatic:
fitting on code changes how strong the bands are (gemma-3-270m 0.134 to 0.024, gpt2-small 0.015 to
0.005) and reshapes the off-diagonal map, but leaves the boundary positions about where seed noise
already puts them.

**Consequences.** Test C in `../bands_vs_computation` took its segmentation labels from this
file's boundaries; with corrected boundaries two of its three models have identical labels in both
arms and the third differs by one layer, so Test C no longer has a contrast to test (see its
ledger). The fit-budget sweep below was re-analysed with the corrected statistic (second
correction block, further down). The note's summary sentence about boundaries moving "a combined
fifteen layers" is withdrawn. Figures `corpus-dependence` and `fit-budget` were regenerated from
the corrected receipts and their titles and alt text now derive from the data instead of carrying
the old story hardcoded.

The sections that follow, up to the fit-budget sweep, are the **superseded** pre-correction
analysis, kept verbatim for the record.

## [SUPERSEDED 2026-09-05] The seed null is essentially zero, which is what makes the rest meaningful

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

## [SUPERSEDED 2026-09-05] Changing the corpus moves the map by 87x to 292x the seed null

| model | layers | wiki_a | wiki_b | **code** | corpus boundary shift | map distance vs seed null |
|---|---|---|---|---|---|---|
| gpt2-small | 11 | (6, 8) | (6, 8) | (6, 8) | 0 layers | 0.0674 vs 0.0002 = **292x** |
| gemma-3-270m | 17 | (3, 15) | (3, 15) | **(12, 14)** | **10 layers** | 0.1028 vs 0.0005 = **198x** |
| qwen3.5-0.8b | 23 | (15, 19) | (15, 19) | **(4, 15)** | **15 layers** | 0.0150 vs 0.0002 = **87x** |

**P1 and P2 both supported in every model.** The corpus effect exceeds the seed null by two
orders of magnitude, and in two of three models the fitted boundaries relocate across most of
the network's depth: qwen3.5-0.8b's early/mid boundary moves from layer 15 to layer 4, and
gemma-3-270m's from layer 3 to layer 12.

## [SUPERSEDED 2026-09-05] Two distinct ways the map changes, which the summary statistic hides

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

## [SUPERSEDED 2026-09-05 where it concerns boundaries] What this means for the atlas, stated plainly

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

# Fit-budget sweep (2026-07-26) — VERDICT: CONVERGED (re-analysed 2026-09-05: still CONVERGED, but the 25-prompt caveat is now triggered)

> **CORRECTION, 2026-09-05.** Same statistic defect as the corpus experiment (correction block at
> the top of this file); re-analysed with linear CKA on the same fits, same reference, same frozen
> rules. Superseded output preserved as `results_fitbudget_legacy_selfgram.json`.
>
> | model | budget | map distance to n=100 (CKA) | x seed null | boundary shift | band_sep |
> |---|---|---|---|---|---|
> | gpt2-small (seed null 2.84e-3, ref boundaries 2/4, band 0.0149) | 25 | 5.02e-3 | 1.8x | **4** (4/6) | 0.0158 |
> | | 50 | 2.26e-3 | 0.8x | 0 | 0.0155 |
> | | 200 | 3.95e-4 | 0.1x | 0 | 0.0147 |
> | | 400 | 7.05e-4 | 0.2x | 0 | 0.0144 |
> | | 1000 | 7.20e-4 | 0.3x | 0 | 0.0145 |
> | gemma-3-270m (seed null 1.34e-4, ref boundaries 3/15, band 0.1343) | 25 | 1.91e-3 | **14.2x** | **3** (3/12) | 0.1371 |
> | | 50 | 4.89e-4 | **3.6x** | **3** (3/12) | 0.1342 |
> | | 200 | 1.31e-4 | 1.0x | 0 | 0.1276 |
> | | 400 | 1.40e-4 | 1.0x | 0 | 0.1302 |
> | | 1000 | 7.3e-5 | 0.5x | 0 | 0.1342 |
>
> **Against the frozen decision table: still CONVERGED.** Every budget at 200 and above sits within
> 2x the seed null (0.1x to 1.0x), so the discharge of the fit-heterogeneity caveat for the range
> the public ~1,000-prompt lenses occupy stands. **But the 25-prompt caveat is now TRIGGERED**:
> gemma-3-270m at 25 prompts sits 14.2x the seed null and at 50 prompts 3.6x, with its boundaries
> at both budgets on 3/12 rather than the reference 3/15 (the same alternative its seed resample
> produces, so the boundary itself is not well identified for this model); gpt2-small's 25-prompt
> boundaries also move (4/6 vs 2/4). Under the frozen rule "25-prompt map far from the rest -> the
> 24-prompt 397B lens is caveated", **the caveat on our released 397B lens applies**, and the two
> paragraphs below that said it did not are withdrawn. Corrected statement: fitting budget is
> inside the seed null from about 200 prompts up (from 50 for gpt2-small), so the public lenses
> and our 100-prompt fits are comparable; a 24-prompt fit is not shown to be, and the 397B map
> should be read with that caveat. Also withdrawn: "fitted boundaries never moved once, at any
> budget" (they moved at 25 and 50 prompts). The text below is the superseded analysis, kept for
> the record.

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
| | **1000** | **9.42e-5** | **0.4x** | **0** | **0.3969** |
| gemma-3-270m (null 5.18e-4, ref boundaries 3/15, band 0.5093) | 25 | 5.27e-4 | 1.0x | 0 | 0.5038 |
| | 50 | 1.81e-4 | 0.4x | 0 | 0.5064 |
| | 200 | 2.91e-4 | 0.6x | 0 | 0.5105 |
| | 400 | 1.97e-4 | 0.4x | 0 | 0.5109 |
| | **1000** | **1.81e-4** | **0.3x** | **0** | **0.5147** |

**Against the frozen decision table: CONVERGED.** Every budget at 200 and above sits within 2x
the seed null (0.3x to 0.6x), so budget is not a confound above ~100 prompts and the note's
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

## Amendment 1 run (2026-07-26): 1,000 prompts — VERDICT UNCHANGED, CONVERGED

Frozen in `PREREG_FITBUDGET.md` Amendment 1 (commit `e09bb2e`) before the fits, and explicitly
disclosed there as **not blind**: the 25/50/200/400 results had been fully inspected first.

The original sweep stopped at 400 prompts, which discharged the fit-heterogeneity caveat over a
range that **excludes the budget the public lenses actually use**. The Neuronpedia collection
fits on the order of a thousand WikiText prompts, and those lenses are 35 of the 36 rows in our
zoo, so a discharge that stops at 400 does not cover the population the caveat was raised about.

Two fits at n=1,000, same recipe, same reference, same analyzer, threshold deliberately
unchanged at 2x the seed null. Results are folded into the table above:

- gpt2-small: map distance **9.42e-5**, **0.4x** the seed null, boundaries unmoved at 6/8.
- gemma-3-270m: map distance **1.81e-4**, **0.3x** the seed null, boundaries unmoved at 3/15.

Both are comfortably inside the bar, so the verdict stands and now covers the range the public
lenses occupy. Across all five budgets (25 to 1,000, a **40-fold** span) neither model's fitted
boundaries moved a single layer, and no budget exceeded 1.9x the seed null.

We had pre-committed that a miss here would **withdraw** the note's discharge of the caveat and
force us to report that the public lenses sit in a regime our own fits never reach. It did not
miss, but the commitment is on the record either way.

Residual scope, unchanged and still worth stating: two small models, one corpus, one seed. The
sweep says budget does not matter for these two models on WikiText; it does not say budget cannot
matter for a 70B model or on a different distribution.

Cost: RTX A5000 at $0.27/hr, ~2.75h, about **$0.74**. Running total for the fit-budget question,
both runs, about **$1.89**.

## 8B extension (2026-09-05): gate override recorded

`PREREG_8B.md` (commit `73fb7cb`) gated the three llama3.1-8b fits on a timing probe (skip if three
100-prompt fits extrapolate to more than 8 h). The probe's conservative rule attributed the whole
warm 2-prompt run (270 s, which includes model load) to the prompts and returned 11.25 h, so the
script skipped the fits. After inspecting the probe (about 3 h per fit realistic) the lead
overrode the gate and launched the same three fits directly (`/workspace/llama_fits.sh` on pod
`aln7lne2jgdcnv`, RTX A6000 at $0.53/hr, about $5 total). Nothing about the design, the
predictions or the analysis changed; only the cost gate was overridden, and it is recorded here
before any 8B result exists.
