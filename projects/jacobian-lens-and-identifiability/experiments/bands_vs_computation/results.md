# Results: two tests of what the J-space depth bands are

Design frozen in [`PREREG.md`](PREREG.md) (commit `98b1503`) before any data, with Amendment 1
(index alignment for Test B) frozen after a mechanical CPU smoke and before any Test B statistic.

Both tests returned the outcome we predicted in advance, and both predictions were the ones that
cost us more. Neither is a positive result. Together they close one door we had left open and
tighten what the atlas can be said to be.

---

# Test B: lens boundaries do NOT track activation boundaries

**Frozen verdict: BANDS ARE A READOUT PROPERTY.**

| model | L | lens boundaries | activation boundaries | agreement | null median | p |
|---|---|---|---|---|---|---|
| pythia-70m-deduped | 5 | 1 / 2 | 1 / 2 | **0** | 2.0 | 0.318 |
| qwen3.5-0.8b | 23 | 7 / 15 | 6 / 11 | 5 | 6.0 | 0.307 |
| gemma-3-270m | 17 | 3 / 15 | 12 / 14 | 10 | 10.0 | 0.680 |
| gemma-2-9b | 41 | 5 / 19 | 15 / 23 | 14 | 14.0 | 0.814 |
| gemma-2-2b | 25 | 2 / 4 | 19 / 21 | 34 | 19.0 | 1.000 |

Pooled mean normalised agreement **0.501** against a null median of **0.459**: the lens
boundaries are, if anything, marginally *worse* than a random segmentation of the same block
sizes at landing where the activations reorganise. **p = 0.608**, and only 2 of 5 usable models
beat their own null median, neither significantly.

**Anchor gate: PASS, but barely.** gemma-2-9b's activation band separation came back **+0.0812**
against the previously measured +0.110, a difference of 0.0288 against a tolerance of 0.030. It
passes the frozen gate with 4% of the tolerance to spare. We are reporting that margin rather
than just the word PASS, because a gate that squeaks through is weaker evidence of pipeline
health than one that clears comfortably, and this run uses 48 prompts where the original used
more.

## Three models excluded, and this compromised the design

`gpt2-small`, `qwen3-4b` and `llama3.1-8b` produced **degenerate** activation maps: off-diagonal
CKA medians of 0.9993, 0.9990 and 0.9998. Under linear CKA on 4,096 sampled token positions,
their raw activations are very nearly indistinguishable from one layer to the next, so there is
no activation structure for lens boundaries to agree or disagree with. The frozen sanity gate
excludes them, and we followed it.

**This is a flaw in our own decision rule and we are stating it plainly.** The positive verdict
required "pooled `p < 0.05` **and** at least 6 of 8 models individually beating their own null
median". With three models excluded, only five remained, so **BANDS TRACK REPRESENTATIONS became
unreachable regardless of the data**. A frozen rule that can only return one of its outcomes is
not a test.

Two things keep this from invalidating the result. First, the pooled `p` of 0.608 is nowhere near
0.05, so the null verdict is carried by the p-criterion alone and does not depend on the broken
clause. Second, the direction of the point estimate is *away* from agreement, not toward it. But
a reader should treat the verdict as "the data show no agreement" and not as "the frozen test
was passed cleanly", because it was not.

## What this does and does not say

**Says:** on the models where activation CKA has any dynamic range, the depth boundaries fitted
to the *lens* do not mark where the *representations* reorganise. Combined with the earlier
gemma-2-9b dissociation, that supports reading the atlas as a map of the model's first-order
**output linearization** rather than of its representational depth organisation.

**Does not say:** that the bands are meaningless. Agreement between two descriptive maps was
never going to establish causation, and a disagreement does not establish absence. It also does
not say activations lack depth structure in general; in three models our own instrument could
not resolve any, which is a limitation of linear CKA on raw activations as much as a fact about
those models.

---

# Test C: the corpus-instability rescue fails

**Frozen verdict: NO PURCHASE.**

The idea under test was ours, and it was the most promising door left open: because fitted
boundaries move up to a combined fifteen layers with the fitting corpus, every earlier causal
test had large measurement error in its independent variable, which would attenuate a real
effect toward null. If so, boundaries fitted on the corpus the damage is measured on should
predict that damage better than boundaries fitted on the other corpus.

They do not.

| model | damage corpus | matched beta | mismatched beta | gates |
|---|---|---|---|---|
| gpt2-small **(control)** | prose | -0.2211 | -0.2211 | PASS |
| gpt2-small **(control)** | code | -0.1953 | -0.1953 | PASS |
| gemma-3-270m | prose | -0.8262 | -0.4160 | PASS |
| gemma-3-270m | code | -0.9271 | -1.4180 | **FAIL** (median D 6.81, band is [0.05, 5.0]) |
| qwen3.5-0.8b | prose | -0.3727 | -0.5395 | PASS |
| qwen3.5-0.8b | code | -0.2420 | +0.1494 | PASS |

- **C1 (does any segmentation predict damage?): no.** Pooled matched `beta` = **-0.464**, i.e.
  cross-boundary pairs suffer *less* damage than within-block pairs, the opposite of the
  direction three earlier rounds predicted. No individual cell reaches significance against its
  own random-segmentation null; the smallest p is 0.160.
- **C2 (does the fitting corpus matter?): no.** Pooled matched-minus-mismatched = **-0.036**,
  essentially zero and the wrong sign. Boundaries fitted on code predict damage on prose about as
  well as boundaries fitted on prose do, which is to say neither predicts it.

**The mechanical control did its job.** gpt2-small's WikiText and code lenses fit identical
boundaries (6/8), so its matched and mismatched arms are the same regression by construction and
its C2 must be exactly zero. It came out **exactly 0.0**, to floating point. The pipeline is
doing what it claims.

**One cell is out of dynamic range and we are not using it.** gemma-3-270m's damage on code has
median `D` of 6.81 nats against a frozen calibration band of [0.05, 5.0]: swapping activations
between layers of a 270m model on code is a saturating intervention. That cell is reported and
excluded from interpretation, which is what the frozen gate requires.

## Honest reading

We wrote in the pre-registration, before running, that this test is underpowered by construction:
two contrast models, both small, both with lopsided fitted blocks, in exactly the setting the v1
post-mortem identified as weak for detecting a boundary. We committed to calling a null here
**weak evidence**, and we are doing that.

But C2 is the part that was well identified. It is a within-model paired contrast where model,
layer, position, distance and prompts are literally identical across arms, and only the
segmentation label changes. A large effect would have shown up even here, and what we see is
-0.036 with the sign against us. The corpus-instability explanation for the block campaign's
nulls is not supported.

---

# Where this leaves the block question

Adding these two to the three earlier designs: **five designs, eleven models, no evidence that
J-space depth bands partition computation**, and now direct evidence that the bands do not track
where representations reorganise either.

The published position moves from *inconclusive* toward *negative*, but not all the way, and the
reason is specific rather than hedging: Test B's frozen rule could not have returned a positive
once three models were excluded, and Test C was underpowered by its own pre-registration. What
we can now say is stronger than before and still bounded:

> The depth bands are a well-replicated description of the **first-order readout geometry**. They
> are not a map of representational reorganisation, and five increasingly careful attempts have
> failed to show them partitioning computation. We think the most likely account is that the
> atlas describes how the output linearization rotates with depth, which is a real and useful
> thing to have mapped, and is not the same thing as a computational stage.

We are stopping here. The remaining ideas we can think of (ignition-depth alignment, routing
alignment in MoE models) test different constructs rather than repairing this one.

## Cost

Test B: 8 models, forward passes only, RTX A6000 at $0.53/hr, about **$1.10** including three
environment failures documented below. Test C: ran on the same warm pod, marginal cost ~**$0.10**.

## What went wrong on the way, for the next person

1. **`atlas_stage_a` imports `cka_layers`.** The local CPU smoke passed because both live in the
   same directory here; shipping only the one file broke every model on the pod. `fitted_seg` is
   now vendored into the runner with an assert that it stays identical to the atlas original.
2. **qwen3.5 silently ignores `output_hidden_states` at construction** and returns `None`. The
   flag now goes on the forward call, and a `None` raises immediately instead of crashing three
   lines later in a list comprehension.
3. **The `DTensor` trap, hit for the second time in one day.** `pip install -U transformers`
   pulls 5.x, which needs torch >= 2.5, onto an image shipping 2.4.1. It is written down in our
   own checkpoint file and we still did it. Upgrade torch first.
4. **Test C could not run on this box at all.** 7.6GB RAM with 5GB already in use, against a
   617MB logits tensor per forward pass and 145 passes per cell. It died silently because the
   launching command piped through `tail`, which buffers until exit. Moved to the warm GPU pod,
   where all six cells took minutes.

---

# Test B re-run (2026-07-26): verdict unchanged, and the instrument understood

Design frozen in [`PREREG_B2.md`](PREREG_B2.md), with full disclosure that we had already seen
the first run's results. Twelve models, four analysis cells reported side by side as committed.

**Anchor gate: PASS, exactly.** gemma-2-9b's raw activation band separation reproduced at
`+0.0812` against a target of `+0.0812`, using **0% of the tolerance**. Contrast this with the
first run's anchor, which used 96% of its tolerance; the pipeline reproduces itself precisely.

## All four cells give the same verdict

| map | gate | usable models | observed | null median | p | beat own null |
|---|---|---|---|---|---|---|
| raw | old (median) | 8 | 0.683 | 0.539 | 0.865 | 3/8 |
| raw | **repaired (range)** | **10** | **0.560** | **0.541** | **0.569** | **5/10** |
| standardised | old | 12 | 0.730 | 0.548 | 0.960 | 3/12 |
| standardised | repaired | 12 | 0.730 | 0.548 | 0.960 | 3/12 |

**Verdict under every cell: BANDS ARE A READOUT PROPERTY.** The conclusion does not depend on the
defective gate, and it survives the instrument being fixed. The repaired gate does exactly what
it should: it restores `gpt2-small`, `qwen3-4b` and `qwen2.5-7b-it`, drops `gemma-2-2b` and
`llama3.1-8b`, and moves the observed agreement from 0.683 to 0.560, closer to the null but still
not beating it.

## The standardisation check: hypothesis confirmed, conclusion unchanged

Per-dimension standardisation transforms the maps exactly as predicted. `llama3.1-8b`'s
off-diagonal range goes from **0.002 to 0.839**; `gpt2-small`'s from 0.136 to 0.514. Saturated
activation CKA really was residual-norm growth dominating the Gram matrix, and **no model is
degenerate once standardised**, which is why all twelve become usable.

And it does not rescue boundary agreement. It makes it **worse**: 0.730 observed against a 0.548
null, with only 3 of 12 models beating their own. Giving the activation map more dynamic range to
express structure does not make the lens boundaries any better at finding it.

## The claim we said this could overturn survives, and strengthens

We pre-registered that this check could overturn our own Gemma headline, since that claim rests
on this instrument. It does not. gemma-2-9b's activation band separation under standardisation is
**+0.2086**, against a lens band separation of +0.005. The dissociation between structured
activations and a flat lens is **larger** under the corrected measurement, not smaller. We would
have reported the opposite with the same prominence.

## What the re-run changes about the first run

The first Test B was right about its conclusion and wrong about its method in two ways we have
now fixed and documented: a usability gate on the wrong statistic, and a decision rule whose
positive branch became unreachable after any exclusion. TJ asked for this re-run over our
recommendation not to bother. The recommendation was wrong: the gate defect was real, it would
have shipped, and the conclusion is now supported by twelve models across four analysis choices
instead of five models under a broken rule.

## Cost

Twelve models, forward passes only, shared pod with the ignition test. About **$2** total.

---

# Correction (2026-09-05): Test B used own-vocabulary lens boundaries; the design says shared

`PREREG.md` and `PREREG_B2.md` compare activation boundaries to the boundaries of each model's
**shared-vocabulary** lens map, and name `../jspace_atlas/atlas_out/<slug>.npz` as that cached map.
That file is Stage A's **own-vocabulary** map (`atlas_stage_a.py`, first line of its docstring); the
shared-probe maps live in `atlas_out/shared_maps/`. Both Test B runs, and the ignition test, took
their lens boundaries from the own-vocabulary file. For 5 of the 12 re-run models the two probes
place the boundaries 3 to 9 layers apart (gemma-2-2b 2/4 vs 2/7, llama3.1-8b 4/17 vs 13/17,
olmo-3-1025-7b 9/18 vs 15/18, qwen2.5-7b-it 4/6 vs 2/4, qwen3.5-0.8b 7/15 vs 2/15).

`analyze_B2.py` now fits the lens boundaries on the shared maps (the own-vocabulary values stay in
the output for the record; `--own-vocab-boundaries` reproduces the superseded numbers exactly).
The activation maps are unchanged (cached in `outB/`), so no GPU was needed.

| map | gate | usable models | observed | null median | p | beat own null | superseded (own-vocab): obs / null / p |
|---|---|---|---|---|---|---|---|
| raw | old (median) | 8 | 0.715 | 0.554 | 0.881 | 2/8 | 0.683 / 0.539 / 0.865 |
| raw | **repaired (range)** | **10** | **0.601** | **0.568** | **0.613** | **4/10** | 0.560 / 0.541 / 0.569 |
| standardised | old | 12 | 0.743 | 0.561 | 0.965 | 4/12 | 0.730 / 0.548 / 0.960 |
| standardised | repaired | 12 | 0.743 | 0.561 | 0.965 | 4/12 | 0.730 / 0.548 / 0.960 |

**Verdict under every cell, on the pre-registered probe: BANDS ARE A READOUT PROPERTY.** The
conclusion is unchanged; the numbers the note quotes must be these, and the deviation is
disclosed. Anchor gate unaffected (activation side). Superseded output:
`results_B2_ownvocab_boundaries.json`.

One thing this exposed that the note did not say: the fitted three-segmentation is poorly
identified for about a third of the zoo. On the shared maps, the set of segmentations within 5% of
the objective's range of the optimum spans 0.4 to 0.85 of the depth for at least one boundary in
gemma-3-27b-it, qwen3-14b, qwen3-1.7b, llama3.3-70b-it, gemma-4-e2b, gemma-4-e4b, gemma-3-270m,
gemma-2-9b and gemma-3-4b-it (the 397B is well identified: 0.08 and 0.07 of depth). Any test that
uses "the boundary" as a variable is, for those models, comparing noise with noise, which biases
Test B toward its null. That is a limitation of Test B we had not stated, and a reason to add an
identifiability gate to any future boundary-based design.

# Test C after the corpus correction (2026-09-05): no contrast left to test

Test C took its segmentation labels from `../corpus_dependence/results.json`. That file was
produced with a statistic that is not CKA (see the correction at the top of the corpus ledger).
Under the pre-registered statistic the WikiText and code boundaries are identical for gpt2-small
(2/4) and gemma-3-270m (3/15) and differ by one layer for qwen3.5-0.8b (6/15 vs 5/15). Re-running
`analyze_C.py --runs outC` on the corrected labels: the mechanical control still returns
C2 = 0.0 exactly (VOID = False), gemma-3-270m becomes a second identical-label model (C2 = 0 by
construction), and the only contrast model gives C2 = +0.012 (matched minus mismatched), with
pooled matched beta -0.500. The superseded values were C2 = -0.036 and beta -0.464
(`results_C_legacy_selfgram_boundaries.json`).

The honest status of Test C is therefore **moot rather than null**: its premise, that the fitting
corpus relocates the boundaries by 10 to 15 layers, was an artifact, so the crossed design has
almost nothing to cross. It should no longer be cited as evidence against the corpus-instability
explanation, and the "five designs" count in the note should describe it that way. The
gemma-3-270m code cell still fails its calibration gate (median D 6.806) and stays excluded.
