# Pre-registration v2: do depth blocks behave as interchangeable representation formats?

**Status: FROZEN before any v2 run.** Written 2026-07-25, after v1 refuted its own hypothesis
and exposed two design faults. No v2 data existed when this was committed.

## Why v2 exists

v1 ([`PREREG.md`](PREREG.md), results in [`results.md`](results.md)) asked whether activation
patches transfer worse across a block boundary. P1 was **not supported** in either model
(gemma-3-270m beta = -0.002, p = 0.98; qwen3-8b beta = +0.148, p = 0.015, the wrong sign).
Three faults, all diagnosed post hoc and all fixed here:

1. **Saturation.** Full-residual patching drove `E` to 0.65-0.99 nearly everywhere, leaving
   no dynamic range for a boundary effect.
2. **Uncontrolled absolute position.** v1 absorbed layer *distance* but not layer *position*.
   With lopsided blocks these are confounded, and the apparent positive effect in qwen3-8b is
   best explained by within-block pairs concentrating in the low-`E` early block.
3. **Lopsided blocks.** Both v1 models were effectively one large block plus slivers
   (3/12/2 and 19/3/13), a weak setting for testing a boundary at all.

## New measure: same-prompt cross-layer swap damage

v1 conflated two things: whether a representation is in a compatible *format*, and whether it
carries a different prompt's *content*. v2 removes the content dimension entirely.

For a single prompt, capture the full residual stream at layer `i`, then re-run the **same
prompt** and substitute that state at layer `j`. Nothing semantic changes; the only question
is whether layers `j..L` can consume a layer-`i` representation of the same input.

    D(i,j) = KL( p_patched(next token) || p_clean(next token) )

`D(i,i) = 0` exactly, by construction, which is the sanity gate. `D` has an unbounded top end
and no ceiling, which fixes fault 1.

## Design

**Primary (confirmatory) model.** Over ordered pairs `i != j`:

    D(i,j) ~ dummies(|i-j|) + dummies(mean_position) + beta * crosses(i,j)

`mean_position = round((i+j)/2)`, which fixes fault 2. Both controls are symmetric under
`(i,j) -> (j,i)`, so `beta` is the boundary contrast averaged over swap direction; v1 showed
that a signed-offset control is algebraically degenerate against a symmetric `crosses`, so it
is not used and not claimed as a control.

**Prediction P1 (primary).** `beta > 0`: swapping across a block boundary damages the output
more than swapping within a block, at matched distance and matched position.

**Control.** Random 3-segmentation null with the same block-size multiset, 1,000 draws, as in
v1. Two-sided permutation p-value.

**Sanity gate (must pass or VOID).** `D(i,i) = 0` for all `i`, and `D` must increase with
`|i-j|`.

**Calibration gate (new, fixes fault 1).** Before any confirmatory reading, the median `D`
over `|i-j| >= 2` must lie in `[0.05, 5.0]` nats. Outside that band the measure is saturated
or dead and the run is VOID for calibration, to be retuned on the free pilot only.

**Target model (fixes fault 3).** Block balance, defined as `min(block sizes)/max(block
sizes)`, computed from the atlas segmentations before this experiment:

| model | fitted_sep | blocks | balance | role |
|---|---|---|---|---|
| olmo-3-1125-32b | +0.283 | 25 / 14 / 24 | **0.56** | confirmatory target |
| gemma-3-270m | +0.227 | 3 / 12 / 2 | 0.17 | free CPU pilot (mechanics + calibration only) |

No model below ~27B has balanced blocks (best affordable is llama3.1-8b at 0.29), which is
itself worth reporting: the clean three-phase segmentation is largely a large-model property.
The pilot is explicitly **not** a fair test of P1 and no verdict will be read from it; it
exists to validate the pipeline and calibrate `D`.

## Decision rules (frozen)

| outcome on olmo-3-1125-32b | verdict |
|---|---|
| `beta > 0`, p < 0.05 | **CONFIRMED**: boundaries mark real format changes |
| `beta` not distinguishable from the null | **NULL**: blocks are geometric, not computational |
| `beta < 0`, p < 0.05 | **REVERSED**: report as such, do not reinterpret post hoc |
| sanity or calibration gate fails | **VOID**: retune on the pilot, re-run |

A null is reported as prominently as a confirmation, and combined with v1's null it would be
a reasonably strong negative result about the causal status of J-space depth bands.

## Materials and cost

40 prompts (the v1 pairs used as independent texts, no source/target roles). Greedy, no
sampling. Full residual, all positions.

1. Pilot: gemma-3-270m, this box, CPU. **$0.**
2. Confirmatory: olmo-3-1125-32b, bf16 ~64 GB, one H100-80 or 2xA100-40, inference only.
   63 layers means 3,969 patched forwards, minutes of compute; the cost is the ~64 GB
   download. **Estimated $3-6**, and it will be stated again with the actual rate before
   any pod is created.

## What v2 still cannot establish

That a boundary marks a format change is not the same as knowing what either side computes.
Functional labels still need ablation-by-capability and per-block feature analysis.
