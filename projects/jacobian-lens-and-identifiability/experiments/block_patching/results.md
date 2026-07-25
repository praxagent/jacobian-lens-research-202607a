# Results: are J-space depth blocks causal boundaries?

Design frozen in [`PREREG.md`](PREREG.md) before any run (commit `8705791`), with Amendment 1
(full-residual intervention) added after the sanity gate failed and before any boundary
statistic was computed.

## Runs actually executed

| model | where | cost | layers | pairs | sanity gate `E(i,i)` |
|---|---|---|---|---|---|
| gpt2-small | this box, CPU | $0 | 12 | 16/17 | 1.000 PASS (mechanics smoke) |
| gemma-3-270m | this box, CPU | $0 | 18 (lens 17) | 20/20 | 1.000 PASS |
| qwen3.5-0.8b | this box, CPU | $0 | 24 (lens 23) | 18/18 | 1.000 PASS |
| qwen3-8b | RTX 4090, RunPod | ~$0.10 | 36 (lens 35) | 18/18 | 1.000 PASS |

A first 3090 pod (`12iu2cy9dx9hw9`) was terminated unused: CUDA reported `is_available()`
True but every allocation failed with "device busy or unavailable". Both pods are terminated
and verified gone.

## Confirmatory result (the frozen analysis)

`E(i,j) ~ per-distance dummies(|i-j|) + beta * crosses(i,j)`, against a random
3-segmentation null with the same block-size multiset, 1,000 permutations.

| model | fitted blocks | beta | null mean / sd | p | frozen verdict |
|---|---|---|---|---|---|
| gemma-3-270m | 3 / 12 / 2 | **-0.0023** | -0.0004 / 0.0353 | 0.978 | P1 NOT SUPPORTED |
| qwen3.5-0.8b | 2 / 13 / 8 | **+0.0029** | -0.0007 / 0.0215 | 0.958 | P1 NOT SUPPORTED |
| qwen3-8b | 19 / 3 / 13 | **+0.1476** | +0.0012 / 0.0659 | 0.015 | P1 NOT SUPPORTED |

**P1 predicted `beta < 0`** (cross-boundary patches transfer worse). It is not supported in
any of the three models. Two of three models (both small, run free on CPU) give a clean null;
qwen3-8b alone is significant, and in the **opposite** direction.

**Headline: the pre-registered hypothesis is refuted.** On this design, block boundaries do
not behave like barriers to activation transfer.

## Do not read the positive qwen3-8b effect as a finding

It is most likely an artifact of two design limitations we can now see clearly, and we say so
rather than promoting it.

**1. Saturation.** Full-residual patching (Amendment 1) is a very strong intervention: mean
`E` is 0.65-0.99 nearly everywhere, so there is little dynamic range in which a boundary
effect could show up. Amendment 1 fixed a too-weak intervention and overshot into a too-strong
one.

**2. Absolute position is not controlled, and is confounded with `crosses` here.** The frozen
model absorbs layer *distance* but not layer *position*. With qwen3-8b's lopsided blocks
(19/3/13) that matters: within-block pairs are dominated by the 19-layer early block, which
sits in the low-`E` region, while cross-block pairs cluster around the late boundary region
where `E` is high. Mean `E` by block pair shows it directly:

|  | -> block 0 | -> block 1 | -> block 2 |
|---|---|---|---|
| **block 0 ->** | +0.832 | +0.825 | +0.649 |
| **block 1 ->** | +0.857 | +0.989 | +0.808 |
| **block 2 ->** | +0.856 | +0.995 | +0.931 |

The single worst cell is early-source into late-target (+0.649), which is a *direction and
position* effect, not a boundary effect. A signed-offset robustness check was attempted and is
**degenerate**: because `crosses(i,j) == crosses(j,i)`, the signed model returns an identical
beta (+0.1476), so it does not test the asymmetry at all.

Direction asymmetry is real and large: mean `E` at offset -20 (early state into a late layer)
is +0.561, against +0.824 at offset +20.

## Honest verdict

**NULL for P1, with acknowledged limited power.** The CKA depth bands are not shown to
partition computation. We are *not* claiming the stronger result that they definitely do not,
because this design had a saturating intervention and an uncontrolled position confound, and
the two models' fitted segmentations are both lopsided (one big block plus slivers), which is
a weak setting for testing a boundary at all.

## What a v2 would need

1. **A non-saturating intervention.** Patch a subspace or interpolate (`alpha * source + (1-alpha) * target`)
   and pick alpha to put mean `E` near 0.5, restoring dynamic range.
2. **Position controls.** Add absolute-position terms, or match cross and within pairs on both
   distance *and* mean layer index.
3. **Models with balanced blocks.** Both models here are effectively one large block plus
   slivers. gemma-3-27b (+0.406) or the 397B have cleaner 3-way structure.
4. **A behavioural task with headroom**, since single-token factual recall saturates easily.

Cost of everything above: **~$0.10**. A v2 is affordable; it is a design problem, not a budget
problem.

---

# v2 results: same-prompt swap damage on a balanced-block model

Design frozen in [`PREREG_V2.md`](PREREG_V2.md) (commit `86aa050`) before any v2 run.

| model | where | cost | blocks | balance | gates |
|---|---|---|---|---|---|
| gemma-3-270m | this box, CPU | $0 | 3/12/2 | 0.17 | PASS (pilot: mechanics + calibration only) |
| olmo-3-1125-32b | A100 80GB, RunPod | ~$1 | 25/14/24 | **0.56** | PASS |

Calibration, the gate v1 lacked: median `D` over `|i-j| >= 2` is **0.164 nats** on olmo (2.29 on
the pilot), inside the frozen [0.05, 5.0] band. The measure has real dynamic range, and
`D(i,i) = 0` exactly. v1's saturation problem is fixed.

## Confirmatory result

`D(i,j) ~ dummies(|i-j|) + dummies(mean_position) + beta * crosses(i,j)`, 1,000-draw
random-3-segmentation null.

| quantity | value |
|---|---|
| beta (distance **and** position absorbed) | **+0.2203** |
| beta without position dummies (reference) | +0.0524 |
| random-boundary null | mean -0.0014, sd 0.1465 |
| two-sided p | **0.181** |
| pairs | 3,906 |

**Frozen verdict: NULL.** P1 predicted `beta > 0` at p < 0.05. The point estimate *is* in the
predicted direction, about 1.5 sd above the null, but it does not clear the threshold.

## This is a different null from v1, and we say so carefully

In v1's small models beta was indistinguishable from zero (-0.002, +0.003). Here it is
**+0.22 in the predicted direction** and simply not significant. Two things follow.

First, the position control now works in the opposite direction from v1. Adding mean-position
dummies *raises* the estimate from +0.052 to +0.220, where in v1 position inflation was our
explanation for a spurious positive. That is the control behaving as a control should: it
removes a confound rather than manufacturing an effect.

Second, the experiment is **underpowered at one model**. The null's spread (sd 0.147) is
driven by segmentation-level variability, not by the 3,906 pairs, so pair count does not buy
power; **models** do. A modest real boundary effect of this size would need several balanced
models to detect.

## Combined position after v1 and v2

Four models, two independent designs, no significant evidence that J-space depth bands
partition computation. We are **not** claiming they definitely do not: v2's point estimate is
positive and the design that could detect a modest effect has been run exactly once. The
honest statement is that the bands are a well-replicated description of representational
**geometry**, and that two causal probes have failed to show them partitioning **computation**,
with the better-designed probe leaving a suggestive but unresolved positive.
