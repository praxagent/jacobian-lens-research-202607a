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
