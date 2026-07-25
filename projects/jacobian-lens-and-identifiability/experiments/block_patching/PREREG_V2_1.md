# Pre-registration v2.1: does the early/middle boundary resist layer swaps?

**Status: FROZEN before any v2.1 run.** Written 2026-07-25. The confirmatory sample below had
no data collected at the time of this commit.

## Where the hypothesis came from (stated plainly)

v2 tested a diffuse "any boundary" effect on olmo-3-1125-32b and returned `beta = +0.220`,
p = 0.18, a null by its own rule. Exploratory analysis C then decomposed that estimate and
found it was **not diffuse**: it sits almost entirely at the first boundary.

| contrast (olmo, exploratory) | beta | p uncorrected |
|---|---|---|
| early <-> middle | +0.4368 | 0.018 |
| middle <-> late | -0.0183 | 0.920 |
| early <-> late | +0.1607 | 0.470 |

That is post hoc, one of three contrasts, and drawn from the model that motivated looking.
Exploration is how a hypothesis gets found; it is not how it gets confirmed. So olmo is
**excluded** from the confirmatory sample below and serves only as the generator. This is the
exploratory/confirmatory split, not a restriction on exploring.

A discordant exploratory datapoint is recorded up front: the same statistic on the CPU pilot
(gemma-3-270m, block balance 0.17) gives `beta = -1.91`, the opposite sign. We expect that
model to be a poor test of a boundary, but it is on the record before v2.1 runs.

## Primary hypothesis (confirmatory)

**P1.** Swapping a layer's state across the **early/middle** boundary damages the output more
than swapping within a block, at matched layer distance and matched mean position:
`beta_b1 > 0`.

Direction is frozen a priori from C2, so P1 is tested **one-sided** at alpha = 0.05; the
two-sided p is also reported.

Model, unchanged from v2 except for the contrast decomposition:

    D(i,j) ~ dummies(|i-j|) + dummies(mean_position)
             + beta_b1 * spans_b1 + beta_b2 * spans_b2 + beta_both * spans_both

`D` is same-prompt cross-layer swap damage, `KL(patched || clean)`, `D(i,i) = 0` by
construction. Null: 1,000 random 3-segmentations with the same block-size multiset.

**Secondary, exploratory, no decision rides on them:** `beta_b2` and `beta_both`.

## Confirmatory sample

New models with the best available block balance, neither used to generate the hypothesis:

| model | hf id | layers | blocks | balance |
|---|---|---|---|---|
| qwen3.5-27b | `Qwen/Qwen3.5-27B` | 63 | 14 / 37 / 12 | 0.32 |
| llama3.3-70b-it | `meta-llama/Llama-3.3-70B-Instruct` | 79 | 37 / 14 / 28 | 0.38 |

**Combined test.** Per-model one-sided permutation p-values are combined by Fisher's method.
P1 is supported if the combined p < 0.05 **and** both per-model `beta_b1` estimates are
positive. Requiring a consistent sign prevents one strong model from carrying a discordant one.

## Gates (unchanged, must pass per model or that model is VOID)

- Sanity: `D(i,i) = 0` for all i, and `D` rising with `|i-j|`.
- Calibration: median `D` over `|i-j| >= 2` within [0.05, 5.0] nats.

## Decision rules (frozen)

| outcome | verdict |
|---|---|
| combined p < 0.05 and both betas > 0 | **CONFIRMED**: the early/middle boundary resists layer swaps |
| combined p >= 0.05, or signs disagree | **NOT CONFIRMED**: olmo's boundary-1 effect does not replicate |
| a gate fails | that model VOID; report the remaining one alone, no combination |

**Conditional extension, frozen now to avoid a post-hoc decision.** The 397B lens model
(`qwen35-397b`, block balance 0.56, the highest fitted separation in the zoo at +0.472) is run
**only if** P1 is CONFIRMED above. It is expensive (~$25-35, download-dominated) and is not
justified by a non-replication. If P1 is not confirmed, the campaign stops and reports.

## Receipt requirement (fixed since v2)

Runs must store **per-prompt** KL, not just the mean, so a prompt-level bootstrap is
computable from the receipt without re-renting the GPU. Analysis C was blocked by exactly this
gap in the v2 receipt.

## Cost

qwen3.5-27b ~$1-2 (54 GB bf16, one 80 GB card); llama3.3-70b-it ~$3-5 (140 GB bf16, one
141 GB card or two 80 GB). Stated again with the actual rate before each pod is created.
