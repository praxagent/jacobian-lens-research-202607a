# Pre-registration: does J-space geometry have local causal purchase?

**Status: FROZEN before any run.** Written 2026-07-25. No data collected at commit time.

## Why this experiment and not the previous ones

Two prospective design audits killed two predecessors (`../block_patching/`). The final and
decisive objection: for small perturbations, output KL is approximately a quadratic function of
the residual update and the **downstream logit Jacobian**, while CKA boundaries are themselves
derived from Jacobian readout geometry, so a boundary-aligned effect could arise from shared
mathematical inputs with no computational structure. That confound poisons positive results,
not just nulls.

This design defuses it by making the **contrast internal**: two perturbation arms at the **same
layer**, **same input**, and **exactly equal norm**, differing only in *direction*. Update
magnitude, depth position, architecture, and generic sensitivity are shared by both arms and
therefore cancel.

It also answers a more basic question than "are the bands computational stages", and one the
whole J-lens enterprise assumes without testing: **does a corpus-averaged first-order lens
predict causal effects at individual inputs?**

## Construct

A Jacobian lens stores `J_l = E[d h_final / d h_l]`, an average over natural text. Every
interpretation built on it assumes that average has purchase on what actually happens at a
particular input. That is an empirical claim about a nonlinear network and it has not, to our
knowledge, been directly tested.

## Design

At layer `l`, for prompt `p`, perturb the residual stream in place at all positions:

    h_l  ->  h_l + eps * v,     ||v||_2 = 1,   eps frozen (see doses)

Three arms, identical except for `v`:

| arm | direction `v` | role |
|---|---|---|
| **ALIGNED** | top right-singular vector of `D_l = U_probe @ J_l` | the lens's own claim about which residual direction most moves the readout |
| **RANDOM** | uniform random unit vector in `R^d`, seed frozen per (model, layer, prompt, repeat) | floor |
| **LOCAL** | unit-normalised `grad_{h_l} log p(t*)`, where `t*` is the clean top-1 next token at this input | ceiling: the true local first-order direction |

`U_probe` is the shared 4,096-token probe used throughout the atlas. RANDOM is drawn 8 times
per cell and averaged, so the floor is estimated rather than sampled once.

## Outcome (one scalar, frozen)

    KL_arm = KL( p_perturbed || p_clean )   on the next-token distribution, in nats

Direction frozen as written. Primary dose is the **smallest** dose (see gates), where the
first-order regime is most defensible.

Primary quantity, per (model, layer):

    A_l = log( KL_aligned / mean KL_random )

A log ratio because KL scales multiplicatively with `eps` in the linear regime, and because
per-layer KL magnitudes vary by orders of magnitude across depth. `A_l > 0` means the lens
direction beats an equal-norm random direction.

Secondary, per (model, layer): `C_l = KL_aligned / KL_local`, the fraction of the achievable
first-order effect the corpus-averaged lens captures. `C_l` near 1 means the average lens is
locally as good as the true gradient; near 0 means it is not.

## Hypotheses

- **P1 (primary, confirmatory).** `A_l > 0` on average across layers, in every tested model.
  One-sided; direction is a priori because the lens claims these directions matter.
- **P2 (secondary).** Report `C_l`, with no threshold. This is a measurement, not a test.
- **P3 (exploratory, explicitly labelled).** Does `A_l` change at frozen CKA boundaries versus
  depth-matched control cuts? Reported as exploratory whatever it shows. No decision rides on
  it, and the previous campaigns are the reason for that caution.

## Inference

Layers within a model are **not** independent. The confirmatory unit is the **model**: compute
the median `A_l` over layers per model, and require P1 to hold in **every** tested model rather
than pooling. With 3 models this is a consistency requirement, not a population claim, and we
will not make a population claim.

Per-model uncertainty comes from bootstrapping over **prompts** (2,000 resamples), which is
available because the runner stores per-prompt KL.

## Gates (frozen; failure means VOID, not null)

1. **Execution.** `eps = 0` reproduces clean logits to `< 1e-6` max abs difference, and the
   injected vector's norm matches `eps` to `< 1e-6` relative.
2. **Equal norm.** `| ||v_aligned|| - ||v_random|| |` and the LOCAL equivalent `< 1e-6`. This
   is the gate the whole design rests on; if the arms are not norm-matched the contrast is
   meaningless.
3. **Linear regime.** Choose `eps` per model as the largest value for which
   `KL(2*eps) / KL(eps)` is within `[3.2, 4.8]` (i.e. approximately quadratic in `eps`, as a
   first-order perturbation should be). Doses scanned: `eps` in `{0.05, 0.1, 0.25, 0.5, 1.0}`
   times the median residual norm at that layer. If no dose satisfies the window, that model is
   VOID rather than reported.
4. **Sensitivity.** `KL_local > KL_random` must hold on average, i.e. the true local gradient
   beats a random direction. If the ceiling does not beat the floor, the instrument cannot
   resolve direction quality at all and the run is VOID. This is the positive control the
   previous designs lacked, and unlike an inserted bottleneck it tests exactly the quantity the
   experiment measures.

## Models

Small, lens-available, runnable free on CPU. Frozen before any outcome:

| model | layers | why |
|---|---|---|
| gpt2-small | 12 | different family, oldest architecture |
| gemma-3-270m | 18 | tied 262k vocab, very narrow (d=640) |
| qwen3.5-0.8b | 24 | third family |

Prompts: 200 WikiText-103 passages, truncated to 64 tokens, **disjoint from the passages used
to fit any lens** (the neuronpedia lenses use a fixed wikitext subset; we take a held-out slice
and record the indices). Perturbation is applied at all positions; KL is read at the final
position.

## Decision rules (frozen)

| outcome | verdict |
|---|---|
| `A > 0` per-model bootstrap CI excludes 0 in all 3 models | **CONFIRMED**: lens directions have local causal purchase beyond equal-norm random |
| CI includes 0 in any model | **NOT CONFIRMED**: report per-model, make no general claim |
| any gate fails | **VOID** for that model; report which gate and why |

A null here would be a substantive result: it would mean the corpus-averaged lens does not
predict local causal effect, which would materially constrain how every J-space result,
including our own atlas, may be interpreted.

## Cost

CPU only, this box, **$0**. Three small models, `3 arms x 5 doses x L layers x 200 prompts`
batched forwards, plus one backward per (prompt, layer) for the LOCAL arm.

## Receipt

Stores per-prompt KL for every arm/dose/layer, both direction vectors' norms, seeds, the
chosen `eps` per model and its dose-scan, gate outcomes, prompt indices and hashes, and package
versions. The test is that the bootstrap and every reported number can be recomputed from the
receipt without re-running the models.

## What this cannot establish

Local causal purchase of lens directions is **not** evidence that CKA depth bands are
computational stages. That question is parked, and the two failed campaigns stand as the
reason.

---

## Amendment 1 (2026-07-25): dose grid extended downward

**Trigger.** The frozen linear-regime gate (gate 3) **fired** on the CPU smoke run: dose
ratios `KL(2e)/KL(e)` came out 2.39, 6.40 and 1.11 across the grid, none inside the required
`[3.2, 4.8]` quadratic window. Under the frozen rule that is VOID, and the prescribed response
is to retune on the pilot rather than to report.

**Diagnosis.** The pattern (sub-quadratic at the smallest dose, super-quadratic in the middle,
saturating at the top) says the first-order regime lies **below** the grid floor of 0.05, so
the grid was searching the wrong range rather than the gate being wrong.

**Change.** Dose grid `{0.05, 0.1, 0.25, 0.5, 1.0}` becomes `{0.0125, 0.025, 0.05, 0.1, 0.2}`,
every adjacent pair an exact doubling, which also yields four ratio estimates instead of two.
The window, the gate, the outcome, the arms, and every decision rule are unchanged.

**Disclosure.** The smoke run was 8 prompts at 32 tokens, explicitly a mechanics test and not
the confirmatory sample, but it did display the primary contrast: aligned KL exceeded
equal-norm random KL by roughly 6 to 8 times in the first four layers, and local exceeded both.
That is the direction P1 predicts a priori and one-sided. We record here that the dose
amendment was made **after** seeing that smoke-level direction, because the gate that forced
the amendment could not have fired without running the model. The amendment changes a
calibration range, not the estimand or any threshold, and the confirmatory numbers will come
from the frozen full-scale run.

**Execution venue.** The frozen spec (200 prompts, 64 tokens, three models) is not tractable on
this box's CPU: the smoke took about 10 s per layer at 8 prompts and 32 tokens, so the full
grid extrapolates to hours per model. It will run on one cheap GPU instead, at an expected
cost under $1. The design is unchanged; only the venue is.

## Amendment 2 (2026-07-25): calibrate the dose on the LOCAL arm for the C measurement

**Trigger.** In the first full run all four gates passed and P1 was confirmed, but the
secondary `C` measurement was found uninterpretable: the LOCAL arm sat at 0.6-1.0 nats in every
layer of every model, and in gemma-3-270m the aligned KL (1.49) exceeded the local KL (0.94),
which is impossible in the linear regime because LOCAL is by construction the first-order
maximum.

**Diagnosis, a flaw in our own gate.** Gate 3 selects the dose by scanning the **aligned** arm
and then applies that dose to all arms. LOCAL is the strongest arm by construction, so a dose
that leaves ALIGNED roughly quadratic leaves LOCAL saturated.

**Change.** A `--calib-arm` flag; for the `C` measurement the gate calibrates on **local**. The
dose grid is extended downward as an exact-doubling ladder
`{0.0015625 ... 0.2}` because calibrating on the strongest arm selects a much smaller dose.

**Scope.** This re-measures **P2 only**, which the pre-registration defines as a measurement
with no threshold and no decision attached. **P1 stands on the already-completed run** and is
not re-tested here; its confirmed result is not revisited, and the saturation present there
biases P1 toward the null rather than away from it. The withdrawn `C = 0.9996` reading is
recorded in `results.md` as an error rather than deleted.

**New risk to watch.** A smaller dose shrinks all arms quadratically, so RANDOM may approach
the numerical noise floor. The re-run records random-arm KL magnitudes so this can be checked,
and any model whose random arm falls below `1e-5` nats will have its `C` reported as
unmeasurable rather than computed.
