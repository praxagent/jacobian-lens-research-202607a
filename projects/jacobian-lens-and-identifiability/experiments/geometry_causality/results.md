# Results: J-space lens directions have local causal purchase

Design frozen in [`PREREG.md`](PREREG.md) before any run (commit `09e5bf7`), with Amendment 1
(dose grid extended downward, after the linear-regime gate fired on the CPU smoke) at `291d8fc`.

## Runs

One RTX 3090, ~$0.15 total, all three models on one consistent stack (torch 2.6.0+cu124,
transformers 5.14.1). 200 held-out WikiText passages at 64 tokens, prompt indices recorded.
Pod terminated and verified gone.

## P1 (primary, confirmatory): CONFIRMED in all three models

`A = median over layers of log(KL_aligned / mean KL_random)`, bootstrapped over prompts
(2,000 resamples). All four gates passed in every model (execution, equal-norm, linear-regime,
and the `local > random` sensitivity control).

| model | median A | 95% CI | as a ratio | P1 |
|---|---|---|---|---|
| gpt2-small | +1.308 | [+1.221, +1.383] | **3.70x** | YES |
| gemma-3-270m | +2.093 | [+1.996, +2.178] | **8.11x** | YES |
| qwen3.5-0.8b | +2.005 | [+1.760, +2.221] | **7.43x** | YES |

**A corpus-averaged Jacobian lens direction moves the output 3.7 to 8.1 times more than an
equal-norm random direction at the same layer and the same input.** Because the arms share
layer, input, and norm, this cannot be explained by update magnitude, depth position,
architecture, or generic sensitivity: those are identical across arms by construction. This is
the confound that killed both `block_patching` designs, and it is what the internal contrast
removes.

The result is also **conservative**: saturation (below) compresses ratios toward 1, so the true
effect is at least this large.

## P2 (secondary measurement): NOT INTERPRETABLE, and we say so

`C = KL_aligned / KL_local` was meant to measure what fraction of the achievable first-order
effect a corpus-averaged lens captures. **That measurement failed its precondition.**

| model | median KL random | aligned | local |
|---|---|---|---|
| gpt2-small | 0.039 | 0.125 | **0.953** |
| gemma-3-270m | 0.134 | **1.487** | 0.938 |
| qwen3.5-0.8b | 0.010 | 0.075 | **0.611** |

The LOCAL arm sits at 0.6 to 1.0 nats in every layer of every model, which is nowhere near a
first-order perturbation. The diagnostic that settles it: in gemma-3-270m the **aligned** KL
(1.49) *exceeds* the **local** KL (0.94), and local is by construction the maximal first-order
direction. A first-order optimum cannot be beaten in the linear regime, so both arms are out of
it there.

**Root cause, and it is a flaw in our gate.** The linear-regime gate selects the dose by
scanning the **aligned** arm, then applies that dose to all arms. LOCAL is the strongest arm by
construction, so at a dose where aligned is roughly quadratic, local has already saturated. The
gate should calibrate on the **strongest** arm, not the one we happen to care about.

So the earlier headline reading of gemma's `C = 0.9996` ("the averaged lens is locally as good
as the true gradient") is **wrong and withdrawn**. It is a saturation artifact. Measuring `C`
honestly requires re-running with the dose calibrated on LOCAL, which is cheap and which we
have not done.

P1 is unaffected: aligned and random are both far below local (0.01 to 0.13 nats for random,
0.07 to 0.13 for aligned in gpt2 and qwen), and where aligned does saturate, the bias is toward
the null.

## What this establishes, and what it does not

**Establishes:** the directions a Jacobian lens identifies are not generic. They carry real,
input-local causal weight in three model families, over an equal-norm random baseline, under a
contrast designed so the obvious confounds cancel. That is the foundational assumption of the
whole J-lens enterprise, our atlas included, and as far as we can tell it had not been directly
tested.

**Does not establish:** anything about CKA depth bands being computational stages. That
question stays parked, with two failed campaigns as the reason. It also does not yet quantify
*how* good the averaged lens is relative to the achievable optimum, because that measurement
saturated.

## Next, if continued

Re-run with the dose calibrated on the LOCAL arm to recover an interpretable `C`. Same models,
same code, one flag, well under a dollar. P3 (does `A` change at CKA boundaries) remains
exploratory and unreported here.

---

## P2 re-measurement (Amendment 2): one model of three yields an interpretable C

Re-ran all three with the dose calibrated on the **LOCAL** arm, the fix for the gate flaw
above. Prompts were loaded from a frozen artifact carrying the **exact dataset indices the P1
receipt recorded** (200000..200513), verified identical across all three P1 receipts, so C and
A are measured on the same inputs. That also removed `datasets` from the pod, whose
pyarrow/numpy ABI mismatch was segfaulting the interpreter on import.

| model | chosen dose | linear gate | median KL random / aligned / local | median C | disposition |
|---|---|---|---|---|---|
| gpt2-small | 0.0015625 | **FAIL** | 1.74e-2 / 1.88e-2 / 1.81e-2 | 1.021 | **VOID** |
| gemma-3-270m | 0.0015625 | **FAIL** | 2.95e-3 / 2.53e-2 / 1.73e-1 | 0.098 | **VOID** |
| qwen3.5-0.8b | 0.0125 | PASS | 7.91e-4 / 1.60e-3 / 3.14e-2 | **0.045** | **REPORTABLE** |

The random-arm floor rule (pre-set in Amendment 2 before these numbers existed) was satisfied
in all three, so nothing was lost to numerical noise; the failures are the linear-regime gate.

**gpt2-small and gemma-3-270m are VOID for C.** No dose in the ladder put the local arm in the
approximately-quadratic window: gpt2's best ratio was 2.83 and gemma's 2.74, both short of the
required 3.2. gpt2's `C = 1.021` is itself the tell, since a value above 1 says the lens arm
beat the first-order optimum, which cannot happen in the linear regime. We report these as
gate failures rather than as measurements.

**qwen3.5-0.8b gives the one clean number: `C = 0.045`.** With every gate passing, a
corpus-averaged lens direction achieves about **4.5% of the output change** that the true
input-specific gradient achieves at the same layer and norm. Directionally far better than
chance, and a long way from optimal.

### A finding hiding in the dose dependence

`A` is **not** dose-invariant, which it should be in a pure first-order regime, where KL scales
as `eps^2` for every direction and the ratio cancels. For qwen3.5-0.8b, `A` falls from +2.005
at the aligned-calibrated dose to +0.701 at the smaller local-calibrated dose (7.4x down to
2.0x). So a meaningful part of the lens direction's advantage appears at **larger** perturbations,
i.e. it is not purely first-order.

That is worth stating because it cuts against the lens's own framing: a Jacobian lens is a
first-order object, but its directions look *more* advantaged where first-order approximation
is *worse*. We flag it as an observation, not a claim: it was not pre-registered, it rests on
two doses in one model, and dose-dependence could also arise from the random arm approaching
its own floor. It is a clean, cheap follow-up: sweep `A` across the full dose ladder in all
three models and see whether the pattern holds.

## Standing summary

- **P1 (confirmed, 3/3 models):** lens directions beat equal-norm random by 3.70x, 8.11x, 7.43x.
- **P2 (1/3 models):** the averaged lens captures ~4.5% of achievable first-order effect in
  qwen3.5-0.8b; VOID in the other two on the linear-regime gate.
- **Unregistered observation:** the lens advantage grows with perturbation size, suggesting a
  non-first-order component.

Total cost of the geometry-causality experiment: about **$0.30**.

---

## P4 (dose-dependence): our own "non-first-order" observation is WITHDRAWN

The addendum froze a decision table before this sweep, precisely so the two explanations could
not be chosen between after the fact. The table decided against us.

| model | `k_random` (small doses) | `k_aligned` (small doses) | verdict on that model |
|---|---|---|---|
| gemma-3-270m | **1.78** | **1.92** | **CLEAN** (`k ~ 2`, first-order regime reached) |
| gpt2-small | 0.05 | 0.07 | **FLOORED** |
| qwen3.5-0.8b | 0.13 | 0.46 | **FLOORED** |

`A` across the full ladder:

| model | 0.0016 | 0.0031 | 0.0063 | 0.0125 | 0.025 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|---|---|---|
| gemma-3-270m | +1.85 | +2.21 | +2.29 | +2.18 | +2.05 | +1.61 | +0.89 | +0.24 |
| gpt2-small | +0.06 | +0.09 | +0.13 | +0.12 | +0.36 | +0.79 | +1.29 | +1.77 |
| qwen3.5-0.8b | +0.08 | +0.17 | +0.34 | +0.72 | +1.07 | +1.57 | +2.03 | +2.63 |

**In the one model where the measurement is valid, `A` is flat.** gemma-3-270m is the only
model whose arms reach the first-order regime (`k ~ 2` at small doses), and there `A` sits
between +1.85 and +2.29 across a full order of magnitude of dose, i.e. **dose-invariant within
noise**, exactly as a first-order quantity should be. Its decline at the top of the ladder
(+0.89, +0.24) is the aligned arm saturating first, which is expected and is outside the
regime the question is about.

**The monotone rise we saw in the other two is a floor artifact.** Their small-dose exponents
are 0.05 to 0.46, nowhere near 2: both arms are pinned on a floor that does not shrink with
dose, so the ratio starts near zero and only climbs as the real signal emerges above it. That
manufactures precisely the "advantage grows with dose" pattern we mistook for physics.

**So the observation reported in the previous section is withdrawn.** `A` is not
dose-dependent; it looked dose-dependent because two of three models were measured below their
precision floor. **P4 is supported**: the lens direction's advantage is a first-order effect,
consistent with the lens's own framing.

**Mechanism of the floor, and a caveat we cannot fully close here.** These runs use bfloat16 on
GPU. With roughly 8 mantissa bits, a perturbation at 0.3% of the residual norm is comparable to
the rounding error of storing the residual itself, so both arms measure rounding rather than
the intended direction. Note this is *not* caught by the execution gate: at `eps = 0` there is
no perturbation and no rounding difference, so the gate passes while small non-zero doses are
floor-dominated. A gate can be correct and still sit beside the failure. The clean follow-up is
float32, which these model sizes easily allow, and which should drop the floor by orders of
magnitude.

**P1 is unaffected, and we checked rather than assumed.** The doses used for P1 sit above each
model's floor: gpt2 at 0.1 gives `A = +1.29` in the sweep against +1.31 reported, qwen at 0.1
gives +2.03 against +2.01, gemma at 0.0125 gives +2.18 against +2.09. All three are far from
the floor-dominated region where `A` collapses toward zero. Where a floor does intrude it
inflates the random arm and therefore **understates** `A`, so the headline 3.7x to 8.1x remains
a lower bound.

## Revised standing summary

- **P1 (confirmed, 3/3):** lens directions beat equal-norm random by 3.70x, 8.11x, 7.43x. Doses
  verified above the precision floor; effect is a lower bound.
- **P2 (1/3):** averaged lens captures ~4.5% of achievable first-order effect (qwen3.5-0.8b);
  VOID elsewhere on the linear-regime gate.
- **P4 (supported):** the advantage is dose-invariant, i.e. first-order, in the one model that
  reaches the regime. Our earlier contrary observation is withdrawn as a floor artifact.
- **Open:** repeat in float32 to lower the floor and bring gpt2 and qwen into the regime, which
  would let P2 and P4 be measured in all three models.

Total cost across the geometry-causality experiment: about **$0.55**.
