# Pre-registration: Test B re-run with a repaired usability gate

**FROZEN before the re-analysis and before any new model is run.** Written 2026-07-26.

## Full disclosure of what we had already seen

This is **not blind**, and pretending otherwise would be worse than the problem it fixes. At the
time of writing we had seen:

- every per-model agreement number and null median from the first Test B run;
- the pooled verdict (`p = 0.608`, BANDS ARE A READOUT PROPERTY);
- the full distributional statistics of all eight activation CKA maps, including the fact that
  the median-based gate and a range-based gate disagree on three models.

We can therefore anticipate roughly how a gate change moves the result. Two safeguards:
**(1)** the repaired gate is justified by a principle stated below that never refers to
agreement, and **(2)** we commit in advance to reporting the verdict under **both** gates side by
side, so a reader sees the sensitivity rather than our preferred slice. If the two gates
disagree on the verdict, that disagreement **is** the finding and we report it as such rather
than picking one.

TJ asked for this re-run over our own recommendation not to do it. That recommendation was
wrong: the instrument check below found a real defect that we would have shipped.

## The defect

The frozen gate excluded a model when its **median** off-diagonal activation CKA was `>= 0.999`.
The gate's purpose was to exclude maps with **no structure to compare against**. The median does
not measure that:

| model | median | range | old gate | has structure? |
|---|---|---|---|---|
| `qwen3-4b` | 0.9990 | **0.9570** | EXCLUDED | yes, enormously (min 0.043) |
| `gpt2-small` | 0.9993 | 0.1361 | EXCLUDED | yes, some |
| `llama3.1-8b` | 0.9998 | 0.0022 | EXCLUDED | no, correctly excluded |
| `gemma-2-2b` | 0.9794 | **0.0655** | **KEPT** | barely, less than either excluded model |

The rank correlation between median and range across the eight models is only `-0.595`. A model
can have a high median because most layer pairs are similar while still separating sharply
somewhere, which is exactly the block structure we are looking for. The old gate threw away the
model with the **most** dynamic range in the whole set and kept one with almost none, and the one
it kept (`gemma-2-2b`) contributed the single worst agreement score, which pushed the pooled
result toward the null.

The result may well survive. The gate was still wrong.

## Repaired gate (frozen)

A model is **usable** if its off-diagonal activation CKA spans a range of at least **0.10**.

The principle, stated without reference to any agreement number: the test asks whether two
segmentations of the same stack coincide. That question is only meaningful if the activation map
contains a boundary to find, which requires the map to vary. Range is the direct measure of
"does this map vary"; the median is a measure of "where does it sit", which is a different
question. The floor of 0.10 is set at roughly the point where a fitted segmentation could move a
boundary at all given the CKA noise we observe within a single map.

Under this gate, `llama3.1-8b` and `gemma-2-2b` are excluded, and `gpt2-small` and `qwen3-4b`
are restored: **6 usable models** from the original eight.

## Repaired decision rule (frozen)

The original rule required "pooled `p < 0.05` **and** at least 6 of 8 models beating their own
null median", which became unreachable once any model was excluded. Replaced with a rule that
scales with the number of usable models `k`:

| outcome | verdict |
|---|---|
| pooled `p < 0.05` **and** a strict majority (`> k/2`) of usable models beat their own null median | **BANDS TRACK REPRESENTATIONS** |
| pooled `p >= 0.05` | **BANDS ARE A READOUT PROPERTY** |
| otherwise | **MIXED** |

`k` must be at least 4 for any verdict to be issued; below that we report per-model only.

## Added models, to restore power

The first run lost a quarter of its set. We add four models that already have cached
shared-vocabulary lens maps and fit on one mid-size GPU, chosen for family and size spread
before any of them is measured:

| slug | HF id |
|---|---|
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B` |
| `gemma-3-1b` | `google/gemma-3-1b-pt` |
| `olmo-3-1025-7b` | `allenai/Olmo-3-1025-7B` |
| `qwen2.5-7b-it` | `Qwen/Qwen2.5-7B-Instruct` |

Added models pass through the same repaired gate. If a model fails to run it is reported as
failed, never silently replaced.

## Second question, run at the same time: is the saturation an artifact of scale?

Linear CKA is invariant to isotropic scaling but **not** to per-dimension scaling. Residual
stream norms grow with depth and a few dimensions can dominate the Gram matrix, which would
inflate CKA toward 1 without the representations being alike in any interesting sense. That is a
live explanation for `llama3.1-8b`'s range of 0.0022.

So every model is measured **twice**: raw, and with each layer's features standardised to zero
mean and unit variance per dimension before the CKA. This is reported as a **robustness check on
the instrument**, not as a second verdict.

**The claim this could overturn is our own.** The note's headline Gemma result, that activations
are structured while the lens is flat, rests on this instrument. If standardisation changes
gemma-2-9b's activation band separation materially, that claim needs re-scoping and we will say
so in the note.

## Prediction

We predict the verdict is unchanged at **BANDS ARE A READOUT PROPERTY** under both gates, and
that standardisation raises the dynamic range of the saturated models without rescuing boundary
agreement. We also predict the Gemma dissociation survives standardisation. All three are stated
so a miss is on the record.

## Gates

- `gemma-2-9b` raw activation band separation must reproduce the first run's `+0.0812` to within
  `0.02`, or the pipeline is VOID.
- Both gates' verdicts are reported regardless of agreement between them.

## Cost

Twelve models, forward passes only, two CKA computations each. One mid-size GPU, estimated
**under $3**.
