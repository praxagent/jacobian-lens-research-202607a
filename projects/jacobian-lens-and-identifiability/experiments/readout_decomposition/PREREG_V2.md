# Pre-registration v2: closing the unexplained variance in the top readout direction

**FROZEN before any v2 run.** Written 2026-07-26. No v2 data existed at commit.

## Where v1 left it

The frozen seven-feature set explains **adjusted `R^2` = 0.572** of direction 0's per-token
effect, with the model's own output prior the single best predictor at 0.455. That landed at
**PARTLY SETTLED** against thresholds fixed in advance, leaving **43% unaccounted for**. This asks
whether that residual is a missing *feature* or a missing *functional form*.

## Disclosure: what we had seen before writing this

Not blind. We had seen all of v1's results, and separately decoded all four directions into
tokens. That decode is what motivates the script hypothesis below: direction 0's positive pole is
dominated by rare CJK and Cyrillic fragments alongside `<unused>` slots, and direction 3 turned
out to be an orthographic axis our ASCII-only `is_punct` could not represent. Both new feature
families below are therefore **post-hoc motivated** and this run is confirmatory of a hypothesis
we already have reason to believe. We say so rather than presenting it as a clean prediction.

## Two candidate explanations for the residual, tested separately

**H1, missing features.** The v1 set has no representation of writing system or byte-level
structure, while the direction's own top tokens are largely non-Latin. Add:

- `is_non_latin`: the token contains any character outside Basic Latin.
- `is_cjk`: contains a CJK ideograph, kana, or hangul.
- `is_cyrillic_or_greek`: contains a Cyrillic or Greek character.
- `bytes_per_char`: UTF-8 byte length divided by character length, a continuous proxy for how
  far outside ASCII the token sits.
- `is_continuation`: the token lacks the word-initial marker, i.e. it continues a word.

**H2, missing functional form.** v1 fitted `model_prior` linearly in log space. The relationship
may not be linear over six orders of magnitude of probability. Refit with `model_prior` entered
as **20 equal-count bins** (a step function), which can absorb any monotone or non-monotone shape
without assuming one.

## Analysis

Four nested models, all on the same vocabulary items that occur in the frequency corpus, all
reporting adjusted `R^2` so added parameters are penalised:

| model | contents |
|---|---|
| M0 | the frozen v1 seven features (reproduces 0.572 or the run is VOID) |
| M1 | M0 + the five H1 features |
| M2 | M0 + binned `model_prior` (H2) |
| M3 | M0 + H1 + H2 |

## Decision rules (frozen)

Verdict thresholds are **unchanged from v1**, applied to M3:

| adjusted `R^2` of M3 | verdict |
|---|---|
| >= 0.80 | **SETTLED** |
| 0.50 to 0.80 | **PARTLY SETTLED** |
| < 0.50 | **UNEXPLAINED** |

Attribution rules, so we cannot claim both explanations after the fact:

- H1 is **supported** if `M1 - M0 >= 0.05` in adjusted `R^2`.
- H2 is **supported** if `M2 - M0 >= 0.05`.
- If both are supported, we report their individual contributions and the overlap
  (`M1 + M2 - M0 - M3`) rather than assigning the residual to whichever we prefer.

## Gates

- M0 must reproduce v1's adjusted `R^2` of 0.572 to within 0.01, or the run is VOID.
- The direction-0 sign convention and the `-0.591` unigram correlation gate from v1 must both
  reproduce, or the run is VOID.

## Predictions

We predict **H1 supported, H2 not**, and a final verdict still at **PARTLY SETTLED** rather than
SETTLED. Stated so that a miss is recorded. If M3 clears 0.80 we will have been wrong about how
much was missing, which is the pleasant direction to be wrong in.

## What this cannot establish

Higher `R^2` means the direction is better *described* by these features, not that the model
computes it that way. This is one direction of one model's readout, on one corpus.

## Cost

CPU only, all inputs already local (the embedding matrix, the `A` matrix, the fitted lens, and
the model prior). **$0**, RAM-frugal single pass over the vocabulary.
