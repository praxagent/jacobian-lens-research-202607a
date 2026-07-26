# Pre-registration: what does the invariant readout subspace encode?

**Status: FROZEN before any run.** Written 2026-07-26. No decomposition data existed at commit.

## The gap this closes

gemma-2-9b's near-flat lens is produced by a low-dimensional invariant readout subspace
(participation ratio 3.8; top direction 48% of the readout energy; cross-layer cosine 0.9997).
We have decoded only its **top** direction, and explained only part of it: it correlates
`-0.591` with log unigram frequency, better than the `+0.483` embedding-norm proxy, leaving
roughly two thirds of the variance unaccounted for.

"What the direction is" is therefore **partly** settled. This closes it, or fails to and says so.

## What "settled" means, fixed in advance

We cannot prove we have found *the* explanation. The achievable claim is that a **small,
interpretable, pre-named feature set accounts for most of the per-token effect**.

| adjusted `R^2` from the frozen feature set | verdict |
|---|---|
| >= 0.80 | **SETTLED**: the direction is accounted for by named features |
| 0.50 to 0.80 | **PARTLY SETTLED**: report the share explained and what is missing |
| < 0.50 | **UNEXPLAINED**: the direction is not what these features describe, and we say so |

Thresholds are frozen now precisely because an awkward middle result is likely and we do not
want to choose the adjective after seeing the number.

## Target quantity

For eigendirection `k` of `A = mean_l J_l^T M J_l`, the per-token effect is
`e_k[t] = mean_l (U (J_l w_k))[t]`, the average across layers of what that direction does to
each token's logit. We analyse the **top four** directions, which together carry ~77% of the
readout energy, not only the top one.

## Frozen feature set

1. **`model_prior`**: the model's own marginal output distribution, `log mean_context softmax`,
   averaged over the fitting corpus. This is the primary hypothesis: a readout direction should
   encode *the model's* default output tendency, not an external corpus count.
2. **`log_unigram`**: log unigram frequency in the model's tokenizer over WikiText (already
   measured; retained so the new features must beat it).
3. **`is_unused`**: indicator for `<unused>`/special/never-trained vocabulary slots. Gemma ships
   a large block of these with untrained embeddings, and lumping them in conflates "rare" with
   "never trained".
4. **`embed_norm`**: the proxy we previously reported, retained as a control.
5. **`word_initial`**: indicator for the space-prefixed (word-start) form.
6. **`tok_len`**: token string length in characters.
7. **`is_punct`**: indicator for punctuation-only tokens.

No feature may be added after seeing results. If we later want another, it is reported as
exploratory and excluded from the verdict.

## Analysis

Ordinary least squares of `e_k` on the standardised feature set, per direction `k`, reporting
adjusted `R^2` and standardised coefficients. Fitted on the vocabulary items that **occur** in
the frequency corpus, with `is_unused` carrying the never-trained population, and repeated on
the full vocabulary as a robustness check.

**Pre-declared predictions.** `model_prior` beats `log_unigram` (a model's own prior should
explain its own readout better than an external corpus count). `is_unused` carries substantial
weight. The top direction is more frequency-like than directions 2 to 4.

## Gates

- The recovered top direction must reproduce the previously reported `-0.591` correlation with
  `log_unigram` to within `0.02`, or the pipeline is VOID rather than informative.
- `model_prior` must be a proper distribution (sums to ~1 before logging) or that feature is
  dropped and the run reported without it.

## Cost

Features 2 to 7 and the regression are free and local. `model_prior` needs forward passes over
gemma-2-9b, roughly 20 minutes on one RTX 3090, about **$0.10**. No backward passes.

## What this cannot establish

A high `R^2` shows the direction is *describable* by these features, not that the model
*computes* it that way. And this is one model: gemma-2-9b is where the flat readout was found,
so any account here is about that checkpoint unless separately replicated.
