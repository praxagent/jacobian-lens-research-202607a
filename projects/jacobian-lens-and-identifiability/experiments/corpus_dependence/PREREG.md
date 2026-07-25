# Pre-registration: is a J-lens depth map a property of the model, or of the fitting corpus?

**Status: FROZEN before any fit.** Written 2026-07-25. No corpus-comparison data existed at
commit time.

## Why this is the question that scopes everything else

Every lens in our 36-model atlas, and every public lens we used, was fitted on **WikiText**. A
Jacobian lens is `J_l = E[d h_final / d h_l]`, an expectation **over a corpus**. If the fitted
depth boundaries move when the expectation is taken over different text, then a large part of
what the atlas reports is a statement about WikiText rather than about the networks: the band
statistic, the cross-family comparisons, and the Gemma flatness result all inherit the caveat.

This is a **geometry** question, not a causal one, so none of the identifiability problems that
killed our block-causality designs apply here. We are comparing two fitted lenses, which is the
operation this project has done most often and tested best.

## Design

For each model, fit **three** lenses with an identical recipe, differing only as stated:

| fit | corpus | seed | role |
|---|---|---|---|
| `wiki_a` | WikiText-103 | 0 | reference |
| `wiki_b` | WikiText-103 | 1 | **seed null**: same corpus, different sample |
| `code` | codeparrot-clean | 0 | corpus arm |

**The seed null is the whole design.** Any two fits differ, because each is an expectation over
a finite sample. Without `wiki_b` we could not tell a corpus effect from ordinary fitting
variation, and "the boundaries moved" would be uninterpretable. Every comparison below is
reported against it.

Prompts are length-matched (`--match-length`) so both corpora contribute windows that fill the
128-token cap, which is the confound a previous arm of this campaign left open.

## Measures

For each pair of lenses (`wiki_a`↔`wiki_b`, `wiki_a`↔`code`), on the **shared-vocabulary probe**:

1. **Boundary shift.** `|b1_x - b1_y| + |b2_x - b2_y|` in layers, from each lens's own fitted
   3-segmentation.
2. **Map distance.** `1 - CKA` between the two lenses' layer-by-layer readout-geometry maps,
   i.e. how differently the whole depth map reads.
3. **Band statistic shift.** `|mid_sep_x - mid_sep_y|`.

## Hypotheses

- **P1 (primary).** Corpus changes the map more than sampling does: for each model,
  `map_distance(wiki_a, code) > map_distance(wiki_a, wiki_b)`.
- **P2.** Boundary shift is larger across corpora than across seeds.
- **P3 (descriptive, no threshold).** Absolute size of the corpus effect: if boundaries move by
  0 to 1 layers on a 24-layer model, the atlas is robust; if they move by a third of the depth,
  much of the atlas is corpus-specific.

Direction for P1 and P2 is a priori (a corpus should matter at least as much as a reseed), so
both are one-sided.

## Decision rules (frozen)

| outcome | verdict, and what the note must then say |
|---|---|
| corpus effect exceeds the seed null in **all** models | **CORPUS MATTERS.** The atlas is scoped to WikiText-fitted lenses and we say so in the headline, not the caveats |
| corpus effect within the seed null in **all** models | **ROBUST.** Depth structure is a model property at the resolution we can measure; the scope caveat is discharged |
| mixed across models | **MIXED.** Report per model; make no general claim |

A "robust" result is as publishable as a "corpus matters" result, and we commit to reporting
either with equal prominence. **This is a scope test we could fail**: if corpus dominates, a
substantial part of our own published note is weakened, and we will say so in the note itself
rather than in a footnote.

## Models and cost

Three models spanning three families, small enough to afford three fits each, and the same
three used in the causal experiment so the note stays self-contained:

| model | layers | d | why |
|---|---|---|---|
| gpt2-small | 12 | 768 | oldest architecture, different tokenizer family |
| gemma-3-270m | 18 | 640 | 262k tied vocab, narrow |
| qwen3.5-0.8b | 24 | 1024 | third family |

Nine fits total. Fitting is backward-pass bound at roughly `d_model / dim_batch` backward
passes per prompt over 100 prompts. On one mid-range GPU we estimate **well under $10**, against
an authorized ceiling of **$50**. If a fit's measured throughput implies breaching that ceiling,
we stop and report the partial result rather than continuing.

## Gates

- Every fit must converge to a finite lens with the expected layer count, or that model is VOID.
- The seed null must be **non-zero**: if `wiki_a` and `wiki_b` produce identical maps, the
  comparison has no resolution and the result is uninterpretable rather than "robust".

## Receipt

Per fit: corpus, seed, prompt count and token-length distribution, dim_batch, model revision,
the lens file hash, and the resulting CKA map. Per comparison: all three measures. The test is
that every number in the note can be recomputed from these without refitting.
