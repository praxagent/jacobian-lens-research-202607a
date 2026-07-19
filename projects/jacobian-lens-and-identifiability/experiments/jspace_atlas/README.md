# J-space atlas — Tier 0 (CPU, $0): the zoo-wide geometry study

**Pre-registration (frozen before any Tier-0 computation; TJ approved Tiers 0+1
2026-07-19).** Motivated by eliebak's Laguna/optimizer J-space threads and the
PrimeIntellect lens releases: apply their analysis toolkit (per-layer participation
ratio, cross-model CKA distance structure, fitted segmentation) across OUR axes —
36 models, 5 families, 500× scale span — with the controls that ecosystem work lacks
(random-transport nulls everywhere, shared-vocab probe for cross-tokenizer rigor).

## Inputs (all existing, no GPU)
- 35 Neuronpedia pre-fitted lenses (`emergence.csv` slugs) + our own Qwen3.5-397B lens
  (`artifacts/lenses-397b`, slug `qwen35-397b-own`).
- Shared-vocab probe: `../jacobian_lens/shared_tokens.json` (4,096 strings, 37
  tokenizers) for every cross-model comparison; own-vocab seed-0 n=4096 probe for
  within-model quantities (the release convention).

## Frozen metrics
- Token geometry `D_l = U_probe @ J_l`, fp32 compute / fp16 store (release convention).
- Linear CKA (`common/cka.py`) on column-centered features.
- **PR per layer**: participation ratio `(Σσ²)² / Σσ⁴` of the singular values of
  `D_l`, reported raw and /d_model. Null: scale-matched random J, same seed.
- **Fitted 3-segmentation**: boundaries (b1,b2) maximizing mean within-segment CKA
  (exhaustive over pairs); `fitted_sep` = mean within-segment − mean between-segment
  CKA. (Our fixed-thirds `mid_sep` remains the pre-registered release statistic; the
  fitted version quantifies how conservative it is.)
- **Cross-model distance**: for models A,B — median over A's layers of
  CKA(D_a, D_b(nearest relative depth)) on the pair's common resolved shared strings;
  distance = 1 − that. 2-D MDS embedding for display only.
- **Family clustering test**: observed statistic = mean between-family distance −
  mean within-family distance; null = 10,000 label permutations; one-sided p.
  Families: {gemma, qwen, llama, olmo, gpt2/pythia/gpt-oss as "other"}.

## Pre-stated predictions
- **P1 (confirmatory):** family clustering p < .05 on the real distance matrix, and
  the SAME test on the random-J null matrix shows no clustering (p > .1). Motivated by
  the mirror-test finding (family—not band—drives geometry) — that finding used
  different statistics, so this is a genuine out-of-sample test of the same idea.
- **P2 (confirmatory):** `qwen35-397b-own`'s nearest neighbor among the 35 zoo models
  is a Qwen-family model. (Chance ≈ 11/35 ≈ .31 under family-blind matching.)
- **P3 (descriptive, no gate):** PR/d rises toward late layers across families (the
  ecosystem-reported "capacity dial" shape); we report the curves and any exceptions.
- **NOT pre-registered (post-hoc, disclosed):** the 397B fitted band being wider than
  the fixed mid-third — we have already SEEN this in the released heatmap; the fitted
  segmentation quantifies an existing observation and is labeled exploratory.

## Stages
- **A (per model, resumable):** within-CKA matrix, PR real+null, fitted segmentation,
  shared-probe geometry basis (`U_shared` rows + resolved strings) → one npz + receipt
  per model in `atlas_out/`.
- **B (cross-model):** 36×36 distance matrix (real + null), MDS, permutation test,
  nearest-neighbor table, full cross-CKA maps for exemplar pairs (nearest, farthest,
  one within-family cross-scale, one cross-family same-scale).
- **Ledger:** `results.md`. Blog note only after results + TJ.

Cost: $0 (CPU, disk ≈ 40–60GB transient for lens downloads; big lenses deleted after
their Stage-A pass if space requires).
