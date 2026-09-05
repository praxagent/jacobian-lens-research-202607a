# Results ledger: J-space atlas, Tier 1 arms (2026-07-23) and later CPU analyses

The atlas itself (Stage A per-model maps, Stage B cross-model matrix) is ledgered in the
companion blog notes and in `atlas_out/summary.csv`, `atlas_out/shared_summary.csv`,
`atlas_out/stageB/stageB_results.json`. This file is the ledger for the Tier-1 GPU arms, which
had none until 2026-09-05 (their numbers lived only in `tier1_out/*.json` and the draft campaign
note), and for the CPU-only analyses added after the 2026-09-05 review.

## Tier 1 (design frozen in `TIER1_PREREG.md`; results committed `bd2af3c`, `dd97697`, `f4631a0`)

| arm | question | prediction (frozen) | result | verdict |
|---|---|---|---|---|
| A | is gemma-2-9b's near-layer-invariance the model or the public fit? | our independent fit: shared-probe off-diagonal CKA median > 0.95, band sep < 0.02 | median **0.9934**, min 0.9749, mid_sep **0.0059** (41 layers) | **reproduces: model property** |
| B | does fitting Qwen3-4B on code vs WikiText (length-matched) reorganise the mid band? | mid-band cross-lens CKA drops below 0.7, late stays > 0.9 | same-layer cross-lens CKA median **0.945** (early 0.887 / mid 0.938 / late 0.959) | **prediction failed**: modest, concentrated early |
| C | does FP8 quantisation change the lens? | cross-lens CKA > 0.9 | genuine dequantised FP8 vs bf16, same GPU/seed/corpus: median **0.99989** (early 0.9991 / mid 0.99991 / late 0.99990) | **confirmed**; first attempt VOID (A6000 path produced identical weights; `fp8_validity.identical = true`) |
| D | do outlier-layer counts differ by optimizer? | availability-gated | our census on eliebak's published per-optimizer CKA matrices: 520M counts 0 to 6 (scion 6, adamw 2, muon / nadamw / mini / soape 0); muon and nadamw clean at both sizes | **partial**: their fits, no random-J null, small counts |

Receipts: `tier1_out/tier1_results.json`, `tier1_out/gen_fp8_result.json`,
`tier1_out/armD_optimizer_outliers.json`, `tier1_out/gemma2_9b_ours.npz`, `tier1_out/qwen_curves.npz`.
The released arm-A lens: `praxagent-org/jacobian-lens-gemma-2-9b`. Cost was not recorded in this
ledger; the pod scripts are `pod_tier1.sh` and `pod_tier1_fp8.sh`.

**Reconciliation note (2026-09-05).** Arm B's statistic is a same-layer cross-lens CKA. The
corpus-dependence experiment (`../corpus_dependence`) measures the distance between whole
layer-by-layer maps and finds a large corpus effect (84x to 888x the seed null) with almost no
boundary movement. On the corpus experiment's three models the arm-B statistic gives medians
0.977 (gpt2-small), 0.919 (gemma-3-270m), 0.766 (qwen3.5-0.8b) for WikiText vs code, so 0.945
for Qwen3-4B is in range; the two results are consistent and describe different aspects of the
same change.

## Boundary identifiability (2026-09-05, CPU, `boundary_identifiability.py`)

Rule frozen in `../bands_vs_computation/PREREG_B3_IDENT.md` (after the spreads were seen, before
the restricted Test B was run): near-optimal set = segmentations within 5% of the objective's
range of the optimum; identified iff both boundary spreads <= 0.25 of depth.

**Shared probe: 20 of 35 lenses identified.** Not identified (worst spread, fraction of depth):
gemma-4-e4b 0.85, gemma-4-e2b 0.82, gemma-3-4b-it 0.70, gemma-3-270m 0.59, gemma-3-270m-it 0.59,
qwen3-14b 0.56, qwen3-1.7b 0.52, qwen3-8b 0.46, gemma-3-27b-it 0.44, llama3.3-70b-it 0.43,
gemma-2-9b 0.41, gemma-2-9b-it 0.41, gemma-2-2b-it 0.40, gemma-3-12b 0.32, gpt-oss-20b 0.26.
The 397B (`qwen35-397b-own`) is identified at 0.08 / 0.07. Full table:
`atlas_out/boundary_identifiability.{json,csv}`; figure `boundary-identifiability.svg` in the
note bundle (`--verify` passes).

## Distance-only (Toeplitz) surrogate (2026-09-05, CPU, exploratory)

A reviewer noted that the random-transport null cannot separate "three phases" from smooth decay
of similarity with layer distance. For each map, a Toeplitz surrogate replaces every cell by the
mean CKA at that layer distance; band statistics of the surrogate measure how much of the
statistic distance decay alone produces. `atlas_out/toeplitz_surrogate.{json,txt}`.

- **397B (release map, own probe):** mid_sep 0.343 real vs **0.275** surrogate (80% of the released
  statistic is reproduced by distance decay); fitted separation 0.407 vs 0.282, an excess of
  **+0.125**; fitted boundaries (13, 46) real vs (11, 48) surrogate. Shared probe: 0.386 vs 0.313,
  fitted 0.472 vs 0.326, excess **+0.146**, the largest in the zoo.
- **Zoo (shared probe, 36 maps):** excess fitted separation median **+0.017**, positive for
  **29 of 36**; seven lenses (gemma-3-12b, gemma-3-27b-it, gemma-4-e2b, gpt2-small,
  qwen2.5-7b-it, qwen3-1.7b, gpt-oss-20b) have no block structure beyond distance decay.

Reading: the fixed-thirds `mid_sep` is a weak discriminator of blocks against distance decay; the
397B's blocks are real departures from it (the largest we measured), but the released number
should be read with the surrogate's 0.275 beside it. Not pre-registered; added after review.
