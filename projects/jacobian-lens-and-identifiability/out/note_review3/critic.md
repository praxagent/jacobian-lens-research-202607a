# Completeness review: what the eight lenses missed

Checked against the note as it stands now (index.md mtime 2026-09-05 18:16, eight commits after the state the lens summaries describe), the research repo at 8246f62, the live HF cards, and the live Anthropic page.

## 1. The HF model card the note points to contradicts the corrected note (highest priority)

The Status line (L43) calls this "the instrument note for the released lens" and links `praxagent-org/jacobian-lens-qwen3.5-397b-a17b`. The live README (fetched today; identical to `fit_our_own/MODEL_CARD-397B.md` plus a 2026-07-18 CKA section) still says: "n=24 is safely in the converged regime" (L56) and "Converged for the band statistic per our calibration" (L105), while the note now says the 25-prompt caveat is triggered for this lens (L60-62, non-claims L118-120); "the strongest 'workspace band' we have measured on any model ... 1.6× ... Qwen3-14B: 0.211" (L22-30), an own-vocabulary comparison the note calls not comparable across rows (L1839); "The workspace band grows to frontier scale" (L30-31), a scale law the note declines (L121-122); "the block structure is a property of the fitted lens, not of the shared unembedding" (L135-136), which the distance-only surrogate now qualifies (80% is decay); and "extracts workspace content (near-perfectly)" (L74), the functional reading the note refuses. Nobody opened the card. AGENTS.md §9.3 item 5 requires downstream surfaces and citing siblings to update in the same sitting. **Follow-up:** a card-vs-note reconciliation pass; regenerate the card's headline block from the claims matrix; same check on the published release note (L748 and L1461 "converged by n≈16").

## 2. Three pre-registered Tier-1 arms were run, ledgered, and left out of the note

`jspace_atlas/TIER1_PREREG.md` froze arms A-D; `jspace_atlas/results.md` (ledger added 2026-09-05) records all four. The note reports only arm A (L429, claims row 5). Arm B (Qwen3-4B fitted on code vs WikiText) **failed its frozen prediction** (median 0.945 vs predicted drop below 0.7) and is directly relevant to the corpus section (L1124-1221), which never mentions it; the ledger even carries a reconciliation note that belongs in the note. Arm C (FP8 vs bf16, median 0.99989) is a robustness result the Precision section (L1443-1459) omits. Arm D (optimizer outlier census, partial) was run, yet L1613-1616 still says the optimizer follow-up "will freeze and test". A failed pre-registered prediction absent from a note that advertises five designs is a selective-reporting exposure. **Follow-up:** a pre-registration completeness audit: every `PREREG*.md` and every Tier-1 arm maps to a claims-matrix row or a stated omission.

## 3. A pre-registered 8B corpus extension is running right now and the note is silent

`corpus_dependence/PREREG_8B.md` (frozen today, 73fb7cb) tests whether "map moves, boundaries do not" survives at llama3.1-8b; `checkpoint.md` L3-13 shows pod aln7lne2jgdcnv fitting the three lenses (~9-10 h, ~$5). If P2 fails, the corpus headline on L9, L1124 and L1180 reverses at 8B. The prereg's cost gate (skip if >8 h) was declined and overridden by the lead (`scratchpad/llama_fits.sh` header); that deviation is not yet in the ledger. **Follow-up:** hold the corpus section until the receipt lands; re-run number-consistency on L1124-1221 and claims rows 19-20 afterwards; record the gate override in `results.md`.

## 4. Anchor 3 cites the behavioral ledger against that ledger's own verdict

L813-833 uses the shared-probe correlations (+0.534, +0.524) as the reason "the middle block's geometry is load-bearing". `behavioral/results.md` L76-110 headlines the opposite: on the shared probe, matched-band pairs (gemma-2-27b 0.1134 vs qwen3.5-2b-pt 0.1143) resolve 0.000 vs 0.970; "Gemma does not uniquely lack the band, it uniquely lacks the function"; band not necessary and not sufficient. Grep of the note for matched-band, never resolves, or family-bound returns nothing. F037 fixed receipts and probe, not the omitted verdict. **Follow-up:** claims-vs-ledger for every cross-repo citation (behavioral, lens_demo, the reader benchmark).

## 5. The Anthropic quotation omits the source's own hedge on sharpness

L746-762 quotes the block-structure and bookend-logic sentences. On the live page the very next sentence is: "We note that in some models the transition is more gradual, sometimes containing sub-blocks, and that the observed sharpness is exaggerated by layer subsampling." The note never quotes it (grep gradual/subsampl: none) and presents the distance-only surrogate and the "regimes not stages" reading as its own correctives. The "quotations checked" assertion is true of what was quoted; the selection is unfair to the prior work and weakens the note's F018 defence. **Follow-up:** quote it in the §3.2 block and beside the surrogate.

## 6. The cost narrative is wrong and there is no compute statement

L1410 "not the $0 of this note" is false: repo ledgers show roughly $10 of GPU behind the note's results (corpus/fit-budget $3.64, Test B/C $3.20, ignition <$1, geometry causality ~$1, block patching ~$1.10, activation-CKA $0.15, readout decomposition $0.10), plus the 8B pod, and the Tier-1 ledger says "Cost was not recorded", a repo cost-discipline gap. **Follow-up:** one compute paragraph with per-experiment cost from the ledgers; fix L1410; record or declare unrecorded Tier-1 cost.

## 7. The research README the note links ~15 times has a stale status table

`README.md` lists five experiments; `block_patching`, `geometry_causality`, `jspace_atlas`, `lens_demo`, `readout_decomposition`, `sae_x_jspace` have zero mentions. A reader following any pinned link lands on a map that omits most of the note's evidence. **Follow-up:** regenerate the table, one row per experiment with verdict and ledger link.

## 8. A "Corrections" panel on a never-published draft (judgment call)

§9.3 governs corrections "after publication". The panel (L52-66) and "an earlier draft ... reported" passages read to a first-time reader as if wrong numbers were published. The public record that did carry them is the research ledger (GitHub), which already has the correction blocks. Consider stating the corrected results plainly with one sentence pointing to the ledger entries. Needs TJ's ruling.

## 9. "Identifiability" collides with the project's own term

The project README and the sibling draft `jacobian-support-and-identifiability` use the nonlinear-ICA sense; this note (L110, L1006-1052) uses estimation sharpness. One disambiguating clause.

## Checked and not gaps

All pinned hashes in the note (c6c7bb1, 51d70aa, 241fa68, de50888, 271376f, 09e5bf7) are reachable from origin/main. Heatmaps use magma and RdBu_r, so colour-channel accessibility is fine beyond F021/F077. The gemma-2-9b HF card matches the note (0.993, 0.0059).

## Lens to run next

A **downstream-surface sync lens** (HF cards, release note, campaign note's "500×" vs "5,700x", three-ways-a-lens-fools-you) and a **pre-registration completeness lens** (gaps 2-3), both after the 8B receipt lands.