# Review: "An Atlas of Depth" (jlens-cka-397b), fresh reviewer, 2026-09-05

Line numbers below refer to the working copy at 18:16 UTC (pre-blog HEAD `e82aa98`, 2,051 lines, 24,618 words including front matter; bundle directory clean in git). The note moved a long way today: eight pre-blog commits since the lens summaries were written, so several of their headline complaints are already closed. I verified each item below against the live file, the corrected ledgers, and the bundle before assigning a status.

## 1. Verdict

**NOT READY for reader review, but close, and the remaining work is prose, plumbing, and cutting, not new science.** The two analysis defects the lead reviewer confirmed (F1: the corpus and fit-budget map statistic was not CKA; F2: Test B and the ignition test fitted lens boundaries on own-vocabulary maps against pre-registrations naming the shared probe) are corrected in the research repo (commit `4db4361`) and the correction has now propagated into the note: corrections panel at L50-64, rewritten corpus section (L1124-1183), fit-budget section, Test B four-cell table on the shared probe (L932-935), ignition sentence at 0.415, Test C marked moot, claims rows 19 to 22 updated, the withdrawn-claims table extended to six rows, and the regenerated `corpus-dependence.*` and `fit-budget.*` figures committed alongside. What still blocks publication under the house guide's tier-1 rules: (a) two summary-surface sentences promise a control the note does not show and the key_result drops the body's central verdict; (b) the §5 provenance requirement is met for eight 397B numbers only, three of those eight still pin a dead URL, and 19 of 25 figure receipts are unindexed; (c) a handful of claims-matrix cells, one image title, and one figure caption still carry numbers or claims the note itself retracts; (d) the design count in the campaign verdict is wrong on its own enumeration. Behind those, the note is roughly twice the length its argument needs (main flow about 16,600 words before the appendices).

**The single sentence a reader would cite this note for:** across 36 public Jacobian lenses, the layer-by-layer CKA depth bands are a property of the lens's first-order readout geometry rather than of the residual stream or of computation: gemma-2-9b's flat lens (+0.005) sits on structured activations (+0.110 raw, +0.2086 standardised), deleting 16 readout directions restores its bands (0.014 to 0.162), and on 12 models the lens boundaries fail to track activation boundaries in all four pre-committed analysis cells (p = 0.613 on the pre-registered cell), while a lens direction still beats an equal-norm random direction 5x to 11x at individual inputs.

## 2. What is genuinely strong

- **The 2026-09-05 correction practice is exemplary and should be kept visible.** The wrong statistic is preserved behind `--legacy`, superseded outputs are kept as `*_legacy_selfgram.json` and `*_ownvocab_boundaries.json`, a run-time identity check against `common.cka.linear_cka` was added (agreement to 1e-6), and each ledger says which claims die and which survive. The note now inherits this: the corrections panel, the in-section correction at L1135, the withdrawn-claims table with "what caught it".
- **The 397B chain is fully machine-verified.** mid_sep +0.343363, null -0.000113, .937/.597/.590, 59 layers, 4,096-row probe, 2.0e-8 consumer gate all re-derive from `provenance.json`, the npz and `band.json`.
- **The Toeplitz surrogate is the right second control and it is honestly reported.** 0.275 of 0.343 is distance decay; +0.125 fitted excess is the zoo's largest; median zoo excess +0.017; seven lenses have none. The key_result now says "about a fifth of the released statistic is block structure beyond smooth decay". Few lens-CKA maps in the ecosystem ship one null; this one ships two.
- **The Gemma dissection chain is careful and receipt-backed at every step:** activation-CKA control (L445), random-Gaussian-probe readout specificity (51x, L491), k=16 ablation (L591), 34-model self-correction (L638). Every number checked matched its receipt.
- **The perturbation section is now one linear story with both doses on the table** (7.9x [7.0, 9.0], 11.2x [10.1, 12.4], 5.3x [4.7, 6.0] at the comparator dose; 7.5x, 9.1x, 8.1x at the bf16-matched dose), the three arms on one figure, the dose caveat of about 0.4 log units stated, and all three C values in float32 (0.053 / 0.077 / 0.026). The qwen float32 C run (`320cd39`) that lowered the qwen headline from 8.1x to 5.3x was run and reported rather than left as a footnote.
- **Test B is the model of how to handle a repaired instrument:** all four cells reported, non-blind amendment disclosed, both broken rules disclosed in one panel (L893-906), a per-model figure with both nulls, a uniform-null sensitivity in the ledger, and the identified-only restriction (L1006-1052) that shows where the null can and cannot be checked.
- **The boundary-identifiability section is a new methods contribution** (20 of 35 lenses identified; 397B spreads 0.08 / 0.07) and the note draws the right consequence: any boundary-based test needs a gate. The `jlens_atlas` tool that prints it, the shipped shared-token list, and the byte-exact gpt2-small selftest make the recipe reusable.
- **The prior-work / contribution / non-claims block (L91-123)** now exists, credits the eliebak explorer for the shared-string probe convention and for the code refits, and states the 24-prompt caveat as a non-claim. This is the block the two earlier reviews and the house guide asked for.
- **Summary (123 words) now leads with findings and carries the readout-geometry verdict.** The title now names the finding.

## 3. Blockers

### B1. F1: corpus/fit-budget statistic was not CKA (CONFIRMED; correction landed in the note; three residuals)

- **Location:** corpus section L1124-1221, fit-budget L1222-1293, atlas L199-222, control panel L272-277, Limitations L1349-1356, verdict L986-1003.
- **Problem and evidence:** the defect is real and documented (`corpus_dependence/results.md` top block; `results_fitbudget.json` gemma-3-270m ratios [14.20, 3.63, 0.98, 1.04, 0.55], shifts [3,3,0,0,0]; gpt2-small 25-prompt shift 4). Under the corrected statistic boundaries move 0/0/1 layers with corpus and 0/3/0 with seed; map distance ratios 84x/888x/86x; band_sep gpt2 0.0149 to 0.0050, gemma-3-270m 0.134 to 0.024; the 25-prompt caveat is triggered so the released 24-prompt 397B lens is caveated; Test C is moot. **All of this is now in the note and every number I checked matches the ledger** (corrections panel L50-64, table and captions, OQ4 L1325-1334, rows 19/20/22, withdrawn-claims rows 5 and 6). What remains:
  1. **The caveat is not placed where the map is presented, but L1355 claims it is** ("Under the frozen rule that caveats our 24-prompt 397B lens, and we say so where the map is presented"). The atlas paragraph L214-219 calls +0.343 "the strongest separation among the 36 models" and the L272-277 panel says only "one lens (n=24 prompts, one seed)". A reader who checks L1355 finds it false.
  2. **The one on-model piece of evidence for the caveat is absent:** the 397B interim read moved from mid_sep +0.3796 at n=16 to +0.3434 at n=24 (`fit_our_own/results.md` L194-197). Quote it beside the caveat.
  3. **The 8B corpus extension** is in the freeze table (`73fb7cb`, L1441) with no results; the pod is running now per `checkpoint.md`. Either slot the result when it lands or mark the row "pending" so the table does not imply data exists.
- **Fix:** one clause in the L272-277 panel ("a budget below the roughly 200 prompts the fit-budget sweep can vouch for, so under that section's frozen rule this map carries a budget caveat"), a parenthetical after +0.343 at L217, the n=16 to n=24 drift in the fit-budget section, and a "pending" mark on the 8B row. No compute.

### B2. F2: Test B and ignition used own-vocabulary boundaries (CONFIRMED; correction landed; three residuals)

- **Location:** claims row 21 (L1771), row 24 (L1774), OQ2 (L1309-1311), Test B panel L893-906, ignition paragraph L961-973.
- **Problem and evidence:** the deviation is real (PREREG.md and PREREG_B2.md name the shared map; `atlas_out/<slug>.npz` is own-vocab; 5 of 12 models differ by 3 to 9 layers). The corrected cells 0.715/0.554/0.881 (8), 0.601/0.568/0.613 (10, 4/10 beat own null), 0.743/0.561/0.965 (12), the shared-probe boundary table, the ignition 0.805 vs 0.415 sentence, and the disclosure of the deviation are all in the note now and match `results_B2.json` and `ignition_depth/results.json`. What remains:
  1. **Row 21 and OQ2 pair "12 models" with the 10-model cell** ("12 models, agreement 0.601 vs null 0.568, p = 0.613"). The 12-model cells give 0.743 vs 0.561, p = 0.965. One-line fix, but it sits in the claims matrix, which is the reviewer's instrument.
  2. **Row 24 still says "2 usable against a frozen minimum of 3."** The body (L966-969) now correctly says the pre-registration set no minimum and the floor of three was added at analysis time; the ledger says the same. The matrix cell contradicts both. Also, applied literally, the frozen rule returns NO ALIGNMENT (pooled gap 0.358 vs null median 0.228, p = 0.90, 0/2 beat own null); the note says so and keeps NO VERDICT, which is the conservative call, but the cell must not call the floor frozen.
  3. **No Test B or ignition entry in `receipts_index.json`** (it lists 2 receipts). `testB-per-model.receipt.json` now exists in the bundle; the ignition numbers are ledger-backed only. Index the first, add a receipt or an explicit ledger pointer for the second.
- **Fix:** rewrite row 21 as "12 run, 10 usable under the repaired gate: 0.601 vs 0.568, p = 0.613" and OQ2 to match; row 24 to "2 usable; the floor of 3 was an analysis-time choice (disclosed), not in the frozen design"; index the receipts.

### B3. Summary-surface parity: the lead promises a control the note does not show, and the key_result drops the body's verdict (merges F002, F014, F012, F018; §9.1 tier 1)

- **Location:** lead L17-20; zoo intro L282; key_result L21-34; atlas figure-note L205.
- **Problem and evidence:**
  - Lead: "The 397B map ships with the random-transport control that says whether its structure is real; the zoo maps carry that control for their per-layer dimensionality curves." The zoo's participation-ratio null exists in the gitignored npz files but is plotted nowhere: `build_zoo_figs.py` has no `pr_null` reference, `zoo-pr-curves.receipt.json` and its caption (L1575-1580) mention no null. L282 still says the zoo ran "the same random-transport null". That is exactly §9.1 item 2 ("never promise a metric you do not show"), and the last two reviews flagged "whether its structure is real" (R2-S01) as exceeding what the null tests (the body at L253-258 says the null is "necessary and not sufficient").
  - key_result: 224 words in one paragraph; opens on "three clean depth phases" (adjacent-layer CKA is 0.979 to 1.000 at every interior layer; the largest layer-to-layer turns sit at 45/46, 49/50, 55 to 58, inside the late block; the surrogate's own fitted segmentation is (11, 48) vs the real (13, 46)); carries the 2.0e-8 tolerance; and contains no sentence about the readout-geometry verdict, the 5x to 11x result, or the 24-prompt caveat. The summary now carries the verdict; the key_result box beneath it does not. The atlas figure-note's "sharp transition near layer 42" has the same problem (CKA to layer 50 climbs 0.66 to 0.97 across layers 41 to 46, a five-layer ramp).
- **Fix:** lead: "ships with two controls, a random-transport null and a distance-only surrogate, that say whether its blocks come from the fitted Jacobians and whether they exceed smooth decay"; delete the dimensionality-curve clause or plot `pr_null` as a grey band in `zoo-pr-curves` first; drop "the same random-transport null" at L282. key_result: three short paragraphs of about 110 words total: (a) map, two controls, "three depth regimes", the fifth-beyond-decay sentence; (b) probe lesson with gemma-3-12b 0.001 to 0.114; (c) the verdict: five pre-registered designs and Test B in 4/4 cells fail to show the bands partition computation; the atlas is first-order readout geometry; lens directions beat equal-norm random 5x to 11x; 24-prompt caveat. Move 2.0e-8 to Integrity. Change L205 to "with the mid-to-late handover concentrated in layers 41 to 46".

### B4. Provenance plumbing (merges F003 and F004; §5 tier 1)

- **Location:** Status line L43-48; `provenance.json`; `receipts_index.json`; `zoo-receipts-index.json`; artifact ledger L1478-1488; `build_cka_post.py`.
- **Problem and evidence:** the false universal sentence is gone; the Status line now says the eight 397B numbers re-derive and other numbers "name the ledger". That is a §5 rule-3 named gap, and the guide says a named gap closes before publication. Concretely: `provenance.json` has 8 entries, all 397B; the `within_mid`, `early_mid`, `mid_late` entries still carry `pinned_url` at `c6c7bb1/.../artifacts/lenses-397b/qwen35_397b_dm.band.json`, which 404s (`artifacts/` is gitignored; `git cat-file -e` fails), and `consumer_gate_delta` depends on that file; the ledger row now points at Hugging Face `blob/main/band.json`, which resolves but is not revision-pinned (the HF commit `a31f2f4` "add band.json" is value-identical on all ten keys). `receipts_index.json` indexes 2 of 25 receipts, `zoo-receipts-index.json` 4. `--verify` is a substring test over 8 values (0.965 and 0.415 each appear in two unrelated places), has no URL/cat-file check, and is not wired into the pre-blog Makefile. Every number after the map sections (Test B cells, perturbation ratios, corpus/fit-budget, decomposition, identifiability) is receipt-backed in the bundle or ledger but asserted by no verifier, which is how superseded numbers survived from July to September.
- **Fix:** re-point the three band entries and the ledger row at `huggingface.co/.../blob/a31f2f4/band.json` after a SHA-256 check against the local receipt (`d3722e17...`); index all 25 receipts; extend the manifest to the evidentiary numbers after L900 (sources exist: `results_B2.json`, `geometry_causality/out/analysis_fp32_ci.json` and `qwen3.5-0.8b_C32.json`, `corpus_dependence/results.json`, `results_fitbudget.json`, `boundary_identifiability.json`, `toeplitz_surrogate.json`); make the verifier anchor each value to its section, assert `git cat-file -e` for every research-repo pin, and run it from `make`. Do not loosen the `artifacts/` ignore (the 1.98 GB `.pt` lives there).

### B5. Retraction marking in pull-able artifacts still incomplete (merges F005, F024; §9 rule 6 tier 1)

- **Location:** zoo-mid-vs-fitted lead-in L329-331, alt L333, Finding sentence L334-338; gemma-slope caption L409-414 and image title; all 25 inline alts.
- **Problem and evidence:**
  - Mid-vs-fitted: the image title is now correct ("not always ... 5 of 36 below the line"), the glossary L188-191 is correct, and the correction paragraph now lists the own-vocab violators correctly (gpt2-small +0.015/+0.001, qwen3-1.7b, gemma-4-e2b, qwen2.5-7b-it, gemma-3-27b-it). But the alt text still says "every model on or above the y equals x line", the Finding sentence still says "every one of the 36 models lies on or above the line ... always finds at least as much separation", and L329-331 still says "the whole zoo confirms it", eight lines before the retraction.
  - Gemma slope: the regenerated image title says "10 of 17 Gemma lenses revive on the shared probe"; the caption says "thirteen of seventeen Gemma lenses reveal real band structure". The receipt supports ten under the generator's own 0.02 rule; the three clay lines that do not clear it are gemma-3-4b-it (0.0146 to 0.0126, falls), gemma-3-1b-it (0.0120 to 0.0194) and gemma-3-4b (0.0455 to 0.0607). Figure, title and caption disagree on the section's payoff count.
  - 25 of 25 inline alt texts differ from their receipt `alt_text`; the pt-vs-it alt still says "Seven" curves (L731) for eight.
- **Fix:** rewrite L329-338 to the 31/5 statement; set the gemma-slope caption to "ten of seventeen rise by more than 0.02 ... three gemma-3 lenses barely move ... the four gemma-2 small checkpoints stay below 0.01 on both probes"; make colour classes and title derive from one rule in `build_shared_vocab_figs.py`; sync inline alts from receipts by script.

## 4. Majors

### M1. Design and model counts contradict the note's own enumeration (F008 residual)
L844-846 says "three pre-registered patching designs across six models ... then the three further tests below"; the heading at L882 says "Three more attempts". That is six designs. L988, L1309 and row 13 say "Five designs, eleven models"; the blockquote at L995 says "five increasingly careful attempts have failed" (which counts the moot Test C as a failed attempt, against the ledger's explicit instruction); L1001 still lists ignition-depth alignment among "the ideas we can still think of" after L961-973 reports it ran. Eleven is the pre-re-run tally (six patching plus the original eight-model Test B); the union across all designs is fifteen (six patching, twelve Test B, three Test C, six ignition). **Fix:** "Six designs, fifteen models; one moot, one without a verdict" at L988/L1309/row 13; blockquote "four attempts failed to show it, a fifth lost its premise, a sixth produced no verdict"; drop ignition from L1001.

### M2. Mechanism language the body's own scoping forbids (R1-B03 / R2-B04 / F059; unchanged since July)
L441 "The model-versus-fit question is already settled (it is the model)" contradicts L434-438 (same recipe and corpus, so fit noise is excluded, not corpus); L550 "pulls the output-facing geometry toward a common answer-producing direction"; L562 "tuning a model to answer rewrites its early layers"; row 11 "restores structure" (the body at L591-637 says it is an operation on the statistic); L219 "late-stack blocks where computation turns toward committing outputs"; L253 "Whatever phase organization appears in the real atlas was put there by the fitted Jacobians"; heading L1053 "The lens directions are real". Third review to flag these. **Fix:** "settled for fit noise, not for corpus"; "is associated with"; "unmasks structure in the statistic"; bound L219 and L253 to readout geometry; retitle L1053 "beat an equal-norm random direction by 5x to 11x".

### M3. Length and structure (F015)
24,618 words; main flow before the first appendix about 16,600. The own-vocabulary H2 is 4,345 words with five H3s of gemma-2 dissection; the blocks H2 is 4,668; the corpus H2 2,100; the campaign appendix 1,596 retells the v1/v2 tables already in the main ledger table. The Status line (L46) still says "instrument note" for what the guide defines as a study report. The site's longest published note is 16,535 words. See §9 for the plan.

### M4. Contact sheets fail the legibility gate (F021)
`zoo-contact-shared.svg` and `zoo-contact-sheet.svg` are still 936 x 979 pt with 8.2 pt titles and 7 pt sep labels; the note uses zero `{{< figure >}}` shortcodes, so every image renders in the 44rem paragraph measure (about 704 px), where the titles are about 6 px and unreadable on a phone. The shared sheet is the figure the note calls "the single most informative view". **Fix:** wrap in a figure element (60rem), rebuild as two 18-panel plates at about 740 pt with 10 pt labels and a shared colorbar, and link `shared_summary.csv` from the caption.

### M5. Draft-history residue and one false "published" (F017 residual)
L1785 "published in earlier drafts and withdrawn" and L1848 "we published and then withdrew six claims": the note has been `draft: true` at every commit; readers will take this as a public retraction record. L710 "(see the interactive explorer below)" resolves only to a References entry 1,300 lines later. L1615 "our approved follow-up" is chat residue. L421 and L704 cite "an earlier independent run" that is the org's own 2026-07-10 sweep (`emergence_shared.csv`, same probe, same code); R1 asked for an identity and independence statement. The campaign-note links at L434 and L703 point to `/posts/2026/07/the-j-space-atlas-campaign-pre-registration-and-first-results/`; the sibling directory is `jspace-atlas-campaign`, it sets no slug, and it is `draft: true`, so the link is dead now and will be dead at the current slug later. **Fix:** "withdrawn before publication"; name the explorer inline with its URL; "a follow-up on the public optimizer-sweep checkpoints, frozen before data, would test this"; "our 2026-07-10 sweep, same pipeline" without "independent"; "a companion note (forthcoming)" instead of a link.

### M6. Uncited ecosystem claims (R1-R01 / R2-R02 residual)
L1472 "recent community J-lens releases at ~33B and ~950B scale ship exactly this, using the same probe recipe" has no reference entry (the memory note names PrimeIntellect's releases; cite the repos or drop the sentence). The Neuronpedia "on the order of a thousand WikiText prompts" fit budget (L1225, L1350, L1796) has no metadata source and now carries the fit-budget verdict's weight. L1572 still says the participation ratio was "introduced for neural dimensionality by Gao et al." while the reference entry disclaims exactly that.

### M7. Census statistics still phrased as laws (R1-S01 / R1-S02 / R2-S03 residual)
L534 "instruct tuning flattens the readout, in every family, monotonically" (three families have base-instruct pairs); L649 "-0.58 ... the strongest single correlate of lens flatness in this whole note" with no candidate set or multiplicity context; L665 gpt2-small and pythia-70m are flat "simply because they are too shallow"; no bootstrap or cluster intervals for rho -0.59 or +0.44 (the shared-probe scatter caption L316-321 does bound +0.53/+0.44 as census, which is the right pattern); row 8's "rho -0.16 instruct" appears nowhere in the body.

## 5. Minors and nits

| id | location | issue | fix |
|---|---|---|---|
| F006 | row 21 L1771, OQ2 L1310 | "12 models" paired with the 10-model cell | see B2 |
| F007 | row 24 L1774 | "frozen minimum of 3" | see B2 |
| F018 | key_result L22, L205 | "three clean", "sharp transition"; surrogate's own segmentation (11, 48) not shown | "three regimes"; add (11, 48) to the surrogate figure-note |
| F058 | L1575-1580, L731 | PR-curves shows no null though the lead now cites one; pt-vs-it alt "Seven" for eight curves; "readout widens as the model approaches committing an output" is the R1-F02 interpretation asked to be dropped | plot `pr_null`; "Eight"; drop the parenthetical |
| F062 | L421, L704 | "earlier independent run" is the org's own sweep | see M5 |
| F063 | L793-796 | Gemma recap still says "a 256k-token vocabulary is associated with a CKA floor near 1.0 ... crowding is the natural explanation" after the summary and key_result stopped saying so | "on some models the own-vocabulary sample lands where the floor sits near 1.0; the comparison does not isolate why" |
| F069 | fit-budget verdict | PREREG's "still falling at 400" branch: gemma-3-270m falls 1.40e-4 to 7.3e-5 from 400 to 1,000, both inside the seed null; the analyzer resolved it favourably without saying so | one sentence stating the clause was evaluated |
| F071 | row 8 | -0.16 unreceipted in the bundle and absent from the body | cite the 8/8 paired result instead |
| F075 | L434, L703, L1755 | "Tier-1" nickname; campaign link dead (see M5); no glossary links at first use of jacobian lens, residual stream, unembedding, bootstrap, bf16 | link or gloss |
| F077 | all SVGs | 25/25 inline alts differ from receipt alt_text; hero figures still text-as-paths | script the alt sync; set `svg.fonttype none` on the hero pair too |
| F079 | L398 alt vs L400 caption | alt "256k vocabulary", caption "262k-token vocabulary" for gemma-3-27b; neither says "4,096 rows sampled from" (R2-F01) | "4,096 rows sampled from its 262k-token vocabulary" |
| F080 | L483-487 area | hedge on gemma-2 activation structure is a raw-map artifact; standardised +0.209/+0.208 exceed gemma-3-1b 0.142, llama3.1-8b 0.139, qwen3.5-0.8b 0.177 | restate on the standardised instrument |
| F082 | front matter | `date: 2026-07-18` predates most content | set at publication |
| new | L1441 | freeze table lists the 8B corpus extension (`73fb7cb`) with no data yet; pod running | mark pending |
| new | og-card | tagline now "8 lineages . 5,700x scale . two controls": accurate | none |
| new | L46 | "instrument note" | "study report" |

Closed since the lenses ran and verified closed: F011 (key_result scale sentence), F016 (Step 3 before the skippable detail), F056 (37 tokenizers), F057 ("one seeded realisation"), F061 (eleven- to nineteen-fold), F065/F066 (perturbation timeline and mixed precision), F067 (outliers to appendix), F068 (corpus map pairs), F070 (ledger rows), F072 (0.29 correctly attributed), F073 (splice), F074 (title), F076 (0 KaTeX-in-backticks), F078 (bundle committed), F081 (OQ2 reason), F1-F5 firsthand stale-text items except those listed above.

## 6. Refuted findings

- **F020** (superseded conclusions hardcoded in zoo figure titles/footers): fixed in `c35f9b4`; the own-vocab sheet is labelled "not comparable across models", band-by-scale's title is data-derived (+0.33), the shared footer says "token-set confound"; SVG hashes match receipts.
- **F022** (Qwen ".23 to .41" range excludes four lenses): the caption now says "seven of the eleven Qwen lenses ... three sit near zero", which is exact against `summary.csv`; the proposed ".21" floor would admit an eighth lens.
- **F023** (mid-vs-fitted violator numbers were shared-probe values): the correction paragraph now lists the own-vocab violators and names the shared-probe five separately; the receipt and `zoo-provenance.json` carry all 36 pairs.
- **F012 sub-claim "double negative"**: none exists in the key_result.
- **F002 sub-claim** that this note's og-card says "every map ships with its null": that string is the campaign note's card; this card now reads "two controls".
- **F003 sub-claim** that `zoo-provenance.json` has zero numeric entries: it holds 108 per-model values; what it lacks is receipt paths, hashes and a working verifier (kept in B4).
- **F016's proposal** to keep the CKA derivation in the main flow: the author moved it to an appendix in `4f772a7` and kept Steps 1 to 3 in place; the lead's "walks through the calculation step by step" remains true. No objection.

## 7. Prior-review closure

| item | status | remaining |
|---|---|---|
| R1-B01 scales, counts, statistic identities | partial | mid-vs-fitted alt and Finding sentence still assert the ordering (B5); everything else fixed |
| R1-B02 vocabulary-size mechanism | mostly fixed | L793-796 recap wording (F063); summary, key_result, panels and figure footers now say probe effect |
| R1-B03 geometry vs mechanism conflated | open | L219, L253, L441, L550, L562, row 11, heading L1053 (M2) |
| R1-B04 perturbation estimand | fixed | estimand, comparator relabel, intervals at comparator dose, both doses tabled |
| R1-S01 census not population | partial | L534 "in every family, monotonically" (M7) |
| R1-S02 narrower statistics | partial | L649 superlative, no intervals for zoo correlations |
| R1-S03 open questions triage | fixed | |
| R1-R01 bibliography | partial | 33B/950B uncited; "earlier independent run" unidentified; explorer linked but unpinned |
| R1-R02 missing support | partial | Gao "introduced" at L1572; freeze-commit table now present (fixed) |
| R1-F01 captions overstate | partial | mid-vs-fitted caption/alt (B5) |
| R1-F02 CKA as percentage | partial | PR-curves parenthetical at L1577 |
| R2-B01 probe dependence as vocab-size causation | mostly fixed | L793-796 |
| R2-B02 corpus overgeneralised, shift undefined | fixed (superseded by the correction) | section rewritten to 0/0/1 with the seed null |
| R2-B03 perturbation uncertainty | fixed | float32 intervals for all three, C precision uniform |
| R2-B04 measurement promoted to mechanism | open | as R1-B03 |
| R2-S01 "computational phases", "real structure" | open | lead L17-18, key_result "three clean depth phases", L219, L253 (B3) |
| R2-S02 CKA rationale, isotropic probe | fixed | |
| R2-S03 zoo correlations as laws | partial | L665 "too shallow" |
| R2-S04 chronology and counts | partial | five/eleven vs six/fifteen (M1); forward pointers, garbled sentence, "twice", withdrawn count all fixed |
| R2-R01 entry roles | partial | Gao; Neuronpedia budget source |
| R2-R02 missing references | partial | 33B/950B; Gao details |
| R2-F01 label inconsistencies | partial | gemma-own-vs-shared alt/caption 256k vs 262k, sampled-rows wording |
| R2-F02 alt texts | partial | pt-vs-it "Seven"; 25/25 inline alts differ from receipts |

Tally: 6 fixed, 14 partial, 3 open (all three open items are the mechanism/real-structure wording cluster, now in B3 and M2).

## 8. Contribution ranking

1. **Bands are readout geometry** (activation-vs-lens dissociation, 51x readout specificity, k=16 ablation, Test B 4/4 on 12 models, identified-only restriction, identifiability of boundaries). Most defensible, most novel, survives every correction unchanged. Prominence: now in the title and summary (right), absent from key_result and lead (wrong, B3); the body evidence arrives at L445 and the verdict at L986.
2. **The 397B map with two controls** (random-transport null -0.000113; surrogate 0.275 of 0.343, +0.125 excess, zoo median +0.017). An artifact contribution and the only lens-CKA map in the ecosystem with a null, now with a second one. Prominence right; the wording ("three clean phases", "says whether its structure is real") overstates and the 24-prompt caveat is missing at the map (B1, B3).
3. **Equal-norm perturbation validity** (7.9x/11.2x/5.3x with intervals at the comparator dose; 7.5x/9.1x/8.1x at the larger dose; C 0.053/0.077/0.026). The one positive pre-registered result and the one an instrument-builder cites. Prominence now adequate; the heading's "are real" should go; the dose caveat is honestly stated.
4. **Boundary identifiability** (20 of 35 identified; 397B 0.08/0.07; nine lenses with spreads 0.41 to 0.85). New methods point that reframes every boundary-based result in the ecosystem, including this note's. Under-promoted: one sentence in key_result would be earned.
5. **Shared-probe lesson** (gemma-3-12b 0.001 to 0.114; gemma-3-27b 0.025 to 0.298). Useful to Neuronpedia and explorer readers, now correctly credited to the explorer's convention, and the sibling methods draft already carries it. Over-weighted at 4,345 words.
6. **Fit-budget scoping** (converged from 200 prompts; 25-prompt caveat triggered at 14.2x; released lens caveated) and **corpus dependence** (84x/888x/86x on the map; boundaries 0/0/1; band strength changes). After correction these are seed-null-quantified confirmations of the explorer's public observation, and the note now says so. Prominence now about right; the new map-pair figure is what makes the 888x legible.
7. **Instruct n=8 and outlier census**: exploratory; outliers correctly moved to the appendix; instruct (370 words) can follow.

## 9. Structural recommendation

**One note.** The title now commits to it ("and What Their Bands Are Not"), the prior-work block gives the four contributions a home, and a split would put the 397B map into an already-published post and trigger the §9.3 ceremony there. But cut it to about 13,000 to 14,000 words total with the main argument under 10,000, from 24,618 and about 16,600 today. Concretely:

- **Own-vocabulary H2 (4,345 words) to about 1,800.** Keep activation-vs-lens, readout specificity, and the k=16 ablation in the main flow as the three-step proof; move "What actually predicts a flat readout" (the 36-lens regression) and the 34-model check to the existing gemma-2 appendix with a 100-word summary each; delete the recap at L790-800; fold the instruct n=8 section (370 words) into the same appendix.
- **Blocks H2 (4,668) to about 2,800.** Merge "Are the blocks computational?" and "Three more attempts" into one section: the elimination ledger table, the four-cell table, one paragraph each for ignition and Test C, the blockquote. Keep identifiability and the perturbation section whole. Cut the campaign appendix (1,596) to about 800 by dropping the v1/v2 table retelling that duplicates the main ledger table.
- **Corpus H2 (2,100) to about 1,500.** The correction panel, the table, the map-pair figure, the fit-budget figure and the boxed lesson carry the section; "What this does and does not weaken" can be three sentences now that the boundary story is gone.
- **Consolidate the tail:** merge Unknowns #4 with the Limitations fit-heterogeneity paragraph; fold "What pre-registered means", "Precision", "Integrity" and the artifact ledger into one Reproducibility section keeping the freeze table.
- **Order of operations:** land B1 to B5 and M1 first (an afternoon, no compute), re-run every generator's `--verify` plus the extended manifest, then restructure, then re-check that the corrections panel's "disclosed in full where it occurs" pointers still land. Do not hold the note for the 8B corpus fits or the 397B split-half; slot them as dated additions.