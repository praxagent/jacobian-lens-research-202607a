# Results ledger — reader benchmark (probe_calibration)

Frozen design: PROTOCOL.md v2 @ freeze `6830d9c` (+5 logged pre-outcome amendments:
`d6eb079` GPU probe, `04578e9`/`8ffb9ea` mini-batch OOM fix, `abac95e` functional final-norm,
plus runbook dep pins). Gates: `gates.py` ALL PASS before any arm trusted. Only runs actually
executed are recorded here; receipts are sha-verified before pod termination.

## Arm 1 — Llama-3.1-8B-Instruct (2026-07-15/16)
Pod a8lpvfptzc4k5l (L40S $0.99/hr, ~7h incl. 3 OOM debug cycles ≈ $7). Splits 510/64/989
completions → 7,321/896/15,891 spans (test prevalence .255).
Receipt `receipts/receipt_llama31_8b.json` (sha-verified, commit `6b5824a`).

| reader | AUROC [95% CI, completion-clustered] | paired verdict vs probe (±.05) |
|---|---|---|
| attention_probe (3 seeds .753–.754) | **.754** [.743,.765] | — |
| native_head_surprisal | .722 [.712,.733] | **equivalent** |
| sae_latent_label_selected (Goodfire l19, latent 48223 of 65,536, val-selected) | .629 [.618,.641] | worse |
| heuristic_len_freq | .604 [.589,.618] | worse |
| logit_lens (L19) | .537 [.525,.551] | worse |
| random_transport_null | .518 [.505,.531] | — |

## Arm 2 — gemma-2-9b-it (2026-07-16)
Pod 6rjs40i0nqw80x (L40S SECURE $0.99/hr ≈ $4; a prior community-cloud L40 never exposed
ssh — terminated, ~$0.45 lesson). 11,120 test spans (prevalence .184).
Receipt `receipts/receipt_gemma2_9b.json` (sha-verified, commit `f0350f7`).

| reader | AUROC | verdict |
|---|---|---|
| attention_probe | **.726** [.712,.740] | — |
| native_head_surprisal | .637 | worse |
| heuristic_len_freq | .597 | worse |
| logit_lens (L20) | .576 | worse |
| random_transport_null | .546 | — |
| sae_latent (gemma-scope-9b **PT** on IT model, l20/16k/l0_68) | .509 | ≈ chance — the pre-disclosed PT→IT transfer caveat |

## Arm 3 — Llama-3.3-70B-Instruct (2026-07-16)
Pod fbli1ym7v85478 (2×H100 $5.98/hr, ~4.7h ≈ $28 incl. 4 dep-fix cycles: SAE filename .pt,
transformers<5 pin, pkill self-match, accelerate; then amendment-5 rerun). 19,890 test spans
(prevalence .161). Receipt `receipts/receipt_llama33_70b.json` (sha-verified, commit `e96ac3f`).

| reader | AUROC | verdict |
|---|---|---|
| attention_probe | **.774** [.761,.787] | — |
| **exploratory_probe_capacity** (16 heads/400 ep; post-outcome label) | *.776* [.763,.789] | **capacity does NOT explain the paper gap (+.002)** |
| sae_latent_label_selected (Goodfire l50) | .730 [.718,.742] | **inconclusive** (nearly ties) |
| native_head_surprisal | .695 | worse |
| heuristic_len_freq | .646 | worse |
| logit_lens (L50) | .518 | worse — *below* the null |
| random_transport_null | .548 | — |

## Grader validation (Gemma-3 labeling gate) — FAILED, demotion fired
`../gemma3_labeling/receipts/val_deepseek{,_v2,_v3}.json` (commits 8026619/eb0987a/00c4380).
DeepSeek + pinned local Wikipedia (dump 2026-07-01): κ .281 (over-flags) → rubric fix κ .278
(105/130 gold positives = Insufficient: intro-only retrieval ceiling) → full-text index κ .179
(under-flags). Total $0.22. **Wikipedia-only deterministic grounding ≠ live-web gold** on
LongFact entities. Per PROTOCOL §8: Gemma-3-12B → label-free reader-agreement arm
(`../gemma3_agreement/`), label release cancelled (gate condition unmet).

## Arm 4 — Gemma-3-12B-IT label-free agreement (2026-07-16, running)
Pod v3talw87lael1q (L40S $0.99/hr). Runner `../gemma3_agreement/run_agreement.py` @ f560e0c.
Receipt row to be added on completion.

## Cross-arm findings (3 labeled arms)
1. Probe stable .726–.774 across 8B/9B/70B; the paper's .94-class number absent at every
   scale on public labels. 2. Capacity ruled out on 70B (+.002 exploratory). 3. Free native
   confidence ties the probe ONLY on Llama-8B; supervision earns ~.08–.09 on gemma-9b/70B.
   4. SAE tracks model-match: PT-on-IT ≈ chance; model-matched l50 nearly ties supervision.
   5. Mid-layer lens ≤ null everywhere: entity signal is output-adjacent.
Total spend: ~$40 GPU + $0.22 grader.

## Arm 4 — Gemma-3-12B-IT label-free agreement (2026-07-16, COMPLETE)
Pod v3talw87lael1q (L40S $0.99/hr, ~6.5h incl. 3 fix cycles: openai pkg, runbook SHA revert,
Gemma Scope 2 lowercase keys; completion cache made reruns resume instantly ≈ $6.5 total).
300 greedy completions → 6,503 extracted entity spans (DeepSeek extraction $0.11, capped).
Receipt `../gemma3_agreement/receipts/receipt_gemma3_agreement.json` (sha-verified).
SCORE AGREEMENT ONLY (I05): Spearman — logit_lens~random_null **.724** (mid-layer surprisal
shares most variance with a random transport: generic norm/frequency, not content);
logit~native .142; native~sae −.298; heuristic ~0 with everything. **The label-free readers
do NOT converge** on which entities are suspect: without supervision there is no shared
"hallucination signal" to read off Gemma-3-12B's residuals at mid-depth — consistent with
the labeled arms (supervision or a model-matched sparse dictionary is what earns signal).

## STUDY COMPLETE — all four frozen confirmatory arms run and reported
Total: ~$47 GPU + $0.33 grader/extraction (envelope was $75–180). Conditional Qwen-397B
extension: decision to TJ per the pre-declared criterion.

## Conditional Qwen-397B extension — NOT RUN (cost decision, TJ 2026-07-16)
Per the pre-registered §11A pattern: the go criterion was "a consistent, interpretable reader
ordering worth confirming at scale." The observed data: the only scale trend is the SAE
closing on the probe (.63→.73, 8B→70B), but Qwen-397B has no public SAE, so the extension
would have tested the J-lens — and lens-class readers were null in all four arms. Expected
value of ~$50–150: a well-measured null. TJ declined. Reported here as required: extension
not run (cost), criterion + full cheap-arm results above so the reader can judge.

## Jury-labeling validation (2026-07-16) — GATE PASSED
Cross-provider 3-judge jury (DeepSeek, GLM-4.6, Gemini-2.5-Flash) on shared archived Serper
evidence, validated vs public gold (40 completions, 566 free-tier searches, ~$0.40 judges).
Receipt `../gemma3_labeling/receipts/jury_validation.json` (committed):
per-judge kappa .41/.47/.48 (search-grounding alone ~doubled the Wikipedia ceiling of .28);
**majority tier kappa .598 (n=229, coverage .40) — PASSES the pre-registered >=0.5 gate**;
unanimous tier kappa .691 (n=87, coverage .15). Coverage caveat: jury labels cover the
~40% of spans where >=2 judges agree on a definitive label; splits/insufficient are
excluded and counted (disclosed in any release). CLEARED to label Gemma-3-12B and prep the
HF release, pending TJ loading the $50 Serper pack (~6,500 entities > 1,930 remaining free).
