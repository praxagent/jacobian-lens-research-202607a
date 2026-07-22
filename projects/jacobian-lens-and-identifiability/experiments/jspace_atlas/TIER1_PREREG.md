# Tier 1 pre-registration (frozen before any GPU run)

Status: DRAFT design, frozen on commit. **No pod launches until TJ gives explicit
go for the specific run + cost.** All four arms fit small models (4-9B); the whole
tier targets the ~$10-30 GPU envelope TJ already approved for Tier 1.

House-rule compliance: design (models, corpora, prompt-length matching, metrics,
gates, predictions) is committed here BEFORE any lens is fitted. Cheap-validation
ladder (repo CLAUDE.md rule 5): every fit runner CPU-smoked on gpt2 first; the
gemma-2-9b arm doubles as a correctness check (must reproduce Neuronpedia's gpt2 lens
via compare.py before any paid fit is trusted).

## Shared method (all arms)

- Fit runner: `fit_our_own/fit_lens.py` (Anthropic `jlens.fit`, Apache-2.0), checkpointed.
- Analysis: within-model shared-vocabulary CKA map (`shared_maps.py` path) + fitted
  segmentation + random-J null, identical to the atlas.
- Probe: the 4,096-string shared vocabulary (`shared_tokens.json`), fp16 store / fp32
  CKA, seed 0. Same convention as the whole campaign.
- Every fit saves the raw ingredients per the house receipt rule (per-prompt args,
  seed, corpus hash, package versions, output head top-k where applicable).

## Arm A -- gemma-2-9b independent refit (model-vs-fit + HF release)

**Question.** The atlas found gemma-2-9b near-layer-invariant (off-diagonal CKA 0.99 on
the shared probe) where its sibling gemma-2-27b is structured. Is that a property of the
model or of Neuronpedia's fit?

**Procedure.** Fit our own lens: `fit_lens.py --model google/gemma-2-9b --corpus
wikitext --n-prompts 100 --seed 0`. Compute its shared-vocab CKA map; compare to
Neuronpedia's gemma-2-9b map (atlas `shared_maps/gemma-2-9b.npz`).

**Pre-stated prediction.** The near-layer-invariance reproduces (our independent fit's
off-diagonal shared CKA median > 0.95, and its fitted band separation < 0.02). Rationale:
the teacher (gemma-2-27b) fit on the same pipeline is structured, which argues the
flatness is model-specific, not a generic fit failure. **Decision rule, frozen:** if it
reproduces, the flatness is reported as a model property; if our fit instead shows
structure (shared band separation > 0.05), the atlas observation is retracted to "a
limitation of the public fit," and the post is corrected.

**HF release (TJ-approved 2026-07-22, gated).** Release the fitted lens to
`praxagent-org/jacobian-lens-gemma-2-9b` regardless of which way the question resolves
(a lens is an artifact, not a result to spin), conditioned on: (1) consumer-path
verification (reload from HF, hash-verify, reproduce the CKA statistic to < 1e-4); (2)
Gemma Terms-of-Use check for derived artifacts (Neuronpedia precedent; include Gemma
terms + attribution in the card); (3) TJ sign-off on the card before it goes public. Card
carries the CKA map, random-J null, and the honest model-vs-fit framing.

## Arm B -- corpus effect on Qwen (code vs wikitext, length-matched)

**Question (the Laguna question, done right).** Does fitting a lens on code instead of
prose reorganize the workspace? The community observation was confounded by code prompts
being longer than wikitext prompts.

**Procedure.** Fit two lenses on the SAME model (Qwen/Qwen3-4B): one on wikitext, one on
a code corpus, with prompt **token-length distribution matched by construction** (both
truncated to the same per-prompt token budget, same n_prompts, same seed). The `corpus="code"` loader (implemented + CPU-smoked: `codeparrot/codeparrot-clean-valid`, ungated) with token-level length-matching via `--match-length` (keeps only passages that fill max_seq_len tokens, so BOTH corpora contribute identical 128-token windows).
Compare CKA(code-lens, wikitext-lens) at matched depth.

**Pre-stated prediction.** If code reorganizes the workspace, the mid-band cross-lens CKA
drops below 0.7 while the late (motor) block stays above 0.9 (the community read: chat/code
formatting hits input-side and content-processing, output-prep is preserved). **Control
for the confound we are removing:** because lengths are matched, a drop cannot be
attributed to prompt length. Report the length-matched result as primary; also report the
unmatched version to show the confound's size.

## Arm C -- quantization A/B (bf16 vs FP8)

**Question.** Does FP8 quantization change the lens geometry? (Laguna's NVFP4 result:
almost not at all.)

**Procedure.** Fit a lens on `Qwen/Qwen3-4B` (bf16) and on `Qwen/Qwen3-4B-FP8`, identical
corpus/prompts/seed; compare CKA. **Precision contract, per GPU_COMPUTE skill (a dtype
label is not a precision spec):** record separately the storage dtype, the FP8 variant
(confirm E4M3 vs E5M2 from the checkpoint config at freeze), the block-scaling rule and
scale dtype/granularity, the multiply/accumulate mode, and the writeback dtype, plus the
determinism flags and full version stack, in the receipt.

**Pre-stated prediction.** Quantization barely moves the geometry: CKA(bf16-lens,
fp8-lens) at matched depth > 0.9 (dissimilarity < 0.1), an order of magnitude smaller than
any corpus or version effect. If it is larger, that itself is the finding and is reported.

## Arm D -- optimizer controls on the Marin checkpoints (availability-gated)

**Question.** Do outlier-layer counts and band structure differ by optimizer at fixed
architecture? (Frozen version of the community "muon has fewer outlier layers" read.)

**Procedure, gated on lens availability.** If fitted lenses for the Marin optimizer
checkpoints (Wen et al., arXiv:2509.02046) already exist publicly (e.g. eliebak's
open-jlens-data), this arm is **free CPU**: run the atlas outlier-census + fitted
segmentation + null on them. If no public lenses exist, fitting ~10 checkpoints is out of
the Tier-1 envelope and this arm **defers** to a later decision. Availability to be
confirmed at freeze (HF lookup was unavailable at drafting).

**Pre-stated prediction (if run).** Optimizer identity produces a measurable difference in
outlier-layer fraction at fixed architecture (a one-way test across optimizers, p < .05),
i.e. the community observation is real; direction (which optimizer is smoothest) reported,
not predicted.

## Cost and go/no-go

Distinct fits: gemma-2-9b (A), qwen3-4b bf16 wikitext (shared B/C baseline), qwen3-4b code
(B), qwen3-4b-fp8 (C) = **4 small-model fits**. On one cheap GPU pod (e.g. A6000 ~$0.5/hr
or a single H100), sequentially with model downloads: est. **~$10-20 GPU**, within the
approved envelope; no API/Serper spend. Arm D is $0 if lenses are public, else deferred.

**Gate:** this pre-registration is committed; the `corpus="code"` loader + length-matching
is implemented and CPU-smoked on gpt2; then TJ gives explicit go for the specific pod +
cost, and only then does a pod launch. Nothing here spends until that go.
