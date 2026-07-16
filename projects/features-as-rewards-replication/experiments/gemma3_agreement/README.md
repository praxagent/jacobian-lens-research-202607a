# Gemma-3-12B label-free reader-agreement arm

The grader failed its pre-registered κ≥0.5 validation gate (three iterations, κ .28/.28/.18 —
deterministic Wikipedia grounding cannot match live-web gold), so per frozen PROTOCOL §8 this
arm is the **label-free fallback**: no gold AUROC, no label release, agreement-language only
(I05: "score agreement", never "detection validity").

## Design (frozen before this arm runs)

1. **Completions:** generate N=300 Gemma-3-12B-IT completions on the first 300 prompts of the
   pinned LongFact++ `longfact_objects` prompt set (greedy, seed 0, max_new_tokens 768,
   bf16, chat template; generation params in the receipt).
2. **Entity spans:** extracted by DeepSeek via the existing `grade.py --spans-from grader`
   extraction prompt (entity EXTRACTION only — no verification labels are minted or used).
   Budget-guarded (~$1). Span → token alignment as in the labeled arms (pre-token scoring).
3. **Readers scored on the same pre-entity states** (primary layer = Gemma Scope 2 resid_post
   50% depth; head layer for native): logit lens, native output-token surprisal,
   random-transport null, length+frequency heuristic, and the **Gemma Scope 2 SAE**
   (top-k latents by span activation — no label selection possible).
4. **Endpoints (descriptive only):** Spearman/Kendall agreement matrix among reader scores
   across spans; overlap@k of their top-decile "most suspect" spans; the heuristic included
   per I05 so shared length/frequency artifacts are visible. **No hallucination-detection
   claim** — correlated readers may agree for boring reasons; that caveat ships in every
   surface.

## Cost
Single 80GB GPU (A100/H100): generation ~1–2h + caching/readers ~1h ≈ **$15–25**.
DeepSeek extraction ≈ $1 (hard-capped).
