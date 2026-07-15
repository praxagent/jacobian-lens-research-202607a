# Grounding + grader design — Gemma-3-12B own-labeled arm

The public arms (Llama-3.1-8B, Llama-3.3-70B, gemma-2-9b) use the public Obeso LongFact++
gold labels. Only **Gemma-3-12B** (the paper's exact model) has no public labels, so we
generate our own — cheaply, with a **hard, un-overrunnable spend ceiling**, and with
grounding that is **deterministic and reproducible** (which the integrity review prefers
over non-deterministic live web search). Labels are stated as *agreement with our grader's
rubric, not ground truth* (Pro-review B01/I05).

## Pipeline

```
LongFact prompts (frozen sample)
   -> Gemma-3-12B completions (pinned model+gen params+seed)        [GPU, one-time]
   -> entity extraction (grader lists claims/entities per completion)
   -> for each entity: LOCAL retrieval over a pinned Wikipedia snapshot (BM25/FTS, FREE)
   -> grader judges Supported / Not Supported given the retrieved passages   [OpenRouter, prepaid]
   -> token-level gold-ish labels -> four-reader benchmark on Gemma-3-12B + Gemma Scope 2 SAE
```

The **only paid step** is the grader LLM tokens, on a **prepaid OpenRouter balance** with a
hard cap. Retrieval is free local compute. No live web search, no postpaid API.

## 1. Wikipedia snapshot (local, one-time, free)

- Dump: `enwiki-<DATE>-pages-articles.xml.bz2` (~22 GB), current-revisions article text.
  **Pin the exact dump DATE** = the reproducible grounding snapshot.
- Extract cleaned plain text (wikiextractor or mwparserfromhell), ~15–20 GB.
- Build a **SQLite FTS5** (BM25) full-text index over article text (title + body).
- **Delete the raw .bz2** after indexing → steady-state **~25–35 GB** on disk (box has ~88 GB
  free, verified 2026-07-15).
- Retrieval: for each entity/claim, FTS query → top-k passages (free, instant).
- Provenance: record dump date, file SHA-256, extractor version, index build command +
  row count (pinned in the per-arm manifest, per B04/I07).

## 2. Grader (OpenRouter, prepaid, guarded)

- **Access:** OpenRouter (`https://openrouter.ai/api/v1`, OpenAI-compatible). Prepaid
  credits = the account-level hard ceiling; when exhausted, calls fail (no overage).
- **Model:** a cheap strong model — candidates **DeepSeek-V3.2** and **GLM (Zhipu 5.x)**.
  We A/B both on a labeled validation sample (below) and freeze whichever agrees better with
  the public Obeso labels. Model id + revision pinned at freeze.
- **Prompt:** per completion, extract falsifiable entities; per entity, given the retrieved
  Wikipedia passages, output {Supported | Not Supported | Insufficient} + a one-line note.
  "Insufficient" spans are excluded from the labeled set (counted).
- **Determinism:** temperature 0, seed pinned; retrieved passages are deterministic given the
  pinned snapshot, so the whole grader is reproducible up to model-provider nondeterminism
  (disclosed).

## 3. Spend guards (belt AND suspenders)

1. **Prepaid balance** — OpenRouter can only spend what is loaded (no card-billed overage),
   **provided auto-top-up/auto-recharge is OFF**.
2. **Per-key limit** — set a $ limit ON THE KEY in the OpenRouter dashboard (e.g. $25) so this
   key hard-stops regardless of the account balance. (Currently the key has *no* per-key
   limit — see "What we need".)
3. **In-code guard** — the grader runner estimates $ before each batch from live OpenRouter
   token rates, maintains a running spent-total from response usage, and **aborts before a
   frozen `--max-usd` ceiling** and after `--max-calls`. Fails closed. This is the guard that
   was missing on the Pro-review overrun; it is now mandatory for every paid-API runner here.
4. **Dry-run first** — every grader run supports `--dry-run` (build prompts, estimate cost,
   zero calls) before `--execute`.

## 4. Validation (Pro-review B05/I05 — quantify label noise)

Before trusting the Gemma-3 labels, grade a **held-out sample of the PUBLIC arms' completions**
(which have Obeso gold labels) with our exact pipeline, and report agreement (accuracy /
Cohen's κ / per-class) vs the public labels. This quantifies our grader's noise and is
disclosed in the writeup. If agreement is poor, the Gemma-3 arm is reported as
noisy-labeled (or demoted to score-agreement only), never as clean gold.

## 5. Cost (all on the prepaid, hard-capped balance)

- Retrieval + index: **$0** (free local compute), one-time ~1–2 h build.
- Grader LLM: ~2,000 completions × (~1–1.5k in + ~0.5k out) ≈ ~3–4 M tokens. At DeepSeek/GLM
  OpenRouter rates (~$0.15–0.40/M blended) ≈ **~$1–5**. The `--max-usd` guard caps it hard.
- Gemma-3-12B completion generation + activation caching + J-lens fit: **~$20–55 GPU**.
- **Gemma-3 arm total ≈ $25–60**, hard-capped. Whole study ≈ **$75–200** (GPU-dominated).

All pinned in the per-arm manifest and the §10 cost table at freeze.
