# Results — lens release demo (ledger of actual runs)

## gpt2-small CPU smoke (2026-07-10, free)

Plumbing check only (gpt2 has no band, `mid_sep` 0.015 — no capability expected).
All three transports ran through the identical readout path; receipts + skips logged
(3 riddle targets multi-token in gpt2's tokenizer); the leakage assert caught a real
authoring bug ("lemon" ⊂ "lemonade") before any GPU was rented. Hit rates 0.00 across
the board, as expected. `demo_gpt2_smoke.json`.

## qwen3-4b mechanics validation (2026-07-10, RTX 3090 @ $0.22/hr, ~35 min ≈ $0.15)

Purpose: prove all three acts + controls on a real GPU with a real (Neuronpedia) lens.
NOT the gate model — prior data says readout reportability is a scale capability
(report_hit 0.00 at 4B vs 0.93 at 27B, same protocol).

| act | jlens | logit-lens | random-J |
|---|---|---|---|
| 1 report (top-20 hit) | 0.00 | 0.00 | 0.00 |
| 2 bridge (top-20 hit) | **0.15** | 0.00 | 0.00 |
| 3 flip (steer / str-0 / rand-dir) | **0.88** / 0.00 / 0.00 | — | — |

- **Act 3 is textbook at 4B already:** flip rate 0.88 (matches the historical swap
  0.88 exactly — independent reimplementation agreement), both controls at 0.00.
- **Act 2 signal is visible in the ranks even where top-20 misses:** J-lens best
  full-vocab bridge ranks include 4 (Argentina), 45 (Russia), 78 (Greece), 156
  (Germany), 161 (Netherlands), 188 (China) out of a 152k vocab; logit-lens ranks for
  the same items sit in the hundreds-to-tens-of-thousands. The J-lens is reading
  bridge content the logit-lens cannot see at the same layers/positions.
- **Act 1 at 0.00 at 4B** — consistent with the known scale dependence; the act's
  go/no-go is decided on qwen3.5-27b (gates in README, frozen before that run).

Receipts: `demo_qwen3-4b.json`, `demo_val.log`. Pod terminated after run (verified
`pods` empty).

## qwen3.5-27b GATE validation (2026-07-10, A100 80GB @ $1.39/hr, ~25 min ≈ $0.60)

Gates were frozen in README.md and committed (`8102510`) BEFORE this run completed.

| act | jlens | logit-lens | random-J | gate | verdict |
|---|---|---|---|---|---|
| 1 report | 0.31 | 0.31 | 0.00 | ≥0.5 AND ≥2× logit | **FAIL — DROPPED from 397B** |
| 2 bridge | **0.85** (clean 0.85, 0 leaks/20) | 0.10 | 0.00 | ≥0.4 clean AND ≥2× logit AND rand ≤0.05 | **PASS (8.5× baseline)** |
| 3 flip | **1.00** / str-0 0.00 / rand-dir 0.00 | — | — | ≥0.6 / ≤0.05 / ≤0.10 | **PASS (perfect)** |

- **Act 2 is the money act:** bridge entities at full-vocab rank **1** (Sweden), 2
  (Kenya), 4 (Germany) out of 248k-class vocabs, while the model's continuation never
  mentions them (0 leaks in 20 items — the "neither input nor output" property held
  universally). Logit-lens at the same layers/positions: 0.10.
- **Act 1's failure is itself informative** (and we report it): when the answer is the
  *next token*, logit-lens ties the J-lens (0.31 = 0.31) — the info is already in the
  output pipeline, so the act doesn't discriminate. The J-lens's distinctive power is
  *intermediate* content (Act 2) — exactly Anthropic's A.6 characterization. The
  keep-it-secret variant scored 0.00 on this base model (doesn't hold instructions).
- Act 3 matches the historical swap rate for this model (1.00) exactly.
- Metric honesty: `median_cand_rank` saturates at 1 even for random-J (min over 21
  band layers × 20 candidates — multiple comparisons); it is NOT a discriminating
  metric and is not used in gates or headline claims. Full-vocab top-20 + best-rank
  are the metrics that matter (random-J: 0.00 hits, ranks in the thousands).

Receipts: `demo_qwen3.5-27b.json`, `demo_gate.log`. Pod terminated (verified empty).

**397B run ships Acts 2 + 3 only, per pre-registration.**
