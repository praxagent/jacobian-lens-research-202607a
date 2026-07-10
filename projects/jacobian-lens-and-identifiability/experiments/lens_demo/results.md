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

## qwen3.5-27b GATE validation — PENDING (A100 80GB, launched 2026-07-10)

Gates were frozen in README.md and committed BEFORE this run completed. Results land
here when the pod reports.
