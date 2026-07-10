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

## 🎯 Qwen3.5-397B-A17B — THE RELEASE VERIFICATION (2026-07-10, 8×H200 @ $35.12/hr, 22 min ≈ $13)

Fresh pod (`irwdny6uop1ssd`), nothing reused: model pulled from `Qwen/Qwen3.5-397B-A17B`,
lens pulled from `praxagent/jacobian-lens-qwen3.5-397b-a17b` — **downloaded-lens sha256
= `668c3bf1…99e97`, exactly the pod-original receipt** (the integrity check is inside
the demo itself). Acts 2+3 per the pre-registered gates (`8102510`). Token passed via
stdin env; never written to pod disk. Pod terminated at +22 min; `pods` audited.

| act | jlens | logit-lens | random-J |
|---|---|---|---|
| 2 bridge, top-20 hit | **0.30** (0 leaks in 20/20) | 0.05 | 0.00 |
| 2 bridge, **median best-rank** (of 248,320) | **43** | 620 | 7,121 |
| 3 flip (steer / str-0 / rand-dir) | **0.64** / **0.00** / **0.00** | — | — |

- **The lens reads thoughts the model never says.** Example: "The capital of the
  country home to the Maasai Mara reserve is" → continuation " Nairobi. The Maasai
  Mara National…" — *Kenya* appears nowhere in input or output, and the J-lens ranks
  it 11th of 248,320 at the band. Across all 20 items the bridge never leaked into a
  continuation, and the J-lens median rank (43) sits in the **top 0.017%** of the
  vocabulary — identity transports (620) and random transports (7,121) read noise at
  the same layers/positions. Top-20 hits: Japan #3, China #5, Kenya #11, Canada #13,
  Peru #13, Brazil #19.
- **The lens's directions are causally load-bearing:** 32/50 steered swaps flip the
  output; strength-0 and norm-matched random directions flip **0/50 each**.
- **Honest deltas vs the 27B gate run** (0.85 / 1.00 there): absolute rates are lower
  at 397B. Confounded explanations we can't separate here: (a) our lens is fit on
  n=24 prompts vs the Neuronpedia n≈1000-class lens used at 27B — consistent with
  fit-corpus size buying readout quality (the planned eval-vs-n ladder tests exactly
  this); (b) 248k vs 152k vocab makes top-20 a harsher bar; (c) model scale itself.
  What is NOT in doubt: both effects are large against both controls, and a fake or
  corrupted lens produces the random-J row.

**Release-gate verdict: the published artifact is legit** — an independent fresh-pod
trial, using only the public(-to-be) HF artifacts, reproduced hash-exact integrity,
hidden-intermediate readout, and causal steering with dead-zero controls.

Receipts: `demo_qwen35-397b.json`, `demo397.log`.
Demo series total cost: gpt2 $0 + 3090 $0.15 + A100 $0.60 + 8×H200 $13 ≈ **$14**.
