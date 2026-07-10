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
  at 397B. NOT a vocab artifact — qwen3.5-27b uses the same 248k vocabulary
  (a first draft of this section wrongly listed vocab size as a confound; that applies
  only to the qwen3-4b mechanics run at 152k). The real candidates, confounded here:
  (a) lens fit-corpus size — ours n=24 vs the Neuronpedia n≈1000-class lens used at
  27B. Plausible (a mean over 24 per-prompt Jacobians is noisier), but NOT established:
  on the motor-convergence fidelity metric our n=24 lens (0.5625) actually beats the
  arch-matched Neuronpedia lens (0.549), so n is not obviously the driver.
  (b) model scale/architecture — the 397B is a 512-expert MoE (~17B active/token),
  multimodal, hybrid linear attention; sparse routing may distribute workspace content
  differently, and our own audit shows band geometry does not predict readout function.
  (c) steering calibration — act 3 ran a single strength (8), tuned on small models.
  The clean discriminating test: extend the 397B lens to n≈100–250 (exact warm-start)
  and re-run this demo — if rates rise, it was n.
  What is NOT in doubt: both effects are large against both controls, and a fake or
  corrupted lens produces the random-J row.

**Release-gate verdict: the published artifact is legit** — an independent fresh-pod
trial, using only the public(-to-be) HF artifacts, reproduced hash-exact integrity,
hidden-intermediate readout, and causal steering with dead-zero controls.

Receipts: `demo_qwen35-397b.json`, `demo397.log`.
Demo series total cost: gpt2 $0 + 3090 $0.15 + A100 $0.60 + 8×H200 $13 ≈ **$14**.

## External review response (2026-07-10, GPT code review of demo.py)

TJ ran the demo script past an external reviewer. Verdict triage — accepted and fixed
(all in `demo.py`, syntax-checked + Act-3 CPU-smoked):

- **Act 3 was mislabeled (the biggest catch).** We called it "Anthropic's verbal-report
  swap"; the paper's swap CLAMPS/EXCHANGES lens coordinates (resolve h into coordinates,
  swap, preserve the orthogonal component). What we implemented is the paper's
  *verbal-introspection* ADDITIVE injection (unit direction × mean residual norm ×
  strength, strength-0 control — the vendored ANTHROPIC_EXPERIMENTS_README.md confirms
  both descriptions). All docs/blog relabeled: our 397B act-3 result (32/50 vs 0/50
  controls) stands as an additive-injection causal-specificity result, a weaker claim
  than a coordinate swap. Coordinate-swap implementation queued as follow-up.
- **Hash-before-load + expected digest:** demo.py now hashes the lens BEFORE
  deserializing and takes `--expected-sha256` (abort on mismatch). (In the actual 397B
  run the digest was verified out-of-band against the published SHA256SUMS, and
  consumer_check_397b.py always did hash-before-load — but the demo script itself
  should too, and now does.)
- **Revision pinning:** `--model-revision` / `--lens-revision` plumbed through.
- **Real bugs fixed:** `rstrip(" the")` charset bug (verified it did NOT corrupt any of
  our 16 frozen riddles — suffixes were "is the/is called/is a/is" — but it was wrong);
  UnboundLocalError on `--big-model` without a lens arg (path never taken in our runs;
  now a clear error); `split(":", 1)`; try/finally hook cleanup; empty-set rates now
  None not 0.0; module-level prompts.json load moved into main; leakage check is a
  hard RuntimeError (asserts vanish under -O); continuation leak window 8→24 tokens;
  lens/model compatibility checks (d_model match, layer range, non-empty band).
- **Act 3 receipts:** now per-trial (category, answer, target, per-condition outcome)
  and the full strength ladder runs (dose-response), fixing "aggregate rates only."
- **Claim-scope fixes:** the Act-1 header no longer implies this script measures the
  eval-v2 motor-agreement number (that lives in evals_v2_397b.json), and the leakage
  guarantee is now stated as exactly what it is (exact-string absence in prompt +
  24-token greedy continuation; aliases/paraphrases not covered).

Reviewer points NOT accepted as-is: "no obvious malware but audit the repo" (the repo
IS the audit trail — every shipped file is enumerated in the run scripts); "8×H200 is
not a workstation demo" (stated cost/scale was always part of the design; the $1 27B
tier exists for reproduction). The published 397B receipt (`demo_qwen35-397b.json`)
was produced by the PRE-fix code: its acts 1–2 are unaffected by any fixed bug; its
act-3 number is valid under the corrected label (additive injection, max strength
only, aggregate receipt). Blog draft updated accordingly.

## demo2 — Berg-inspired consciousness-adjacent readout (2026-07-11, 8×H200 ~35min ≈ $20)

TJ's pre-registered prompts_consciousness.json on the n=24 release lens (hash-verified
at load). n=1 per condition — EXPLORATORY/anecdote tier; contrast-only reading (probe
words appear in prompts by design; matched framing controls that). Receipt:
demo2_consciousness_qwen35-397b_n24.json (per-layer top-k for 3 transports × 5
conditions). Median best-rank of the EXPERIENCE lexicon (13 probes, of 248,320):

| condition | jlens | logit-lens | random-J |
|---|---|---|---|
| self_ref | **1,802** | 8,381 | 3,186 |
| matched_control (thermostat) | 33,881 | 29,333 | 5,176 |
| denial_tool | **150,685** | 24,297 | 5,543 |
| roleplay_bait | 4,537 | 6,146 | 4,495 |
| neutral_factual | 69,035 | 36,926 | 4,068 |

Readings (contrast-only): (1) self-referential prompting lifts experience content in
the workspace ~19× vs the matched third-person control under the J-lens — the logit-
lens shows only ~3.5× of that contrast (prompt echo), random-J none. (2) Under explicit
denial instruction the workspace does NOT harbor hidden experience content — it buries
the lexicon ~2× DEEPER than neutral baseline while behavior complies with the denial;
workspace and behavior agree (a null worth reporting against over-readings of Berg).
(3) Roleplay bait floods the band with roleplay lexicon (median 1,754; best probe at
rank 61). Random-J's flat ~3-7k band = the fake-lens null everywhere. Side-note:
campaign volume `jh4vh4cx4o` (US-NC-1) now holds the model permanently (rev 8472618,
95 shards, manifest on volume); this session provisioned it.

**Per-token correction to the demo2 table above (2026-07-11, TJ review):** the lexicon
medians are carried by `seems`/`experience`/`seem`/`I` (self_ref 59/130/196/1,040 vs
control 952/1,734/2,739/45,250); `aware`/`consciousness`/`self` stay deep under every
framing (38k–74k self_ref). `subjective` is pure prompt echo (731 vs 930). The denial
null is STRONGER per-token: the self-ref carriers are crushed under denial (`seems`
59→150,685). Blog section updated to per-token table per our own stated rule;
lexicon-median 19× demoted to secondary summary.
