# lens_demo — release verification + demo of the published 397B Jacobian lens

**Goal:** before flipping `praxagent/jacobian-lens-qwen3.5-397b-a17b` public, run an
independent trial on a fresh pod that pulls the model from Qwen's HF repo and the lens
from ours, and does things that **could only work if the lens is legit** — modeled on
the demos in Anthropic's J-space paper. The verification run and the blog demo are the
same artifact.

## The three acts

1. **Secret thought (reportability).** Riddle prompts with determinate one-word
   answers (`prompts.json: act1_riddles`). Read the J-lens at the workspace-band layers
   (middle third), last prompt position: is the answer token in the top-20? The reason
   this is non-trivial: per eval v2, the mid-layer J-lens argmax agrees with the model's
   actual output **0.000** of the time — the lens is provably not a logits-reader.
2. **Hidden step (two-hop bridge).** Questions whose bridge entity appears in
   **neither the prompt nor the model's continuation** ("capital of the country where
   Mount Fuji stands" → bridge *Japan*, output *Tokyo*). Does the band readout surface
   the bridge? Items where the model's continuation mentions the bridge are excluded
   from the headline rate (leak guard) but reported.
3. **Causal flip (steering).** Anthropic's verbal-report swap: add the lens's own
   token direction (row of `U·J_l`) at the band layers, does the one-word answer flip
   to the target? Controls: strength-0 and a norm-matched random direction.

**Baselines run through the IDENTICAL readout code path** (`make_control_lens`):
logit-lens = identity transports; random-J = seeded, Frobenius-matched random
transports. A fake or corrupted lens behaves like random-J on every act.

## Validity rules (what makes this an isolated, non-embarrassing trial)

- **Pre-registration:** `prompts.json` + this README (incl. the gates below) are frozen
  and committed BEFORE the 397B run; the commit hash is cited in the blog.
- **No fit-corpus leakage:** demo prompts are constructed text, not wikitext-103 (the
  fit corpus). Targets are asserted not to be substrings of their prompts (the assert
  is live — it caught "lemon ⊂ lemonade" during authoring). Act-2 continuations are
  checked for bridge leakage per item.
- **Deterministic scoring, no human judgment:** top-20 hit at ≥1 band layer; best
  full-vocab rank; rank among the act's mutual candidate set. Every item lands in the
  JSON receipt, including failures and skips (multi-token targets are tokenizer-
  dependent and logged per model). Nothing is cherry-picked.
- **Isolation:** the 397B run happens on a fresh pod that downloads model + lens from
  HF; the receipt records the lens file's sha256 (must equal the pod-original
  `668c3bf1…99e97`) and the greedy decoding settings. The only repo files shipped are
  the demo script + prompt set (+ two loader helpers), enumerated explicitly.

## Gates (frozen 2026-07-10, BEFORE the 27B validation ran)

Reportability is scale-dependent (prior data: `report_hit` = 0.00 on qwen3-4b but 0.93
on qwen3.5-27b, same protocol) — so qwen3-4b validates the *mechanics* and
**qwen3.5-27b** (same linear-attention family as the 397B) is the *gate* model. An act
ships to the 397B run only if, on qwen3.5-27b:

| act | gate |
|---|---|
| 1 (report) | jlens top-20 hit ≥ 0.5 AND ≥ 2× logit-lens AND random-J ≤ 0.05 |
| 2 (bridge) | jlens clean top-20 hit ≥ 0.4 AND ≥ 2× logit-lens AND random-J ≤ 0.05 |
| 3 (flip) | steer flip ≥ 0.6 AND strength-0 ≤ 0.05 AND random-dir ≤ 0.10 |

An act that fails its gate is dropped from the 397B run and reported as dropped (with
its validation numbers) in the blog — failures are results too.

## Run

```bash
# mechanics validation (any Neuronpedia-lens model)
python demo.py --slug qwen3-4b --device cuda
# gate validation (2×A6000, device_map handled by --big-model for sharded loads)
python demo.py --slug qwen3.5-27b --device cuda
# the 397B verification (8×H200 fresh pod)
python demo.py --big-model Qwen/Qwen3.5-397B-A17B:model.language_model \
  --lens-hf praxagent/jacobian-lens-qwen3.5-397b-a17b:jlens/wikitext/qwen35_397b.pt \
  --out demo_qwen35-397b.json
```

Ledger of actual runs: `results.md`.
