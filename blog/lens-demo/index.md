---
title: "We Read a 0.4-Trillion-Parameter Model's Mind — and You Can Check Our Work"
date: 2026-07-10
draft: true
authors: ["praxagent"]
tags: ["interpretability", "j-space", "jacobian-lens", "open-science", "demo"]
summary: "We fit a Jacobian lens on Qwen3.5-397B — the largest open model anyone has published one for — and then verified it the way you'd want us to: a fresh cloud machine, downloading only the public artifacts, running a pre-registered trial it could only pass if the lens is real. It reads entities the model thinks about but never says, and its directions causally steer the model's answers. Every control reads noise."
---

Ask a language model:

> *"The capital of the country home to the Maasai Mara reserve is"*

Qwen3.5-397B answers: **" Nairobi. The Maasai Mara National…"**

The word *Kenya* appears nowhere — not in the question, not in the answer. But the
model had to think it. Point our [Jacobian lens](https://huggingface.co/praxagent/jacobian-lens-qwen3.5-397b-a17b)
at the model's mid-network "workspace" layers at that moment, and out of the 248,320
tokens in the vocabulary, **Kenya ranks 11th**. For "capital of the country where Mount
Fuji stands" (→ " Tokyo"), *Japan* ranks **3rd**. The Great Wall question puts *China*
at **5th**. The model never says these words. The lens reads them anyway.

That's the demo. The rest of this post is us trying to convince you we didn't fool
ourselves — because the entire trial is designed so that a fake lens **cannot pass it**.

## What this is

Anthropic's [J-space paper](https://transformer-circuits.pub/2026/workspace/index.html)
introduced the Jacobian lens: for each layer, the average causal map from that layer's
residual stream to what the model is about to say. Composed with the unembedding, it
gives a per-token readout of what's *active in the model's workspace* — which Anthropic
showed is reportable, causally steerable, and load-bearing in Claude.

We [audited that claim across 35 open models](../jspace-audit/) and, as part of the
audit, fit our own lens on **Qwen3.5-397B-A17B** — to our knowledge the largest open
model with a published lens. This post is the release check for that artifact.

## The trial: designed to be impossible to pass with a fake

**Isolation.** A fresh rented 8×H200 machine downloads the model from Qwen's HF repo
and the lens from ours, and runs one script. The script's first act is to hash the
lens it downloaded: `668c3bf1…99e97` — byte-identical to the fit machine's original.

**Pre-registration.** The prompt sets, the deterministic scoring rule, and the
ship/drop gates were committed to git (`8102510`) *before* the gate validation ran,
and the gate validation ran on a different model (qwen3.5-27b) *before* the 397B was
touched. Every item lands in a JSON receipt — including the failures.

**Controls through the identical code path.** Every readout is also run with two
impostor lenses: a **logit-lens** (identity transports — "you could read this without
the lens") and a **random-J** (seeded random transports, scale-matched per layer —
"a fake lens"). Same prompts, same layers, same positions, same code.

**Leakage guards.** Target words are asserted absent from their prompts (the assert
caught a real authoring bug — "lemon" hiding inside "lemonade"). For the hidden-entity
act, the model's actual continuation is checked per item: in **20 of 20 items** the
bridge entity never appeared in the output.

## Reading the hidden step ("act 2" in the receipts)

Twenty two-hop questions ("capital of the country where X is") whose bridge entity
appears in neither input nor output. Can each lens find the bridge at the workspace
band?

| readout (same code, same layers) | top-20 hit | median rank (of 248,320) |
|---|---|---|
| **Jacobian lens (ours)** | **0.30** | **43** |
| logit-lens (identity) | 0.05 | 620 |
| random-J (fake lens) | 0.00 | 7,121 |

The J-lens's *median* landing spot is the top 0.017% of the vocabulary. The named hits:
Japan #3, China #5, Kenya #11, Canada #13, Peru #13, Brazil #19.

## Steering with the lens's own directions ("act 3" in the receipts)

Anthropic's verbal-report swap: add the lens's direction for a different concept at
the band layers, and see if the model's one-word answer flips to it.

| condition | flips (of 50) |
|---|---|
| **lens direction, strength 8** | **32 (64%)** |
| strength 0 | 0 |
| random direction, same norm | 0 |

The directions aren't just decodable — they're causally load-bearing. And a random
vector of identical magnitude at identical layers does *nothing*.

## What we're NOT claiming (the honest ledger)

- **A third test ("act 1" in the receipts) failed its gate and was dropped — per
  pre-registration, before the 397B ran.** Direct riddles ("the striped African horse
  is the…") scored 0.31 on the gate model — *exactly tied with the logit-lens*. When the concept is the model's
  next token, you don't need a Jacobian lens to see it. The lens's distinctive power
  is **intermediate** content — which is precisely Anthropic's own characterization
  (their A.6: the J-lens is deliberately bad at next-token reading). We report the
  drop because a demo that hides its dead ends isn't a verification.
- **Rates are lower at 397B than at 27B** (bridge hit 0.30 vs 0.85; flips 64% vs 100%
  with a Neuronpedia-fit lens there — same 248k vocabulary, so that's not it). We
  can't yet separate the causes: our 397B lens is fit on 24 prompts (theirs ~1000) —
  though on the motor-convergence fidelity metric our n=24 lens actually *beats* the
  architecture-matched Neuronpedia lens, so fit-size is plausible but unproven; the
  397B's sparse 512-expert routing may spread workspace content differently; and our
  steering strength was calibrated on small models. The discriminating experiment
  (extend the lens to more prompts — the math warm-starts exactly — and re-run) is
  queued. What's not in doubt: every effect is large against both controls.
- **This is one model, one lineage, twenty items per act.** It's a verification demo,
  not a survey — the [35-model survey is here](../jspace-audit/).

## Reproduce it

You don't need 8×H200s to check us. The same script on **qwen3.5-27b** (one A100,
about a dollar) produces the stronger version of everything above: bridge hit 0.85
(Sweden at rank **1** of the 248k vocabulary), flips 100%, all controls at zero.

```bash
git clone https://github.com/praxagent/research-and-replications
cd projects/jacobian-lens-and-identifiability/experiments/lens_demo
python demo.py --slug qwen3.5-27b            # ~$1 tier (Neuronpedia lens)
python demo.py \                              # the 397B verification itself
  --big-model Qwen/Qwen3.5-397B-A17B:model.language_model \
  --lens-hf praxagent/jacobian-lens-qwen3.5-397b-a17b:jlens/wikitext/qwen35_397b.pt
```

Receipts: pre-registration `8102510` → gate `4f44976` → result `d9fc376`; per-item
JSONs and pod logs in
[`experiments/lens_demo/`](https://github.com/praxagent/research-and-replications/tree/main/projects/jacobian-lens-and-identifiability/experiments/lens_demo).
The whole verification series — CPU smoke, two validation pods, and the 397B run —
cost about **$14**.

*Built on Anthropic's Apache-2.0 [jlens](https://github.com/anthropics/jacobian-lens)
and Neuronpedia's open lens collection. Corrections welcome — that's what the receipts
are for.*
