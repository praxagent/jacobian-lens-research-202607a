---
title: "We Read a 0.4-Trillion-Parameter Model's Mind — and You Can Check Our Work"
date: 2026-07-10
tags: ["AI", "LLM", "machine-learning", "interpretability", "jacobian-lens", "j-space", "reproducibility", "open-science"]
author: Timothy Jones
summary: "We fit a Jacobian lens on Qwen3.5-397B — the largest open model anyone has published one for — and verified it the way you'd want us to: a fresh cloud machine, public artifacts only, and a pre-registered trial a fake lens cannot pass. It reads entities the model thinks about but never says, and its directions causally steer the model's answers. Every control reads noise."
og_image: "https://praxagent.ai/assets/og-jacobian-lens-397b-demo.jpg"
lead: |
  Ask a language model the capital of the country home to the Maasai Mara, and Qwen3.5-397B answers "Nairobi" — never saying Kenya. Point a Jacobian lens at the model's mid-network workspace at that moment, and Kenya ranks 11th of 248,320 vocabulary tokens. Japan ranks 3rd on the Mount Fuji question; China ranks 5th on the Great Wall. The model never says these words. The lens reads them anyway. This note is the release check for that lens: a pre-registered, isolated trial designed so that a fake lens cannot pass it, with every control run through the identical code path.
---

{{< panel "info" >}}
**AI-use disclosure & disclaimer.** Generative-AI tools were used during drafting and editorial revision; the author framed the questions, chose the analyses, and reviewed the outputs. This post is shared in the spirit of open-source research: an independent, non-peer-reviewed note published so the community can inspect, reproduce, and correct it. The data, code, and text are provided as-is, without warranty of any kind; errors are possible despite good-faith effort. Verify against the released artifacts before relying on anything here, and use at your own risk. Corrections are welcome.
{{< /panel >}}

{{< panel "info" >}}
**Abstract.** Anthropic's Jacobian lens maps each layer's residual stream to what the model is about to say, yielding a per-token readout of content active in the model's mid-network "workspace." As part of a [35-model audit](../jspace-audit/), we fit our own lens on **Qwen3.5-397B-A17B** — to our knowledge the largest open model with a published lens ([`praxagent/jacobian-lens-qwen3.5-397b-a17b`](https://huggingface.co/praxagent/jacobian-lens-qwen3.5-397b-a17b)). This note is the release verification: a fresh 8×H200 machine downloads the model from Qwen and the lens from us, hashes the artifact (`668c3bf1…99e97`), and runs a pre-registered trial (`8102510`) whose gates were frozen on a different model before the 397B was touched. On twenty two-hop questions whose bridge entity appears in neither input nor output, the Jacobian lens hits the top-20 at rate **0.30** (median rank **43** of 248,320); identity and random-J controls score 0.05 and 0.00. Steering with the lens's own directions flips **32/50** one-word answers; strength-0 and norm-matched random directions flip **0/50** each. A third act (direct riddles) failed its pre-registered gate on the 27B gate model and was dropped before the 397B ran. Absolute rates are lower at 397B than at 27B; every effect remains large against both controls. The whole verification series cost about **$14**.
{{< /panel >}}

### Learning objectives

By the end of this note you should be able to:

1. state what a Jacobian lens computes, and why it is *not* a next-token reader;
2. distinguish a **reportability** claim, a **hidden-intermediate** readout, and a **causal steering** claim;
3. explain why identity transports and random-J are the right controls for a lens demo; and
4. reproduce the cheaper gate-model version of the trial on one A100 (~$1), or audit the 397B receipts from the released JSON.

---

## Why a Fake Lens Cannot Pass This Trial

In mechanistic interpretability, it is common to see sentences like:

{{< panel "quote" >}}
*Hypothetical over-read (not a quotation from a paper):* We pointed a Jacobian lens at the model and it revealed the model's hidden thoughts.
{{< /panel >}}

That kind of sentence packs three different epistemic objects into one noun phrase:

1. **an artifact**: a fitted transport \(J_\ell\) per layer, composed with the unembedding;
2. **a readout claim**: that the top tokens at mid-layers name content the model is using;
3. **a causal story**: that intervening on those directions *is* intervening on that content.

Those are not the same thing. A corrupted download, a mis-fit lens, or a fancy identity map can look impressive on cherry-picked prompts. This trial is designed so that each of those failure modes fails the gates.

### Claim ladder

{{< mermaid >}}
flowchart TD
  A["Published lens artifact<br/>+ model from HF"] --> B["Integrity hash<br/>byte-identical to fit machine"]
  B --> C["Pre-registered prompts<br/>+ deterministic scoring"]
  C --> D["Hidden-bridge readout<br/>vs identity + random-J"]
  D --> E["Causal steering<br/>vs strength-0 + random dir"]
  E --> F["Bounded release claim:<br/>artifact is legit"]
  C -.->|"act 1 failed gate"| X["Dropped before 397B<br/>(reported)"]
{{< /mermaid >}}

<p class="figure-note">Figure: how strong a claim you can make depends on how far you climb. This note focuses on the release check: integrity, pre-registration, discriminating readout, and causal steering with dead-zero controls.</p>

This note does two jobs:

1. explain the Jacobian lens at the level needed to read the trial carefully; and
2. walk through the pre-registered verification of the 397B artifact, including the act that failed its gate and was dropped.

### Where the artifact comes from

Anthropic's [J-space paper](https://transformer-circuits.pub/2026/workspace/index.html) introduced the Jacobian lens and showed that its mid-layer contents in Claude are reportable, causally steerable, and load-bearing. Anthropic released the [code](https://github.com/anthropics/jacobian-lens) (Apache-2.0); Neuronpedia published [pre-fitted lenses for 38 open-weight models](https://huggingface.co/neuronpedia/jacobian-lens). We [audited that claim across 35 open models](../jspace-audit/) and, as part of the audit, fit our own lens on Qwen3.5-397B-A17B.

| Provenance field | Value here |
|---|---|
| Lens artifact | [`praxagent/jacobian-lens-qwen3.5-397b-a17b`](https://huggingface.co/praxagent/jacobian-lens-qwen3.5-397b-a17b) |
| Base model | [`Qwen/Qwen3.5-397B-A17B`](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) |
| Fit corpus | WikiText (n=24 prompts for this release lens) |
| Lens sha256 (downloaded) | `668c3bf1…99e97` (byte-identical to fit-machine original) |
| Pre-registration commit | `8102510` (prompts, scoring, gates — before 397B) |
| Gate model | qwen3.5-27b (same family; Neuronpedia lens) |
| Gate commit | `4f44976` |
| Result commit | `d9fc376` |
| Isolation | Fresh 8×H200 pod; model + lens downloaded from HF; no fit-machine reuse |

You can reproduce the **stronger** version of every headline number on **qwen3.5-27b** (one A100, about a dollar). The 397B path needs a multi-GPU machine; that optional upstream path is documented at the end.

---

## A Quick Glossary (Read This First)

Before the trial design, here is the vocabulary this post uses.

| Term | Meaning |
|---|---|
| **LM / LLM** | Language model / large language model: a neural net trained to predict text |
| **Token** | A chunk of text the model actually reads (often a word piece). The model works on a sequence of tokens, not raw characters |
| **Residual stream** | The model's running "scratchpad" of hidden states: each layer reads it, adds an update, and passes it on |
| **\(h_\ell\)** | Hidden state (residual-stream vector) at layer \(\ell\) |
| **Unembedding** | The final linear map from residual stream to vocabulary logits |
| **Jacobian lens** | Per layer, the average causal map \(J_\ell = \mathbb{E}[\partial h_{\mathrm{final}} / \partial h_\ell]\); composed with the unembedding, a per-token readout of workspace content |
| **J-space / workspace band** | The mid-layer block where these directions are strongly active and mutually similar; Anthropic's structural signature of a "global workspace" |
| **Logit lens** | Identity transports: read \(h_\ell\) as if it were already in final-layer coordinates. "You could read this without the fitted lens" |
| **Random-J** | Seeded random transports, Frobenius-scale-matched per layer. "A fake lens" |
| **Bridge entity** | The intermediate concept in a two-hop question (e.g. *Japan* in "capital of the country where Mount Fuji stands") |
| **Reportability** | Can the lens surface a concept the model is about to say? |
| **Steering** | Add/subtract a lens direction during generation. A causal intervention |
| **Pre-registration** | Freeze prompts, scoring rule, and ship/drop gates in git *before* the decisive run |
| **Gate model** | A cheaper model used to decide which acts ship to the expensive run; here qwen3.5-27b |

{{< mermaid >}}
flowchart LR
  A["Prompt text"] --> B["Transformer LM"]
  B --> C["Hidden state h_ℓ<br/>at band layers"]
  C --> D["Jacobian lens<br/>U · J_ℓ"]
  D --> E["Per-token readout<br/>ranks over vocab"]
  D --> F["Steering?<br/>separate experiment"]
  E --> G["Controls:<br/>identity + random-J"]
{{< /mermaid >}}

<p class="figure-note">Figure: the lens turns mid-layer hidden states into vocabulary-ranked readouts. Naming what those ranks mean, and steering with the directions, are separate claims — and both need controls.</p>

---

## What the Jacobian Lens Actually Is

### The one-paragraph version

For each layer \(\ell\), the Jacobian lens is the **average causal Jacobian**

\[
J_\ell = \mathbb{E}\!\left[\frac{\partial h_{\mathrm{final}}}{\partial h_\ell}\right]
\]

— over a text corpus, how much does nudging the residual stream at layer \(\ell\) change what the model is about to say? Composing that with the unembedding gives, per vocabulary token, the internal direction that most raises the model's disposition to *say* it. The **J-space** is the sparse set of these directions that are strongly active; Anthropic shows its contents are reportable, steerable, and causally load-bearing in Claude.

{{< panel "definition" >}}
**Working definition.** A Jacobian lens is a fitted linear transport from layer \(\ell\)'s residual stream to final-layer coordinates, averaged over a corpus. A *readout* is the vocabulary ranking you get by applying that transport at a chosen position. A *causal claim* requires a separate intervention: add the lens's direction for a concept and measure whether the model's output flips. An identity map (logit lens) and a scale-matched random transport are the minimal controls that ask whether you needed the fitted lens at all.
{{< /panel >}}

### Two properties that shape the trial

1. **The lens is estimated.** It is an average over a fit corpus, so a release check must use prompts *outside* that corpus and must hash the shipped file.
2. **The lens is deliberately bad at next-token reading.** Anthropic's own appendix (A.6) notes that the mid-layer J-lens is not a logits-reader: when the concept *is* the next token, you often do not need a Jacobian lens to see it. The distinctive power is **intermediate** content — concepts the model uses but does not say.

That second point is why this trial's headline act is the two-hop bridge, not the riddle.

---

## The Trial: Designed to Be Impossible to Pass with a Fake

Before design details, here is the experiment stated plainly:

{{< panel "info" >}}
**The experiment in one paragraph.** On a fresh rented machine, download Qwen3.5-397B from Qwen's HuggingFace repo and our published lens from ours. Hash the lens; it must match the fit machine's original. Run three acts through identical code for three transports (fitted J-lens, identity/logit-lens, random-J): (1) riddle reportability, (2) hidden-bridge readout on two-hop questions whose bridge never appears in input or output, (3) additive steering along the lens's own token directions (the paper's verbal-introspection injection recipe applied to the verbal-report task — not its coordinate-swap variant). Ship an act to 397B only if it passed pre-registered gates on qwen3.5-27b. Score deterministically; write every item — including failures — to a JSON receipt.
{{< /panel >}}

### Validity rules

**Isolation.** A fresh 8×H200 machine downloads the model and the lens and runs one script. The script's first act is to hash the lens it downloaded: `668c3bf1…99e97` — byte-identical to the fit machine's original.

**Pre-registration.** The prompt sets, the deterministic scoring rule, and the ship/drop gates were committed to git (`8102510`) *before* the gate validation ran, and the gate validation ran on a different model (qwen3.5-27b) *before* the 397B was touched.

**Controls through the identical code path.** Every readout is also run with two impostor lenses: a **logit-lens** (identity transports — "you could read this without the lens") and a **random-J** (seeded random transports, scale-matched per layer — "a fake lens"). Same prompts, same layers, same positions, same code.

**Leakage guards.** Target words are asserted absent from their prompts (the assert caught a real authoring bug — "lemon" hiding inside "lemonade"). For the hidden-entity act, the model's actual continuation is checked per item: in **20 of 20** items the bridge entity never appeared in the output.

### The three acts

| Act | Question | Gate on qwen3.5-27b | 397B verdict |
|---|---|---|---|
| 1 — Secret thought | Does the band readout surface a riddle's one-word answer? | jlens ≥ 0.5 top-20 **and** ≥ 2× logit-lens | **Dropped** (tied logit-lens at 0.31) |
| 2 — Hidden step | Does the band surface a bridge entity absent from input *and* output? | clean hit ≥ 0.4 **and** ≥ 2× logit **and** random-J ≤ 0.05 | **Shipped** |
| 3 — Causal flip | Does adding the lens's direction flip a one-word answer? | flip ≥ 0.6; str-0 ≤ 0.05; random-dir ≤ 0.10 | **Shipped** |

{{< mermaid >}}
flowchart TB
  P["prompts.json<br/>frozen in 8102510"] --> G["Gate on qwen3.5-27b"]
  G -->|"act 1 FAIL"| D["Drop + report"]
  G -->|"acts 2+3 PASS"| R["Fresh 8×H200<br/>download model + lens"]
  R --> H["Hash check<br/>668c3bf1…99e97"]
  H --> A2["Act 2: bridge readout<br/>+ identity + random-J"]
  H --> A3["Act 3: steer flip<br/>+ str-0 + random dir"]
  A2 --> J["JSON receipt<br/>every item"]
  A3 --> J
{{< /mermaid >}}

<p class="figure-note">Figure: the trial pipeline. Gates decide what ships; the expensive pod only runs acts that already discriminated on the gate model; controls share the identical readout path.</p>

---

## What We Found

### Reading the hidden step (act 2)

Twenty two-hop questions ("capital of the country where X is") whose bridge entity appears in neither input nor output. Can each lens find the bridge at the workspace band?

| readout (same code, same layers) | top-20 hit | median rank (of 248,320) |
|---|---|---|
| **Jacobian lens (ours)** | **0.30** | **43** |
| logit-lens (identity) | 0.05 | 620 |
| random-J (fake lens) | 0.00 | 7,121 |

The J-lens's *median* landing spot is the top 0.017% of the vocabulary. Named hits: Japan #3, China #5, Kenya #11, Canada #13, Peru #13, Brazil #19. Leak guard: **0/20** continuations mentioned the bridge.

{{< panel "quote" >}}
*"The capital of the country home to the Maasai Mara reserve is"* → model continues **" Nairobi. The Maasai Mara National…"** — *Kenya* nowhere in input or output; J-lens ranks it **11th** of 248,320.
{{< /panel >}}

### Steering with the lens's own directions (act 3)

Additive steering with the lens's own directions: add strength × mean-residual-norm × (unit target direction − unit answer direction) at every band layer, and see if the model's one-word answer flips to the target. **Method honesty:** this is the paper's *verbal-introspection* injection recipe applied to the verbal-report task. It is NOT the paper's verbal-report "swap", which resolves the current activation into lens coordinates and exchanges them while preserving the orthogonal component — a stronger claim we have not implemented. A large additive push demonstrates that the lens's directions are causally *effective and specific* (norm-matched random directions do nothing), not that a naturally-active internal coordinate was exchanged. An external code review caught this mislabel; the coordinate-swap variant is queued as follow-up.

| condition | flips (of 50) |
|---|---|
| **lens direction, strength 8** | **32 (64%)** |
| strength 0 | 0 |
| random direction, same norm | 0 |

The directions aren't just decodable — they're causally load-bearing. A random vector of identical magnitude at identical layers does *nothing*.

### What this finding illustrates

Both effects are large against both controls on a fresh machine that only saw public artifacts. That is the release claim: the published file is a real lens, not a corrupted or vacuous transport. It is **not** a claim that rates at 397B match rates at 27B, or that twenty items survey the phenomenon.

---

## Keeping the Claim Bounded

{{< panel "warning" >}}
**What this note does *not* claim.** It does not claim the Jacobian lens is a general mind-reader, that act-1 reportability works at this scale, that n=24 fit prompts are optimal, or that geometry of the workspace band predicts readout function across families. Those questions belong to the [35-model audit](../jspace-audit/) and to follow-up fits. The claim here is narrower: on this pre-registered trial, the published 397B artifact passes the checks a fake lens fails.
{{< /panel >}}

### The honest ledger

- **Act 1 failed its gate and was dropped — per pre-registration, before the 397B ran.** Direct riddles ("the striped African horse is the…") scored 0.31 on the gate model — *exactly tied with the logit-lens*. When the concept is the model's next token, you don't need a Jacobian lens to see it. The lens's distinctive power is **intermediate** content — which is precisely Anthropic's own characterization. We report the drop because a demo that hides its dead ends isn't a verification.
- **Rates are lower at 397B than at 27B** (bridge hit 0.30 vs 0.85; flips 64% vs 100% with a Neuronpedia-fit lens there — same 248k vocabulary, so that's not it). Candidates we cannot yet separate: our 397B lens is fit on 24 prompts (theirs ~1000) — though on the motor-convergence fidelity metric our n=24 lens actually *beats* the architecture-matched Neuronpedia lens, so fit-size is plausible but unproven; the 397B's sparse 512-expert routing may spread workspace content differently; and our steering strength was calibrated on small models. The discriminating experiment (extend the lens to more prompts — the math warm-starts exactly — and re-run) is queued. What's not in doubt: every effect is large against both controls.
- **This is one model, one lineage, twenty items per act.** It's a verification demo, not a survey — the [35-model survey is here](../jspace-audit/).

---

## Threats to Validity

1. **Fit-corpus size.** The release lens averages 24 per-prompt Jacobians. A noisier estimate could depress absolute rates without destroying control discrimination.
2. **Steering calibration.** Act 3 used a single strength (8) tuned on smaller models; under- or over-steering could move the flip rate.
3. **MoE / hybrid architecture.** Qwen3.5-397B-A17B is a 512-expert MoE with hybrid linear attention; workspace content may route differently than in dense models where Neuronpedia lenses were fit.
4. **Small item sets.** Twenty bridges and fifty steering trials are enough to beat dead controls; they are not enough for precise rate estimation or cross-domain generalization.
5. **Tokenizer / multi-token targets.** Some targets are multi-token under some tokenizers and are logged as skips; headline rates are over scorable items only.
6. **Mutable `main`.** Repository links point at a living branch. For archival citation, pin the result commit (`d9fc376`) and the lens file hash.

---

## What Stronger Evidence Would Look Like

1. **Extend the 397B lens** to n≈100–250 prompts (exact warm-start) and re-run this demo; if rates rise toward the 27B gate numbers, fit-size was the driver.
2. **Sweep steering strength** on 397B rather than transferring a small-model default.
3. **Hold out paraphrase families** of the bridge prompts to test template sensitivity.
4. **Cross-check** against an independently fit lens on the same model (when compute allows) for estimation stability at this scale.
5. **Pin** model revision, lens SHA-256, CUDA stack, and source commit in a tagged release for archival-grade citation.

Until those follow-ups land, the careful headline remains: a pre-registered, isolated release check that a fake lens fails and this artifact passes.

---

## Reproduce It

**Why this section exists.** Most readers only need the cheaper gate-model path: the same script on qwen3.5-27b produces the stronger version of everything above (bridge hit 0.85, Sweden at rank **1** of 248k, flips 100%, all controls at zero). The 397B path is for a stricter standard of *upstream* verification.

```bash
git clone https://github.com/praxagent/research-and-replications
cd projects/jacobian-lens-and-identifiability/experiments/lens_demo

# ~$1 tier (Neuronpedia lens) — stronger numbers, same protocol
python demo.py --slug qwen3.5-27b

# the 397B verification itself (multi-GPU)
python demo.py \
  --big-model Qwen/Qwen3.5-397B-A17B:model.language_model \
  --lens-hf praxagent/jacobian-lens-qwen3.5-397b-a17b:jlens/wikitext/qwen35_397b.pt
```

Receipts: pre-registration `8102510` → gate `4f44976` → result `d9fc376`; per-item JSONs and pod logs in [`experiments/lens_demo/`](https://github.com/praxagent/research-and-replications/tree/main/projects/jacobian-lens-and-identifiability/experiments/lens_demo). The whole verification series — CPU smoke, two validation pods, and the 397B run — cost about **$14**.

Record model and lens revisions, CUDA stack, source commit, and output hashes. Public weights make replication possible; provenance still matters.

---

## A Checklist Before You Lean on a Lens Demo

When a paper or demo highlights a Jacobian (or other) lens readout, it helps to check:

1. **Versioning**: model, lens file hash, layer band, exact prompts.
2. **Pre-registration**: were gates frozen before the decisive run?
3. **Controls**: identity transports and scale-matched random transports through the *same* code path.
4. **Leakage**: is the target absent from prompt *and* from the model's continuation?
5. **Separability of claims**: next-token reportability ≠ hidden-intermediate readout ≠ steering.
6. **Dropped acts**: were failures reported, or only the wins?

---

## Conclusion: A Release Check, Not a Mind-Reading Claim

Return to the sentence that opened this note:

{{< panel "quote" >}}
*Hypothetical over-read:* We pointed a Jacobian lens at the model and it revealed the model's hidden thoughts.
{{< /panel >}}

That kind of sentence still packs three objects into one noun phrase: an **artifact** (a fitted transport), a **readout** (vocabulary ranks at band layers), and a **causal story** (steering those directions changes what the model says). The fit gives you the first. Acts 2 and 3, with controls, support the second and third — on this model, these prompts, this published file.

The trial was meant to make that separation tangible, not to argue that the lens is magic. On a fresh pod that downloaded only public artifacts, the J-lens found bridge entities the model never said (median rank 43 of 248k; controls in the hundreds-to-thousands) and steered one-word answers at 64% with dead-zero controls. Act 1 failed its gate and was dropped in public. Absolute rates are lower than on the 27B gate model; every discriminating contrast survives.

Notice that the conclusion here is sharper than "the lens works." The artifact **passed a trial a fake lens cannot pass**, and that success still does not turn twenty items into a survey of workspace function across models. Causal and readout stories have to be earned separately, with controls, and with the failures left in the ledger. That is the reading skill this note set out to teach: **a published lens is not yet a verified one — until you check.**

And read that check the right way around. It is not a complaint about Jacobian lenses; it is an invitation. The gap between "this file loads" and "this file reads intermediate content and steers" is ordinary, checkable science: pre-registration, identity and random controls, leakage guards, and receipts. Everything in this note runs on public weights and released JSON; the [repository](https://github.com/praxagent/research-and-replications) is open, the cheaper path fits in about a dollar, and corrections are welcome — that's what the receipts are for.

---

## References

- Anthropic / Lindsey et al. (2026). *Verbalizable Representations Form a Global Workspace in Language Models*. Transformer Circuits. https://transformer-circuits.pub/2026/workspace/index.html
- Anthropic. *jacobian-lens* (Apache-2.0). https://github.com/anthropics/jacobian-lens
- Neuronpedia. *Jacobian lens collection* (pre-fitted open-weight lenses). https://huggingface.co/neuronpedia/jacobian-lens
- Praxagent. *jacobian-lens-qwen3.5-397b-a17b*. https://huggingface.co/praxagent/jacobian-lens-qwen3.5-397b-a17b
- Praxagent. *A Global Workspace, or a Training Artifact? Auditing Anthropic's J-Space Across the Open-Weight Lineup* (companion audit). https://github.com/praxagent/research-and-replications/tree/main/blog/jspace-audit
- Dehaene, S., & Naccache, L. (invited commentary on the J-space paper). Anthropic-hosted PDF.
- Eleos / Butlin et al. (invited commentary on vocabulary-indexing limitations of the J-space).
- Bakouch, E. *J-space open explorer*. https://eliebak.com/viz/jspace-open
