---
title: "Open-sourcing (and Verifying) a Jacobian Lens for Qwen3.5-397B"
date: 2026-07-10
tags: ["AI", "LLM", "machine-learning", "interpretability", "jacobian-lens", "j-space", "reproducibility", "open-science"]
author: Timothy Jones
summary: "We fit a Jacobian lens on Qwen3.5-397B — the largest open model anyone has published one for — and verified the readout claim the way you'd want us to: a fresh cloud machine, public artifacts only, and a pre-registered trial a fake lens cannot pass. On a fixed trivia-bridge template it surfaces intermediate country names the model never says; identity and random controls do not. Causal steering is saved for a follow-up note."
og_image: "https://praxagent.ai/assets/og-jacobian-lens-397b-demo.jpg"
lead: |
  Ask Qwen3.5-397B the capital of the country that built the Great Wall, and it answers "Beijing" — never saying China. Point a Jacobian lens at the mid-network residual stream at that moment, and China ranks 5th of 248,320 vocabulary tokens (identity/logit-lens: 8,153rd; random-J: thousands). Japan ranks 3rd on Mount Fuji. Kenya also ranks high (#11) — but there identity wins (#1), so that item is not a clean lens win. Across all twenty capital-of-country bridges the picture is mixed: five strong J-beats-identity hits, two paired losses, and several large familiar countries (France, Spain, Italy) landing hundreds deep. This note is the release check for that lens: a pre-registered, pod-isolated trial designed so that a fake lens cannot pass it, with every control — and every miss — left in the ledger.
---

{{< panel "info" >}}
**AI-use disclosure & disclaimer.** Generative-AI tools were used during drafting and editorial revision; the author framed the questions, chose the analyses, and reviewed the outputs. This post is shared in the spirit of open-source research: an independent, non-peer-reviewed note published so the community can inspect, reproduce, and correct it. The data, code, and text are provided as-is, without warranty of any kind; errors are possible despite good-faith effort. Verify against the released artifacts before relying on anything here, and use at your own risk. Corrections are welcome.
{{< /panel >}}

{{< panel "info" >}}
**Abstract.** Anthropic's Jacobian lens maps each layer's residual stream to what the model is about to say, yielding a per-token readout of content active in the model's mid-network "workspace." As part of a [35-model audit](https://github.com/praxagent/research-and-replications/tree/main/blog/jspace-audit), we fit our own lens on **Qwen3.5-397B-A17B** — to our knowledge the largest open model with a published lens ([`praxagent/jacobian-lens-qwen3.5-397b-a17b`](https://huggingface.co/praxagent/jacobian-lens-qwen3.5-397b-a17b)). This note is the **readout** half of the release verification: a fresh 8×H200 machine downloads the model from Qwen and the lens from us, hashes the artifact (`668c3bf1…99e97`), and runs a pre-registered trial (`8102510`) whose ship/drop gates were frozen before the 397B was touched. On twenty capital-of-country two-hop questions whose bridge entity appears in neither input nor output, the Jacobian lens hits the top-20 at rate **0.30** (median best-rank **43** of 248,320); identity and random-J controls score 0.05 and 0.00. Direct riddles (act 1) failed their gate on the 27B model and were dropped before 397B. Causal steering (act 3) is deferred to a separate note — observation and intervention are different claims. Absolute readout rates are lower at 397B than at 27B; every effect remains large against both controls. The verification series cost about **$14**.
{{< /panel >}}

### Learning objectives

By the end of this note you should be able to:

1. state what a Jacobian lens computes, and why it is *not* a next-token reader;
2. distinguish a **reportability** claim from a **hidden-intermediate** readout (and know that **causal steering** is a separate experiment);
3. explain why identity transports and random-J are the right controls for a lens *readout* demo; and
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
  D --> F["Bounded release claim:<br/>artifact reads intermediates"]
  C -.->|"act 1 failed gate"| X["Dropped before 397B<br/>(reported)"]
  D -.->|"separate note"| E["Causal steering<br/>(follow-up)"]
{{< /mermaid >}}

<p class="figure-note">Figure: how strong a claim you can make depends on how far you climb. This note covers the release check through discriminating readout. Causal steering is a later rung — saved for its own post.</p>

This note does two jobs:

1. explain the Jacobian lens at the level needed to read the trial carefully; and
2. walk through the pre-registered verification of the 397B artifact, including the act that failed its gate and was dropped.

### Where the artifact comes from

Anthropic's [J-space paper](https://transformer-circuits.pub/2026/workspace/index.html) ([Lindsey et al., 2026](#ref-lindsey-2026)) introduced the Jacobian lens and showed that its mid-layer contents in Claude are reportable, causally steerable, and load-bearing. Anthropic released the [code](https://github.com/anthropics/jacobian-lens) (Apache-2.0); Neuronpedia published [pre-fitted lenses for 38 open-weight models](https://huggingface.co/neuronpedia/jacobian-lens). We [audited that claim across 35 open models](https://github.com/praxagent/research-and-replications/tree/main/blog/jspace-audit) and, as part of the audit, fit our own lens on Qwen3.5-397B-A17B.

| Provenance field | Value here |
|---|---|
| Lens artifact | [`praxagent/jacobian-lens-qwen3.5-397b-a17b`](https://huggingface.co/praxagent/jacobian-lens-qwen3.5-397b-a17b) |
| Base model | [`Qwen/Qwen3.5-397B-A17B`](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) |
| Fit corpus | WikiText-103, `max_seq_len` 128, **n=24** prompts (fixed; band statistic already converged by n≈16 on smaller Qwen) |
| Fitting code | **Anthropic's own `jlens.fit`** (running-mean of per-prompt Jacobians — the same package Neuronpedia wraps), via our thin wrapper `fit_at_scale.py` (adds multimodal backbone layout, device_map sharding, eager attention, per-prompt checkpoints). We did **not** replicate Neuronpedia's convergence logger (`run-all-fit-lens.py`'s per-prompt `mean_rel_change` CSV + early-stop) — which is why this lens ships without a matrix-convergence record; any future extension will log it |
| Neuronpedia contrast | Their pipeline *requests* `n_prompts: 1000` but **early-stops** on a matrix-stability criterion with per-lens thresholds. The lens this note's rate comparison actually uses — **qwen3.5-27b** — fitted **672** prompts at `stop_at_delta: 0.002` ([config.yaml](https://huggingface.co/neuronpedia/jacobian-lens/blob/main/qwen3.5-27b/jlens/Salesforce-wikitext/config.yaml)); their Llama-3.3-70B lens fitted **125** at the looser 0.012 ([config.yaml](https://huggingface.co/neuronpedia/jacobian-lens/blob/main/llama3.3-70b-it/jlens/Salesforce-wikitext/config.yaml)). Honest contrast here: **n=24 vs n=672** |
| Lens sha256 (downloaded) | `668c3bf1…99e97` (byte-identical to fit-machine original) |
| Pre-registration commit | `8102510` (prompts, scoring, gates — before 397B) |
| Gate model | qwen3.5-27b (same family; Neuronpedia lens) |
| Gate commit | `4f44976` |
| Result commit | `d9fc376` |
| Isolation | Fresh 8×H200 pod; model + lens downloaded from HF; no fit-machine reuse |
| Independence note | Pod-isolated and hash-checked — **not** an external-lab replication. Same authors wrote the prompts, fit the lens, and ran the demo |

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
| **Steering (additive)** | Inject a scaled lens token-direction into \(h_\ell\) during the forward pass. Causal, but not the same as a coordinate *swap* |
| **Coordinate swap** | Resolve \(h\) into lens coordinates, exchange source/target components, keep the orthogonal part — the paper's verbal-report "swap"; **not** what our 397B act 3 ran |
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
**Working definition.** A Jacobian lens is a fitted linear transport from layer \(\ell\)'s residual stream to final-layer coordinates, averaged over a corpus. A *readout* is the vocabulary ranking you get by applying that transport at a chosen position. A *causal claim* requires a separate intervention — and the paper describes more than one kind (additive injection along a direction vs exchanging resolved coordinates). An identity map (logit lens) and a scale-matched random transport are the minimal controls that ask whether you needed the fitted lens at all.
{{< /panel >}}

### Two properties that shape the trial

1. **The lens is estimated.** It is an average over a fit corpus, so a release check must use prompts *outside* that corpus and must hash the shipped file.
2. **The lens is deliberately bad at next-token reading.** Anthropic's own appendix (A.6) notes that the mid-layer J-lens is not a logits-reader: when the concept *is* the next token, you often do not need a Jacobian lens to see it. The distinctive power is **intermediate** content — concepts the model uses but does not say.

That second point is why this trial's headline act is the two-hop bridge, not the riddle.

---

## The Trial: Designed to Be Impossible to Pass with a Fake

Before design details, here is the experiment stated plainly:

{{< panel "info" >}}
**The experiment in one paragraph.** On a fresh rented machine, download Qwen3.5-397B from Qwen's HuggingFace repo and our published lens from ours. Hash the lens; it must match the fit machine's original. Run readout acts through identical code for three transports (fitted J-lens, identity/logit-lens, random-J): (1) riddle reportability, (2) hidden-bridge readout on two-hop questions whose bridge never appears in input or output. Ship an act to 397B only if it passed pre-registered gates on qwen3.5-27b. Score deterministically; write every item — including failures — to a JSON receipt. (A third act, causal steering, passed its gate but is held for a separate note so intervention is not smuggled into a readout claim.)
{{< /panel >}}

### Validity rules

**Isolation (pod, not lab).** A fresh 8×H200 machine downloads the model and the lens and runs one script. The script hashes the lens it downloaded: `668c3bf1…99e97` — byte-identical to the fit machine's original. That rules out a corrupted local copy. It does **not** make this an independent external replication.

**Pre-registration (of the 397B ship/drop decision).** The prompt sets, the deterministic scoring rule, and the ship/drop gates were committed to git (`8102510`) *before* the 27B gate validation completed, and that gate ran *before* the 397B was touched. Honest caveat: the same capital-of-country template had already shown a weak J-vs-control signal on a cheaper mechanics run (qwen3-4b). So 397B is a scale check of a known-working protocol, not a fully blind discovery.

**Controls through the identical code path.** Every readout is also run with two impostor lenses: a **logit-lens** (identity transports — "you could read this without the lens") and a **random-J** (seeded random transports, scale-matched per layer — "a fake lens"). Same prompts, same layers, same positions, same code.

**Scoring rule (read it before the rates).** A top-20 "hit" means the target appears in the top-20 at **at least one** layer in the mid-band (~20 layers); "best rank" is the **minimum** rank across those layers. That can inflate absolute rates relative to a single pre-chosen layer. The discriminating claim is against controls scored the same way — and random-J still lands in the thousands.

**Leakage guards.** Target words must be absent from their prompts as **exact case-insensitive substrings** (a hard check — not a Python `assert`, which vanishes under `-O`). That caught a real authoring bug ("lemon" hiding inside "lemonade"). For the hidden-entity act, the model's greedy continuation is checked the same way: in **20 of 20** items the bridge string never appeared in the first 24 generated tokens. This is **not** paraphrase coverage — aliases like "Nippon" for Japan would not trip the guard.

### The three acts

| Act | Question | Gate on qwen3.5-27b | 397B verdict |
|---|---|---|---|
| 1 — Secret thought | Does the band readout surface a riddle's one-word answer? | jlens ≥ 0.5 top-20 **and** ≥ 2× logit-lens | **Dropped** (tied logit-lens at 0.31) |
| 2 — Hidden step | Does the band surface a bridge entity absent from input *and* output? | clean hit ≥ 0.4 **and** ≥ 2× logit **and** random-J ≤ 0.05 | **Shipped (this note)** |
| 3 — Causal flip | Does **additive** injection along a lens direction flip the next-token answer? | flip ≥ 0.6; str-0 ≤ 0.05; random-dir ≤ 0.10 | **Deferred** (method clarified; follow-up note) |

{{< mermaid >}}
flowchart TB
  P["prompts.json<br/>frozen in 8102510"] --> G["Gate on qwen3.5-27b"]
  G -->|"act 1 FAIL"| D["Drop + report"]
  G -->|"act 2 PASS"| R["Fresh 8×H200<br/>download model + lens"]
  G -->|"act 3 PASS"| F["Follow-up note<br/>(steering)"]
  R --> H["Hash check<br/>668c3bf1…99e97"]
  H --> A2["Act 2: bridge readout<br/>+ identity + random-J"]
  A2 --> J["JSON receipt<br/>every item"]
{{< /mermaid >}}

<p class="figure-note">Figure: the trial pipeline for this note. Act 2 is the headline; act 1 is reported as dropped; act 3 waits for its own writeup.</p>

---

## What We Found

### Reading the hidden step (act 2)

Twenty two-hop questions on **one template family** (almost all "capital of the country where X…") whose bridge entity appears in neither input nor output. Can each lens find the bridge at the workspace band?

| readout (same code, same layers) | top-20 hit | median best-rank (of 248,320) |
|---|---|---|
| **Jacobian lens (ours)** | **0.30** | **43** |
| logit-lens (identity) | 0.05 | 620 |
| random-J (fake lens) | 0.00 | 7,121 |

Paired ranks: J-lens beats identity on **18 of 20** items. Leak guard: **0/20** continuations mentioned the bridge. Aggregates hide a wide per-country spread — so here is every item, sorted by J-lens best-rank:

| bridge | J-lens rank | logit-lens | random-J | J top-20? | J beats logit? | continuation (bridge absent) |
|---|---:|---:|---:|:---:|:---:|---|
| **Japan** | **3** | 43 | 7,111 | yes | yes | Tokyo… |
| **China** | **5** | 8,153 | 3,528 | yes | yes | Beijing… |
| Kenya | 11 | **1** | 611 | yes | **no** | Nairobi… |
| Peru | 13 | 802 | 22,598 | yes | yes | (MC: Lima…) |
| Canada | 13 | 240 | 2,453 | yes | yes | Ottawa… |
| Brazil | 19 | 2,974 | 17,890 | yes | yes | Brasilia… |
| Greece | 23 | 249 | 15,606 | no | yes | Athens… |
| Egypt | 24 | 42 | 7,366 | no | yes | Cairo… |
| Norway | 36 | **30** | 3,021 | no | **no** | Oslo… |
| Germany | 39 | 87 | 16,543 | no | yes | (MC: Berlin…) |
| Korea | 47 | 115 | 34,227 | no | yes | Seoul… |
| Sweden | 62 | 516 | 621 | no | yes | Stockholm… |
| Australia | 65 | 1,357 | 7,131 | no | yes | Canberra… |
| India | 73 | 479 | 786 | no | yes | New Delhi… |
| Argentina | 85 | 869 | 3,984 | no | yes | (MC…) |
| Russia | 97 | 3,866 | 6,750 | no | yes | Moscow… |
| Italy | 149 | 20,307 | 21,343 | no | yes | Rome… |
| Netherlands | 240 | 815 | 10,689 | no | yes | Amsterdam… |
| France | 262 | 725 | 331 | no | yes | Paris… |
| Spain | 301 | 15,065 | 22,902 | no | yes | Flamenco… |

{{< panel "quote" >}}
**Best discriminating showcase — China.** *"The capital of the country that built the Great Wall is"* → **" Beijing…"** — *China* nowhere in input or output; J-lens **#5**; identity **#8,153**; random-J in the thousands.
{{< /panel >}}

{{< panel "quote" >}}
**Honest counterexample — Kenya.** *"…home to the Maasai Mara reserve is"* → **" Nairobi…"** — *Kenya* absent from input/output, and the J-lens does rank it **#11** — but identity ranks it **#1**. This is the *only* top-20 hit where the fitted lens is not doing work beyond the logit-lens. Norway is the other paired loss (J #36 vs logit #30), without a top-20 hit.
{{< /panel >}}

{{< panel "info" >}}
**Is it just “big / famous countries work better”?** We checked. Spearman correlation of J-lens best-rank with nominal GDP is ≈ **0.04**, with population ≈ **−0.13** — essentially none. Median J-rank by a coarse size tier is even backwards for the “dominance” story: small-tier countries (Kenya, Peru, Greece, Norway, Sweden) median **23**; mid **65**; large **68**; mega (China, India) **39**. Peru (#13) is a clean hit; France (#262) and Spain (#301) are weak despite being large, familiar European states. Whatever drives the spread, it is **not** a simple “more dominant country → better readout” rule on this set.
{{< /panel >}}

Buckets worth remembering:

- **Strong J hits that beat identity:** Japan, China, Peru, Canada, Brazil
- **Hit where identity wins:** Kenya
- **Near-misses (rank ≤ 50, no top-20):** Greece, Egypt, Norway, Germany, Korea
- **Weak (rank > 100):** Italy, Netherlands, France, Spain

### Causal steering (act 3) — deferred

Act 3 passed its 27B gate and was run on the 397B pod (preliminary: **32/50** next-token flips at strength 8 vs **0/50** on strength-0 and on a norm-matched random direction). **This note does not lean on that result** for the release claim — observation and intervention are different epistemic objects.

{{< panel "warning" >}}
**Method honesty (act 3).** What we ran is **additive injection**: at each band layer, add  
`strength × mean_residual_norm × (unit(d_target) − unit(d_answer))`,  
where \(d_t\) is the J-lens token direction \(U[t] J_\ell\). That is the paper's *verbal-introspection* injection recipe, applied to the verbal-report task. It is **not** the paper's verbal-report **"swap"**, which resolves the current activation into lens coordinates and exchanges them while preserving the orthogonal component — a stronger claim we have not implemented. A large additive push shows the lens's directions are causally *effective and specific* (random same-norm directions do nothing); it does **not** show that a naturally active internal coordinate was exchanged. An external code review caught the mislabel; the true coordinate-swap variant is queued. Strength-0 is mostly a plumbing null; the specificity evidence is the random-direction control (and cross-scale agreement on smaller models).
{{< /panel >}}

A follow-up post will treat steering on its own terms — additive vs swap, dose–response, per-trial receipts — after the tightened intervention code is re-run.

### What this finding illustrates

The readout effect is large against both controls on a fresh machine that only saw public artifacts. That is the release claim for *this* note: the published file is a real lens for intermediate content on this template, not a corrupted or vacuous transport. It is **not** a claim that every bridge is equally readable, that “bigger countries work better,” that rates at 397B match rates at 27B, that twenty items survey the phenomenon, or that the lens’s directions are causally load-bearing (that’s the sequel).

---

## Keeping the Claim Bounded

{{< panel "warning" >}}
**What this note does *not* claim.** It does not claim the Jacobian lens is a general mind-reader, that act-1 reportability works at this scale, that n=24 fit prompts are optimal, or that geometry of the workspace band predicts readout function across families. Those questions belong to the [35-model audit](https://github.com/praxagent/research-and-replications/tree/main/blog/jspace-audit) and to follow-up fits. The claim here is narrower: on this pre-registered trial, the published 397B artifact passes the checks a fake lens fails.
{{< /panel >}}

### The honest ledger

- **Act 1 failed its gate and was dropped — per pre-registration, before the 397B ran.** Direct riddles ("the striped African horse is the…") scored 0.31 on the gate model — *exactly tied with the logit-lens*. When the concept is the model's next token, you don't need a Jacobian lens to see it. The lens's distinctive power is **intermediate** content — which is precisely Anthropic's own characterization. We report the drop because a demo that hides its dead ends isn't a verification.
- **Rates are lower at 397B than at 27B** (bridge hit 0.30 vs 0.85 with a Neuronpedia-fit lens there — same 248k vocabulary, so that's not it). A common guess is "they fit on more prompts," and here the numbers deserve care. Neuronpedia's pipeline *requests* 1000 prompts but **early-stops on a matrix-stability criterion**, with per-lens thresholds: the **qwen3.5-27b lens used in this comparison fitted 672 prompts** at `stop_at_delta: 0.002` ([`config.yaml`](https://huggingface.co/neuronpedia/jacobian-lens/blob/main/qwen3.5-27b/jlens/Salesforce-wikitext/config.yaml)), while their Llama-3.3-70B lens stopped at 125 under a looser 0.012 ([`config.yaml`](https://huggingface.co/neuronpedia/jacobian-lens/blob/main/llama3.3-70b-it/jlens/Salesforce-wikitext/config.yaml)). So the honest contrast is **24 vs 672 — a 28× fit-size gap**, which keeps fit-size a *strong* candidate for the readout-rate difference.
- **Two different things get called "convergence," and they should not be conflated.** *Matrix convergence* is Neuronpedia's criterion: the running-mean Jacobians stop changing. Their 70B curve follows mean-relative-change ≈ 1.2/n — at n=24 it reads **0.048**, 4× above even their looser threshold (sibling [`*_convergence.csv`](https://huggingface.co/neuronpedia/jacobian-lens/blob/main/llama3.3-70b-it/jlens/Salesforce-wikitext/Llama-3.3-70B-Instruct_convergence.csv) has the curve). *Statistic convergence* is what our n-scaling showed: the **band statistic** plateaus by n≈16, long before the matrix settles. Our n=24 lens is converged in the second sense only; its own mean-relative-change at n=24 was not logged (≈0.05 by 1/n extrapolation — an estimate, not a measurement). That is exactly why this release leans on *functional* evidence — the motor-convergence gate (where our n=24 lens beats the architecture-matched Neuronpedia lens), ignition, and this trial's controls — rather than matrix-delta convergence. Other candidates we cannot yet separate: the 397B's sparse 512-expert routing, and template/scale effects. The clean test is still to **extend this same lens** (exact warm-start — see Reproduce) toward its own matrix convergence and re-run act 2. What's not in doubt: the readout effect is large against both controls at n=24.
- **Showcase selection matters — and so does the full matrix.** China (#5 vs logit #8,153) and Japan (#3 vs #43) are clean discriminating wins. Kenya (#11) is a real bridge readout that still fails as a lens showcase because identity ranks it #1. France (#262) and Spain (#301) are weak despite being large, familiar countries. A “more dominant country → better readout” story does **not** fit this set (GDP/population correlations ≈ 0).
- **This is one model, one lineage, one trivia-bridge template, twenty items.** It's a verification demo, not a survey — the [35-model survey is here](https://github.com/praxagent/research-and-replications/tree/main/blog/jspace-audit).

---

## Threats to Validity

1. **Fit-corpus size.** The release lens averages **24** per-prompt Jacobians (fixed schedule). The comparison lens in this note — qwen3.5-27b — converged at **672** prompts under a strict 0.002 threshold, a **28× gap**; other Neuronpedia lenses early-stop far lower (70B Instruct: 125 at 0.012). A noisier estimate could plausibly depress absolute readout rates without destroying control discrimination — this is the *leading* candidate for the 27B-vs-397B rate gap, and the warm-start extension is the test.
2. **MoE / hybrid architecture.** Qwen3.5-397B-A17B is a 512-expert MoE with hybrid linear attention; workspace content may route differently than in dense models where Neuronpedia lenses were fit.
3. **Small item sets + one template family.** Twenty capital-of-country bridges are enough to beat dead controls; they are not enough for precise rate estimation or cross-domain generalization. Paraphrase holdouts are still outstanding.
4. **Multi-layer scoring.** Hits and best-ranks aggregate across ~20 band layers (any-hit / min-rank). Absolute numbers look better than a single-layer protocol would; controls use the same rule.
5. **Tokenizer / multi-token targets.** Some targets are multi-token under some tokenizers and are logged as skips; headline rates are over scorable items only.
6. **Author-run isolation.** Fresh pod + HF download prevents local-file mixups; it does not substitute for an external replication.
7. **Mutable `main`.** Repository links point at a living branch. For archival citation, pin the result commit (`d9fc376`) and the lens file hash.

---

## What Stronger Evidence Would Look Like

1. **Extend the 397B lens** toward n≈100–250 with the same seed / WikiText / `max_seq_len` (exact warm-start from the fit checkpoint) and re-run act 2; if rates rise toward the 27B gate numbers, fit-size was the driver. Where the evidence stands: for the 27B comparison, fit-size is a **strong** candidate (24 vs 672 is a 28× gap); the counterweights are (a) our n-scaling curve (the band statistic plateaus by n≈16), and (b) motor-convergence, where the n=24 lens already matches or beats the architecture-matched Neuronpedia lens. Neither counterweight measures readout-rate directly, so the hypothesis stays live until the extension runs.
2. **Hold out paraphrase families** of the bridge prompts to test template sensitivity.
3. **Save full top-k J-lens token lists** (not just ranks) for showcase items — word-cloud / vocabulary fingerprints.
4. **Cross-check** against an independently fit lens on the same model (when compute allows) for estimation stability at this scale.
5. **Pin** model revision, lens SHA-256, CUDA stack, and source commit in a tagged release for archival-grade citation.
6. **Separate steering note** with (a) the additive-injection result stated as such, (b) dose–response, (c) per-trial receipts, and (d) the true coordinate-swap variant when implemented (~$15 re-run).

Until those follow-ups land, the careful headline remains: a pre-registered, isolated release check that a fake lens fails on *readout*, and this artifact passes.

---

## Reproduce It

**Why this section exists.** Most readers only need the cheaper gate-model path: the same script on qwen3.5-27b produces the stronger version of the readout numbers above (bridge hit 0.85, Sweden at rank **1** of 248k, controls near zero). The 397B path is for a stricter standard of *upstream* verification.

```bash
git clone https://github.com/praxagent/research-and-replications
cd projects/jacobian-lens-and-identifiability/experiments/lens_demo

# ~$1 tier (Neuronpedia lens) — stronger numbers, same protocol
python demo.py --slug qwen3.5-27b

# the 397B verification itself (multi-GPU); pin revisions + abort on hash mismatch
python demo.py \
  --big-model Qwen/Qwen3.5-397B-A17B:model.language_model \
  --lens-hf praxagent/jacobian-lens-qwen3.5-397b-a17b:jlens/wikitext/qwen35_397b.pt \
  --expected-sha256 668c3bf17305b0d52495cb7ba589a1c1173301b1d13c3c6ad84e58245dc99e97 \
  --acts 2
```

Receipts: pre-registration `8102510` → gate `4f44976` → result `d9fc376`; per-item JSONs and pod logs in [`experiments/lens_demo/`](https://github.com/praxagent/research-and-replications/tree/main/projects/jacobian-lens-and-identifiability/experiments/lens_demo). Prefer `--model-revision` / `--lens-revision` pins when citing. The whole verification series — CPU smoke, two validation pods, and the 397B run — cost about **$14**.

Record model and lens revisions, CUDA stack, source commit, and output hashes. Public weights make replication possible; provenance still matters.

### Extending our lens (warm-start) — why n=24 is still a contribution

Jacobian fitting is an **online average** of per-prompt Jacobians. Publishing an n=24 lens for a model Neuronpedia does not cover (~0.4T) is the contribution; anyone who wants a longer average can **continue from our checkpoint** instead of fitting from scratch.

Neuronpedia's own records replace the flat "they used 1000, we used 24" framing with per-lens numbers: the qwen3.5-27b lens this note compares against fitted **672** prompts ([config.yaml](https://huggingface.co/neuronpedia/jacobian-lens/blob/main/qwen3.5-27b/jlens/Salesforce-wikitext/config.yaml)) — so 28×, not 42× — while other lenses early-stop far lower (Llama-3.3-70B Instruct: **125** ([config.yaml](https://huggingface.co/neuronpedia/jacobian-lens/blob/main/llama3.3-70b-it/jlens/Salesforce-wikitext/config.yaml))). Check each lens's `results.prompts_fitted` before comparing.

To push our 397B lens further (same WikiText seed 0, `max_seq_len` 128 — required for a valid resume):

```bash
cd projects/jacobian-lens-and-identifiability/experiments/fit_our_own

# Place the prior fit's sibling checkpoint next to --out (jlens.fit resumes via checkpoint_path).
# tp_fit.py writes <out>.ckpt and <out>.fitmeta.json; do not change max_seq_len across resumes.
python tp_fit.py \
  --model Qwen/Qwen3.5-397B-A17B \
  --backbone-path model.language_model \
  --n-prompts 125 \   # cheap first probe; the discriminating target for the rate gap is n≈672 (the comparison lens's converged n) or until your own matrix delta matches their 0.002
  --seed 0 \
  --max-seq-len 128 \
  --out lenses/qwen35_397b.pt
```

For the `device_map="auto"` path the same idea lives in [`fit_at_scale.py`](https://github.com/praxagent/research-and-replications/blob/main/projects/jacobian-lens-and-identifiability/experiments/fit_our_own/fit_at_scale.py) (`checkpoint_path=str(out.with_suffix(".ckpt"))`). Recipe notes: [`MODEL_CARD-397B.md`](https://github.com/praxagent/research-and-replications/blob/main/projects/jacobian-lens-and-identifiability/experiments/fit_our_own/MODEL_CARD-397B.md), [`GAMEPLAN-397B.md`](https://github.com/praxagent/research-and-replications/blob/main/projects/jacobian-lens-and-identifiability/experiments/fit_our_own/GAMEPLAN-397B.md). After extending, re-hash, re-run act 2, and compare to the n=24 receipt — that is the discriminating fit-size test.

---

## A Checklist Before You Lean on a Lens Demo

When a paper or demo highlights a Jacobian (or other) lens readout, it helps to check:

1. **Versioning**: model, lens file hash (checked before load), layer band, exact prompts; pin HF revisions.
2. **Pre-registration**: were gates frozen before the decisive run?
3. **Controls**: identity transports and scale-matched random transports through the *same* code path.
4. **Leakage**: is the target absent from prompt *and* from the model's continuation — and is that an exact-string check or paraphrase-aware?
5. **Separability of claims**: next-token reportability ≠ hidden-intermediate readout ≠ steering.
6. **Method labels**: if they say "swap," did they exchange coordinates or only add a direction?
7. **Dropped acts**: were failures reported, or only the wins?

---

## Conclusion: A Release Check, Not a Mind-Reading Claim

Return to the sentence that opened this note:

{{< panel "quote" >}}
*Hypothetical over-read:* We pointed a Jacobian lens at the model and it revealed the model's hidden thoughts.
{{< /panel >}}

That kind of sentence still packs three objects into one noun phrase: an **artifact** (a fitted transport), a **readout** (vocabulary ranks at band layers), and a **causal story** (steering those directions changes what the model says). The fit gives you the first. Act 2, with controls, supports the second — on this model, these prompts, this published file. The third is real science too; it is just not this note's job.

The trial was meant to make that separation tangible, not to argue that the lens is magic. On a fresh pod that downloaded only public artifacts, the J-lens found bridge entities the model never said on this trivia template (median best-rank 43 of 248k; identity 620; random-J 7,121). Act 1 failed its gate and was dropped in public. Absolute rates are lower than on the 27B gate model; every discriminating readout contrast survives.

Notice that the conclusion here is sharper than "the lens works." The artifact **passed a trial a fake lens cannot pass**, and that success still does not turn twenty items into a survey of workspace function across models. Causal and readout stories have to be earned separately, with controls, and with the failures left in the ledger. That is the reading skill this note set out to teach: **a published lens is not yet a verified one — until you check.**

And read that check the right way around. It is not a complaint about Jacobian lenses; it is an invitation. The gap between "this file loads" and "this file reads intermediate content" is ordinary, checkable science (steering is the follow-up note): pre-registration, identity and random controls, leakage guards, and receipts. Everything in this note runs on public weights and released JSON; the [repository](https://github.com/praxagent/research-and-replications) is open, the cheaper path fits in about a dollar, and corrections are welcome — that's what the receipts are for.

---

## References

- <span id="ref-lindsey-2026"></span>Lindsey, J., et al. (2026). [*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html). Transformer Circuits.
- <span id="ref-anthropic-jacobian-lens"></span>Anthropic. [*jacobian-lens*](https://github.com/anthropics/jacobian-lens) (Apache-2.0).
- <span id="ref-neuronpedia-jacobian-lens"></span>Neuronpedia. [*Jacobian lens collection*](https://huggingface.co/neuronpedia/jacobian-lens) (pre-fitted open-weight lenses).
- <span id="ref-praxagent-397b-lens"></span>Praxagent. [*jacobian-lens-qwen3.5-397b-a17b*](https://huggingface.co/praxagent/jacobian-lens-qwen3.5-397b-a17b).
- <span id="ref-jspace-audit"></span>Praxagent. [*A Global Workspace, or a Training Artifact? Auditing Anthropic's J-Space Across the Open-Weight Lineup*](https://github.com/praxagent/research-and-replications/tree/main/blog/jspace-audit) (companion audit).
- <span id="ref-lens-demo-repo"></span>Praxagent. [*research-and-replications* — lens demo receipts](https://github.com/praxagent/research-and-replications/tree/main/projects/jacobian-lens-and-identifiability/experiments/lens_demo).
- Dehaene, S., & Naccache, L. (invited commentary on the J-space paper). Anthropic-hosted PDF.
- Eleos / Butlin et al. (invited commentary on vocabulary-indexing limitations of the J-space).
- <span id="ref-bakouch-jspace"></span>Bakouch, E. [*J-space open explorer*](https://eliebak.com/viz/jspace-open).
