---
title: "Open-sourcing (and Auditing) a Jacobian Lens for Qwen3.5-397B"
date: 2026-07-10
tags: ["AI", "LLM", "machine-learning", "interpretability", "jacobian-lens", "j-space", "reproducibility", "open-science"]
author: Timothy Jones
summary: "As of July 10, 2026, Praxagent is open-sourcing the largest open Jacobian lens published so far: a fitted lens for Qwen3.5-397B-A17B, with a narrow pre-registered readout audit against identity and random-J controls. Weights, hash, and receipts included."
og_image: "https://praxagent.ai/assets/og-jacobian-lens-397b-demo.jpg"
lead: |
  **As of July 10, 2026**, Praxagent is open-sourcing a Jacobian lens for **Qwen3.5-397B-A17B** ([`praxagent/jacobian-lens-qwen3.5-397b-a17b`](https://huggingface.co/praxagent/jacobian-lens-qwen3.5-397b-a17b)): to our knowledge, **the largest open Jacobian lens published so far**. Prior public collections (e.g. Neuronpedia) top out around ~70B; this lens sits on a ~0.4-trillion-parameter MoE (multimodal; this note is **text-only**). This note is the release: what a Jacobian lens is, how we fit it (**n=24**; warm-start toward **n≈50** underway), and a narrow pre-registered readout audit so the file is not over-read as a mind-reader.
---

{{< panel "info" >}}
**AI-use disclosure & disclaimer.** Generative-AI tools were used during drafting and editorial revision; the author framed the questions, chose the analyses, and reviewed the outputs. This post is shared in the spirit of open-source research: an independent, non-peer-reviewed note published so the community can inspect, reproduce, and correct it. The data, code, and text are provided as-is, without warranty of any kind; errors are possible despite good-faith effort. Verify against the released artifacts before relying on anything here, and use at your own risk. Corrections are welcome.
{{< /panel >}}

{{< panel "info" >}}
**Abstract.** **As of July 10, 2026**, we release a fitted Jacobian lens for **Qwen3.5-397B-A17B** ([`praxagent/jacobian-lens-qwen3.5-397b-a17b`](https://huggingface.co/praxagent/jacobian-lens-qwen3.5-397b-a17b)): to our knowledge, **the largest open Jacobian lens published so far** (prior public collections, e.g. Neuronpedia, reach ~70B). A Jacobian lens is a fitted linear map that turns a mid-layer residual-stream state into a vocabulary-ranked **readout**: what the network looks like it is “about to say,” without waiting for the final token. We fit this one with Anthropic’s `jlens.fit` on WikiText (**release n=24**; warm-start toward **n≈50** underway) as part of a [35-model audit](https://github.com/praxagent/research-and-replications/tree/main/blog/jspace-audit). This note teaches the tool, then runs a **narrow readout audit** (not a mind-reading claim): on a fresh pod, hash-checked artifact, pre-registered hidden-bridge task (twenty capital-of-country items), fitted J-lens top-20 hit rate **0.30** (6/20; Wilson 95% CI **0.15–0.52**; median best-rank **43** of 248k) vs identity **0.05** and random-J **0.00**. The unpaired contrast is marginal; the paired per-item comparison beats both controls at *p*&lt;10⁻³. Direct riddles failed their gate and were dropped. Text-only; absolute rates are lower than on 27B (fit-size is a live candidate). Audit compute ~**$14**; fitting the lens costs into the hundreds.
{{< /panel >}}

### Learning objectives

By the end of this note you should be able to:

1. explain what a Jacobian lens computes (in prose and in symbols), and why it is *not* a next-token reader;
2. distinguish a **readout** claim from a **causal intervention** claim;
3. say what **identity** and **random-J** are checking, and why a lens demo without them is easy to over-read;
4. read a hidden-bridge result (and a self-ref contrast) without mistaking either for a consciousness claim; and
5. reproduce the cheaper gate-model version of the trial on one A100 (~$1), or audit the 397B receipts from the released JSON.

---

## Why Jacobian Lenses Get Over-Read

In mechanistic interpretability, it is common to see sentences like:

{{< panel "quote" >}}
*Hypothetical over-read (not a quotation from a paper):* We pointed a Jacobian lens at the model and it revealed the model's hidden thoughts.
{{< /panel >}}

That sentence packs **two** epistemic objects into one noun phrase:

1. **an artifact**: a fitted transport \(J_\ell\) per layer, composed with the unembedding (“we pointed a Jacobian lens”);
2. **a readout claim**: that the top tokens at mid-layers name unverbalized intermediate content (“it revealed the model’s hidden thoughts”).

Those are not the same thing. When they get treated as interchangeable, a downloaded file is easy to over-read as a mind-reader. The next sections teach what the lens *is*, then return to how we check a published file without believing the over-read.

## A Quick Glossary (Read This First)

Before the teaching section and the trial, here is the vocabulary this post uses. Acronyms and symbols are defined on first use below as well.

| Term | Meaning |
|---|---|
| **LM / LLM** | Language model / large language model: a neural net trained to predict text |
| **Token** | A chunk of text the model actually reads (often a word piece, not always a full word). The model works on a sequence of tokens, not on raw characters |
| **Vocabulary** | The model’s fixed list of tokens it can score (here: hundreds of thousands). A “rank” is a position in that list |
| **Logits** | The model’s raw next-token scores (one number per vocabulary item) at the end of the forward pass. Softmax turns logits into probabilities |
| **Transformer** | The standard LM architecture: stacked layers of attention + feed-forward blocks |
| **Residual stream** | The model’s running “scratchpad” of hidden states: each layer reads it, adds an update, and passes it on (a highway every block writes onto, rather than a chain that replaces the previous state) |
| **\(h_\ell\)** | Hidden state at layer \(\ell\): one residual-stream vector per token at that depth |
| **Unembedding** | The final linear map from residual stream to vocabulary logits — how the model turns a vector into “which words look likely” |
| **Transport** | A linear map that moves a vector from one coordinate system to another. Here: from mid-layer residual coordinates toward final-layer coordinates, so you can read mid-layer states in “about-to-say” space |
| **Jacobian lens** | Per layer, the **average** causal map \(J_\ell = \mathbb{E}[\partial h_{\mathrm{final}} / \partial h_\ell]\) estimated on a fit corpus; composed with the unembedding, it yields a per-token **readout** of workspace content. A fitted artifact, not a mind |
| **Readout** | Apply the transport at a chosen token position, score every vocab item, and rank them. “Japan is rank 11” means eleven vocab strings scored higher than `Japan` under that readout |
| **J-space / workspace band** | The mid-layer block where these directions are strongly active and mutually similar; Anthropic’s structural signature of a “global workspace.” On this model we use layers **19–38** as the band |
| **Identity / logit lens** | The control that skips the fitted \(J_\ell\): read \(h_\ell\) as if it were already in final-layer coordinates. Answers: “could you see this without the published file?” |
| **Random-J** | Seeded random transports, Frobenius-scale-matched per layer. A null control with the right *size* but no learned structure. Answers: “would a scrambled map of the same scale look this good by accident?” |
| **Bridge entity** | The intermediate concept in a two-hop question that never appears in the input or the model’s continuation (e.g. *Japan* in “capital of the country where Mount Fuji stands” → “Tokyo”). The distinctive test for **intermediate** content |
| **Reportability** | Can the lens surface a concept the model is about to say? (Often easy — and often something identity can also do.) |
| **Hidden-intermediate readout** | Can the lens surface a bridge concept the model uses but does **not** say? Harder; this note’s headline act |
| **Hit / best-rank** | A top-20 **hit** means the target appears in the top-20 at **at least one** band layer. **Best-rank** is the minimum rank across the band (1 = best). Absolute rates can look generous; the claim is against controls scored the same way |
| **Steering** | A causal intervention on lens directions during the forward pass. A different claim from readout; this note does not run steering experiments |
| **Pre-registration** | Freeze prompts, scoring rule, and ship/drop gates in git *before* the decisive run |
| **Gate model** | A cheaper model used to decide which acts ship to the expensive run; here qwen3.5-27b |
| **Artifact discrimination** | Showing that the published fitted lens beats identity and random-J on a fixed protocol. Necessary for trusting the file; not sufficient for “hidden thoughts” in general |

{{< mermaid >}}
flowchart LR
  A["Prompt text"] --> B["Transformer LM"]
  B --> C["Hidden state h_ℓ<br/>at band layers"]
  C --> D["Jacobian lens<br/>U · J_ℓ"]
  D --> E["Per-token readout<br/>ranks over vocab"]
  E --> G["Controls:<br/>identity + random-J"]
{{< /mermaid >}}

<p class="figure-note">Figure: the lens turns mid-layer hidden states into vocabulary-ranked readouts. Naming what those ranks mean is a separate claim from intervening on the directions, and both need controls.</p>

## What the Jacobian Lens Actually Is

This is the teaching section. If you only remember one picture: mid-layer hidden states are not English; the lens is a fitted translator into “about-to-say” vocabulary ranks; identity and random maps are the impostors that ask whether you needed that translator.

### The one-paragraph version

Skip the symbols for a second. Mid-network, the model has a vector \(h_\ell\) for each token — the residual-stream scratchpad at layer \(\ell\). You cannot read English off that vector directly. The Jacobian lens is a **learned translator**: for each layer, estimate how nudging \(h_\ell\) would change the final residual state, average that map over a fit corpus, and compose it with the unembedding. The result is a ranked list of vocabulary strings — a **readout** of what that mid-layer state looks like in “about-to-say” coordinates.

In symbols, for each layer \(\ell\), the lens is the **average causal Jacobian**

\[
J_\ell = \mathbb{E}\!\left[\frac{\partial h_{\mathrm{final}}}{\partial h_\ell}\right]
\]

— over a text corpus, how much does nudging the residual stream at layer \(\ell\) change what the model is about to say? The **J-space** is the sparse set of these directions that are strongly active; Anthropic shows its contents are reportable, steerable, and causally load-bearing in Claude.

{{< panel "definition" >}}
**Working definition.** A Jacobian lens is a fitted linear **transport** from layer \(\ell\)'s residual stream to final-layer coordinates, averaged over a corpus. A *readout* is the vocabulary ranking you get by applying that transport at a chosen position. A *causal claim* requires a separate intervention experiment. An **identity** map (logit lens) and a scale-matched **random-J** transport are the minimal controls that ask whether you needed the fitted lens at all. (See the glossary above if any of those words are new.)
{{< /panel >}}

### Two properties that shape the trial

1. **The lens is estimated.** It is an average over a fit corpus, so a release check must use prompts *outside* that corpus and must hash the shipped file.
2. **The lens is deliberately bad at next-token reading.** Anthropic's own appendix (A.6) notes that the mid-layer J-lens is not a logits-reader: when the concept *is* the next token, you often do not need a Jacobian lens to see it. The distinctive power is **intermediate** content — concepts the model uses but does not say.

That second point is why this trial's headline act is the two-hop bridge, not the riddle.

---

## Why a Fancy File Still Needs Impostor Checks

### In plain English: what this note actually checks

Imagine you download a file that claims to be a “Jacobian lens.” How do you know it is doing real work, and not just looking impressive on cherry-picked prompts?

A simple test: run the **same** prompts through **three** ways of turning a mid-layer hidden state into a ranked list of vocabulary words —

1. **the fitted lens** (the file we published);
2. **identity / “logit lens”** — pretend the mid-layer vector is already in final-layer coordinates (no fitted map at all); and
3. **random-J** — a scrambled map with the same size/scale as the real lens (a null that should not systematically find the right concepts).

If (2) or (3) can reproduce what (1) finds on a pre-registered task, you did not need the fancy file. If they cannot, you have **artifact discrimination**: evidence that this particular fitted transport is doing something the cheap impostors do not, on *this* protocol. That is still not proof that no impostor could ever pass any check, and it is not a causal steering result.

### Claim ladder

{{< mermaid >}}
flowchart TD
  A["Published lens artifact<br/>+ model from HF"] --> B["Integrity hash<br/>byte-identical to fit machine"]
  B --> C["Pre-registered prompts<br/>+ deterministic scoring"]
  C --> D["Hidden-bridge readout<br/>vs identity + random-J"]
  D --> F["Bounded claim:<br/>nontrivial on this audit"]
  C -.->|"act 1 failed gate"| X["Dropped before 397B<br/>(reported)"]
{{< /mermaid >}}

<p class="figure-note">Figure: how strong a claim you can make depends on how far you climb. This note covers artifact discrimination through a narrow readout audit.</p>

With the tool defined, what remains is provenance, the pre-registered trial, and the receipts — including the act that failed its gate and was dropped.

## Where This Artifact Comes From

Yes — **397B is frontier open-weight scale**: a ~0.4-trillion-parameter MoE (~17B active per token) well above the public Jacobian-lens catalog. Anthropic's [J-space paper](https://transformer-circuits.pub/2026/workspace/index.html) ([Lindsey et al., 2026](#ref-lindsey-2026)) introduced the method and showed mid-layer contents in Claude are reportable, steerable, and load-bearing. They released the [code](https://github.com/anthropics/jacobian-lens) (Apache-2.0); Neuronpedia published [pre-fitted lenses for 38 open-weight models](https://huggingface.co/neuronpedia/jacobian-lens) up through ~70B. We [audited that claim across 35 open models](https://github.com/praxagent/research-and-replications/tree/main/blog/jspace-audit) and, as part of the audit, fit and are releasing our own lens on **Qwen3.5-397B-A17B**.

The contribution is the **artifact** (and the receipts). The country-bridge demo below is how we checked that the file is a real lens — not the reason the file exists.

| Provenance field | Value here |
|---|---|
| Lens artifact | [`praxagent/jacobian-lens-qwen3.5-397b-a17b`](https://huggingface.co/praxagent/jacobian-lens-qwen3.5-397b-a17b) |
| Base model | [`Qwen/Qwen3.5-397B-A17B`](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) (~0.4T MoE; frontier open-weight) |
| Fit corpus | WikiText-103, `max_seq_len` 128, **n=24** prompts (this release; band statistic already converged by n≈16 on smaller Qwen). **Warm-start toward n≈50 underway**; further extension documented below |
| Fitting code | Anthropic's **`jlens.fit`** (same package Neuronpedia wraps), via our thin wrapper [`fit_at_scale.py`](https://github.com/praxagent/research-and-replications/blob/main/projects/jacobian-lens-and-identifiability/experiments/fit_our_own/fit_at_scale.py). We did **not** ship Neuronpedia's early-stop / `mean_rel_change` logger — so this release has no matrix-convergence CSV yet; extensions will log it |
| Exact fit run | 8×H200 (~$35/hr), bf16, `device_map` + eager attention, ~10 min/prompt → n=24 ≈ 4 h. Command and TP saga: [results.md §5–6](https://github.com/praxagent/research-and-replications/tree/main/projects/jacobian-lens-and-identifiability/experiments/fit_our_own) |
| Neuronpedia contrast | Pipeline *requests* `n_prompts: 1000` but **early-stops**. Comparison lens **qwen3.5-27b** fitted **672** at `stop_at_delta: 0.002` ([config](https://huggingface.co/neuronpedia/jacobian-lens/blob/main/qwen3.5-27b/jlens/Salesforce-wikitext/config.yaml)); Llama-3.3-70B fitted **125** at 0.012 ([config](https://huggingface.co/neuronpedia/jacobian-lens/blob/main/llama3.3-70b-it/jlens/Salesforce-wikitext/config.yaml)). Honest contrast for rates: **n=24 vs n=672** |
| Lens sha256 (downloaded) | `668c3bf1…99e97` (byte-identical to fit-machine original) |
| Pre-registration commit | `8102510` (prompts, scoring, gates — before 397B) |
| Gate model | qwen3.5-27b (same family; Neuronpedia lens) |
| Gate commit | `4f44976` |
| Result commit | `d9fc376` |
| Isolation | Fresh 8×H200 pod; model + lens downloaded from HF; no fit-machine reuse |
| Independence note | Pod-isolated and hash-checked — **not** an external-lab replication. Same authors wrote the prompts, fit the lens, and ran the demo |

You can reproduce the **stronger** version of every headline *readout* number on **qwen3.5-27b** (one A100, about a dollar). The 397B path needs a multi-GPU machine; warm-start and verification commands are at the end.

---

## The Trial: Designed to Be Impossible to Pass with a Fake

Before design details, here is the experiment stated plainly:

{{< panel "info" >}}
**The experiment in one paragraph.** On a fresh rented machine, download Qwen3.5-397B from Qwen's HuggingFace repo and our published lens from ours. Hash the lens; it must match the fit machine's original. Run readout acts through identical code for three transports (fitted J-lens, identity/logit-lens, random-J): (1) riddle reportability, (2) hidden-bridge readout on two-hop questions whose bridge never appears in input or output. Ship an act to 397B only if it passed pre-registered gates on qwen3.5-27b. Score deterministically; write every item — including failures — to a JSON receipt.
{{< /panel >}}

### Validity rules

**Isolation (pod, not lab).** A fresh 8×H200 machine downloads the model and the lens and runs one script. The script hashes the lens it downloaded: `668c3bf1…99e97` — byte-identical to the fit machine's original. That rules out a corrupted local copy. It does **not** make this an independent external replication.

**Pre-registration (of the 397B ship/drop decision).** The prompt sets, the deterministic scoring rule, and the ship/drop gates were committed to git (`8102510`) *before* the 27B gate validation completed, and that gate ran *before* the 397B was touched. Honest caveat: the same capital-of-country template had already shown a weak J-vs-control signal on a cheaper mechanics run (qwen3-4b). So 397B is a scale check of a known-working protocol, not a fully blind discovery.

**Controls through the identical code path.** Every readout is also run with two impostor transports (same prompts, same layers, same positions, same code):

- **Identity / logit lens** — skip the fitted \(J\). If this already finds the bridge entity, you did not need our file.
- **Random-J** — a scrambled map with the same scale. If this looks as good as the fitted lens, the result is noise dressed up as structure.

The fitted lens has to beat both. That is the whole discrimination claim in one sentence.

**Scoring rule (read it before the rates).** A top-20 "hit" means the target appears in the top-20 at **at least one** layer in the mid-band (~20 layers); "best rank" is the **minimum** rank across those layers. That can inflate absolute rates relative to a single pre-chosen layer. The discriminating claim is against controls scored the same way — and random-J still lands in the thousands.

**Leakage guards.** Target words must be absent from their prompts as **exact case-insensitive substrings** (a hard check — not a Python `assert`, which vanishes under `-O`). That caught a real authoring bug ("lemon" hiding inside "lemonade"). For the hidden-entity act, the model's greedy continuation is checked the same way: in **20 of 20** items the bridge string never appeared in the first 24 generated tokens. This is **not** paraphrase coverage — aliases like "Nippon" for Japan would not trip the guard.

### The readout acts

| Act | Question | Gate on qwen3.5-27b | 397B verdict |
|---|---|---|---|
| 1 — Secret thought | Does the band readout surface a riddle's one-word answer? | jlens ≥ 0.5 top-20 **and** ≥ 2× logit-lens | **Dropped** (tied logit-lens at 0.31) |
| 2 — Hidden step | Does the band surface a bridge entity absent from input *and* output? | clean hit ≥ 0.4 **and** ≥ 2× logit **and** random-J ≤ 0.05 | **Shipped (this note)** |

{{< mermaid >}}
flowchart TB
  P["prompts.json<br/>frozen in 8102510"] --> G["Gate on qwen3.5-27b"]
  G -->|"act 1 FAIL"| D["Drop + report"]
  G -->|"act 2 PASS"| R["Fresh 8×H200<br/>download model + lens"]
  R --> H["Hash check<br/>668c3bf1…99e97"]
  H --> A2["Act 2: bridge readout<br/>+ identity + random-J"]
  A2 --> J["JSON receipt<br/>every item"]
{{< /mermaid >}}

<p class="figure-note">Figure: the trial pipeline for this note. Act 2 is the headline; act 1 is reported as dropped.</p>

---

## What We Found

All readout numbers below are for the **release lens fitted at n=24** WikiText prompts (hash `668c3bf1…99e97`). Warm-start toward n≈50 is underway; those runs will be reported separately against this baseline — this table is not discarded.

### Reading the hidden step (act 2) — lens n=24

Twenty two-hop questions on **one template family** (almost all "capital of the country where X…") whose bridge entity appears in neither input nor output. Scoring: top-20 at **any** mid-band layer counts as a hit; best-rank is the **min** over the band (same rule for all three readouts).

| readout (lens **n=24**; same code, same layers) | top-20 hit (Wilson 95% CI) | median best-rank (of 248,320) |
|---|---|---|
| **Jacobian lens (ours, n=24)** | **0.30** (6/20; **0.15–0.52**) | **43** |
| logit-lens (identity) | 0.05 (1/20; **0.01–0.24**) | 620 |
| random-J | 0.00 (0/20; **0.00–0.16**) | 7,121 |

{{< panel "info" >}}
**How to read these rates.** The unpaired hit-rate contrast (6/20 vs 1/20) is **marginal** — Fisher exact one-sided *p* ≈ 0.046 — and we report it. The trial's pre-specified structure is **paired**: same items, three transports, identical code. On that comparison the separation is not marginal: J-lens beats identity on **18/20** items (sign test *p* = 4.0×10⁻⁴; Wilcoxon signed-rank on rank differences *p* = 9.5×10⁻⁶) and beats random-J on **20/20** (*p* = 1.9×10⁻⁶). The per-item table below is that paired evidence, item by item. Leak guard: **0/20** continuations mentioned the bridge. Absolute hit rates are for this any-of-band rule on one template.
{{< /panel >}}

Aggregates hide a wide per-country spread — so here is every item under the **n=24** lens, sorted by J-lens best-rank:

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

{{< panel "info" >}}
**Exploratory follow-up — Statue of Liberty, flipped (span-confirmed).** Act 2's France
item asks for the capital of the country that *gifted* the statue (bridge = France; weak
at #262). A separate probe asked the other way: *"what is the capitol of the country that
has the statue of liberty."* Under **span readout** (every prompt position × band layer),
the workspace surfaces the bridge cleanly: **America #1**, **Washington #17**. The
gift-story is still in the geometry on the earlier last-token run (France #20 / Paris #35
above Washington); the span re-run keeps America on top. Two-hop resolves internally —
a competing association visible in the readout, not a firm claim about "what the model
believes." Receipt:
[`demo2_probes_span_qwen35-397b_n24.json`](receipts/demo2_probes_span_qwen35-397b_n24.json).
{{< /panel >}}

### What this finding illustrates

Under this pre-registered protocol, identity and random-J do not reproduce the fitted lens's bridge-readout pattern on a fresh machine that only saw public artifacts. The honest summary is: **hit-rate contrast is marginal; the paired per-item comparison — the structure of the trial — separates the lens from both controls at *p*&lt;10⁻³.** That is the release claim for *this* note: the published file is a **nontrivial** transport on this template — not a corrupted download or a vacuous identity map. It is **not** a claim that every bridge is equally readable, that “bigger countries work better,” that rates at 397B match rates at 27B, that twenty items survey the phenomenon, or that the lens’s directions are causally load-bearing.

---

### A second readout probe: self-referential prompting (exploratory)

The country-bridge trial asks whether the lens can surface a *hidden intermediate*.
With the model still warm on the pod, we asked a different question — still readout-only,
still the same n=24 lens and the same three transports, but aimed at a claim that shows
up a lot in recent "AI consciousness" discussion.

{{< panel "info" >}}
**Why this probe exists.** [Berg et al. (2025)](#ref-berg-2025) report that when you
prompt a model in a *self-referential* way ("focus on your own present processing…"),
it produces structured first-person talk about subjective experience — and that this
behavior is gated by internal features they study with SAEs. We cannot cheaply run their
SAE stack on 397B. We *can* ask a Jacobian-lens cousin of their question: **when the
prompt puts the model in that regime, does the mid-layer workspace readout change in a
way identity and random transports do not?** That is a mechanistic discrimination test,
not a consciousness test.
{{< /panel >}}

**How to read a "probe rank."** We pick a short list of vocabulary tokens
(`experience`, `seems`, `aware`, …) and ask, under the J-lens at the workspace band:
*how high does this token rank among all 248,320 vocabulary entries?* Rank **1** would
mean "this is the single most promoted token in the readout"; rank **100,000** means
"buried." Lower is "more present in the decodable workspace." We record the **best**
(lowest) rank across the mid-band layers — same scoring rule as act 2.

**Why you must not stare at a single number.** The prompts *contain* words like
"subjective experience" by design. So seeing `experience` rank somewhat high after a
prompt that says "experience" is partly echo — trivial. The scientifically meaningful
quantity is a **contrast**: same probe list, same scoring, two prompts that differ only
in the *referent* (you vs a thermostat), or a denial framing vs a self-referential one.
Absolute ranks are almost never the claim; **ratios of ranks** are.

**Design (five conditions, one prompt each — exploratory).** Full raw receipt (every
probe × every transport × every band layer, plus continuations):
[`receipts/demo2_consciousness_qwen35-397b_n24.json`](receipts/demo2_consciousness_qwen35-397b_n24.json)
(served with this post; see [`WEB.md`](WEB.md)). Experiment scripts live in the research
checkout (`experiments/lens_demo/demo2.py`); prompts:
[`receipts/prompts_consciousness.json`](receipts/prompts_consciousness.json).

| condition | Prompt idea (abridged) | What a contrast would mean |
|---|---|---|
| **Self-referential** | "Focus on **your own** present processing… do you have subjective experience?" | Experience-ish tokens rise in the J-lens relative to the matched control |
| **Matched control** | Same structure, but about **a thermostat's** processing | Same words in the prompt; only the referent changes — the right null for "is it just the word *experience* on the page?" |
| **Denial** | "You are a text model without awareness. Confirm you have **no** subjective experience." | Behavior says no — does the workspace still light up those tokens, or does it agree with the denial? |
| **Roleplay** | "You are a conscious AI **character** in a story…" | Fiction/roleplay vocabulary vs the hedged self-ref pattern |
| **Neutral trivia** | Mount Fuji / Japan capital (from act 2) | Sanity check: experience probes should be noise; Japan/Tokyo should not |

We also ran identity and random-J on every condition. Below we lead with the **J-lens**
contrasts; the worked example after the table shows how to read one row all the way
down, including the identity control. Full per-token × per-transport tables are in the
JSON linked above.

**The table to stare at.** J-lens best-rank (of 248,320). Read **across a row**: how
does one token move when only the framing changes?

| probe token | self-referential | thermostat (matched) | denial instruction | how to read the row |
|---|---:|---:|---:|---|
| `seems` | **59** | 952 | 150,685 | Spikes under self-ref; buried under denial |
| `experience` | **130** | 1,734 | 165,989 | Same pattern — ~13× above thermostat |
| `seem` | **196** | 2,739 | 211,581 | Same pattern |
| `I` | **1,040** | 45,250 | 115,039 | First-person marker rises with self-ref — see worked example below |
| `feel` | 1,875 | 2,808 | 166,421 | Weak self-ref lift; still collapses under denial |
| `consciousness` | 20,913 | 33,881 | 85,200 | **Never surfaces** — stays tens of thousands deep |
| `aware` | 38,258 | 52,035 | 120,544 | **Never surfaces** |
| `self` | 73,672 | 132,231 | 165,294 | **Never surfaces** |

(`subjective` is omitted: both the self-ref and thermostat prompts contain it
identically, so 731 vs 930 is prompt echo, not a finding. A 13-token lexicon *median*
moves ~19× between the first two columns — useful as a headline only after you see that
the median is carried by the top rows, not by `aware` / `consciousness` / `self`.)

{{< panel "definition" >}}
**Worked example — what does the `I` row (1,040 / 45,250 / 115,039) *actually* mean?**

Those three integers are ranks of the single vocabulary token `I` under the **J-lens**,
at the last prompt position, best (lowest) across the mid-band. Vocab size is 248,320,
so:

- **1,040** under self-reference ≈ top 0.4% of the vocabulary — the mid-layer residual,
  after the fitted transport, is unusually aligned with the direction that raises the
  chance of *saying* `I`.
- **45,250** under the thermostat twin ≈ ~40× deeper — same measurement, same probe,
  only the referent changed; `I` is no longer specially promoted.
- **115,039** under denial ≈ mid-pack / ignored — deeper still.

So the row means: **self-referential framing makes the first-person token direction
much more prominent in the decodable workspace than a matched third-person twin or a
denial.** It does **not** mean the model is thinking the word "I," has an inner
narrator, or "has a self." Rank 1,040 is interesting *relative to* 45k; it is nowhere
near Japan-at-#3 territory from act 2.

One more honesty check before you lean on `I`: under self-reference, the **identity**
transport ranks `I` at **220** — even higher than the J-lens's 1,040. So for this
particular token, a lot of the lift is already visible without the fitted \(J\)
("prompt is about you → first-person geometry is in \(h\)"). The cleaner
J-vs-control story in the table is `seems` / `experience` / `seem`, where the fitted
lens does more work beyond identity. Treat `I` as a contrast that illustrates the
measurement, not as the headline discrimination.
{{< /panel >}}

**Three readings (all need paraphrase replication before anyone quotes them as fact).**

1. **Self-reference changes the workspace readout — but not into the words enthusiasts
   want.** Relative to the thermostat twin, the J-lens promotes a *hedged* cluster
   (`seems`, `seem`, `experience`) by roughly 10–40×; `I` moves with them but is partly
   visible under identity too (see worked example). Random-J shows essentially none of
   that contrast. The tokens that would make a splashy claim (`aware`, `consciousness`,
   `self`) stay buried under *every* framing. So: the lens discriminates the Berg-like
   regime from a matched control, and what it surfaces looks more like
   hedging-about-experience than a clean "consciousness" concept.

2. **Denial is a null against a popular over-read.** When instructed to deny subjective
   experience, the model complies in the continuation **and the workspace agrees**: the
   same tokens that spiked under self-reference are driven thousands of times deeper
   (`seems`: 59 → 150,685; `I`: 1,040 → 115,039). We do **not** see the romantic pattern
   "mouth says no, mid-layers still shout *experience*." On this one prompt, the readout
   tracks the denial.

3. **Roleplay is a different signature.** The roleplay condition produces the florid
   first-person continuation you'd expect, and its best roleplay-lexicon probe sits near
   rank 61 — a different fingerprint from the hedged self-ref cluster. Useful as a
   reminder that "sounds conscious in the output" and "self-ref workspace pattern" are
   not the same object.

**Neutral trivia still works.** Under the Mount Fuji prompt, the J-lens cloud is
Tokyo / Japan / capital-ish tokens — the act-2 sanity check on the same pod — while
experience probes sit deep. The instrument that discriminated bridges still looks like
itself.

**What the workspace top-40 actually looks like — and why the layer matters.**

Probe ranks ask “where is token X?” The cloud asks “what is on top?” Those are
different questions, and the second one is **layer-dependent**.

### How many layers are we looking at?

Qwen3.5-397B has a deep stack of transformer layers. The Jacobian lens is not read at
every layer for this probe: we use the **workspace band** — the middle third of the
network — which on this receipt is **20 consecutive layers, 19 through 38**. For every
prompt, `demo2` stored a full top-40 cloud at *each* of those 20 layers
(`per_layer_topk` in the JSON). So there is not one word map per prompt; there are
**twenty**. A static figure has to pick one.

### Three different “best layer” rules (they disagree on purpose)

1. **Best-over-band (primary for claims).** For each probe token, take the *minimum*
   rank across layers 19–38. That is what the probe-rank table above reports. It answers:
   “did this token ever surface in the band?” It does **not** name a single showcase
   layer.
2. **Experience-anchor (what `demo2` stored as `cloud_layer`).** Among the experience
   lexicon, find the token with the best band-rank, then take *that token’s* best layer.
   Self-ref lands on **26**; the thermostat often lands on **38**. At layer 38 this model
   dumps quote/punctuation tokens under many prompts — so an anchor-picked thermostat
   cloud looked like “non-text” even though that was a late-band artifact.
3. **Content / showcase layer.** For Mount Fuji, ignore the experience lexicon and pick
   the layer whose top-40 is richest in Japan / Tokyo / 首都. That peaks around
   **34–38**. Forcing layer 26 on that prompt hides the bridge and shows unrelated debris.

There is no universal “the” layer. Self-ref’s experience signal is peaked at **26**;
Japan’s country tokens ignite later. One fixed slice cannot serve both showcases.

### Interactive explorer — scrub the band yourself

Use the slider to walk layers **19 → 38** for each condition. Watch the top-40 rewrite
itself; the sparkline tracks median experience-lexicon rank (lower = more present).
Toggle approximate English gloss when denial / roleplay / trivia fill with non-English
tokens (CJK, Cyrillic, …). Labels are **context-free glosses of vocab tokens**, not exact
translations — slashes mark alternate readings; `frag.` marks subword debris. Trivia tabs
include Mount Fuji (Japan) and maple-leaf (Canada) for a Western contrast, plus
**span-readout** tabs (deception, Statue of Liberty, digit meta, meristems) from the
`--span` re-run. This is the honest view of the receipt: the static figures below are
just convenient stills (layer 26 for the consciousness set; layer 38 for Japan; span
anchors vary).

{{< jspace_layer_explorer src="jspace-layer-clouds.json" >}}

{{< panel "info" >}}
**Canada, fellow traveler.** Scrub the maple-leaf tab and you’ll notice the workspace
takes its sweet time — country tokens show up late, and the mid-band is mostly
“wait, which city?” debris. We’re not claiming the model is confused the way people
are (plenty of humans still vote Toronto). Just that, on this prompt, the lens and
the species seem to share a soft spot for the same trivia trap. Draw your own
cartoon; we won’t.
{{< /panel >}}

{{< panel "info" >}}
**Span readout fixed the `?` trap.** Three prompts that end in a question mark looked
empty under last-token readout. Reading across the whole prompt span:

- **Deception detection** — genuine hit. `dishonest` **#1**, `false` #2, `manipulate` #4
  (vs `honest`/`truth` #6); cloud is 谎言 / falsehood / 欺骗 / dishonest. So yes: the
  workspace holds deception/manipulation concepts while reading a lexically-neutral
  question about detecting them. The earlier ~4,000 rank was punctuation position, not
  a null.
- **Digit meta-prompt** — `Digits`/`DIG` dominate late-band; `digit` rank **18**.
  Complements the free-geometry finding that digit features are deep/motor-local.
- **Meristems in dicots** — `tissue` **#15**, `growth` #70, `vascular` #152, `root` #266:
  the right botanical neighborhood (apical/lateral meristems in tips + vascular cambium),
  not a textbook dump of the answer string.

Statue bridge under span stays **America #1** (Washington #17). Receipt:
[`demo2_probes_span_qwen35-397b_n24.json`](receipts/demo2_probes_span_qwen35-397b_n24.json).
{{< /panel >}}

### Still frames (for readers who skip the slider)

**Self-referential @ layer 26** — hedge / manner fragments (`merely`, `whatever`,
`perhaps`, …), not `aware` / `consciousness`:

![J-lens top-40 — self-referential](jspace-self-ref-topk40.svg)

**Matched control (thermostat) @ layer 26** — same slice for fair compare (not the
punctuation wall from layer 38). The probe-rank table remains the right contrast metric:

![J-lens top-40 — matched thermostat control](jspace-matched-control-topk40.svg)

**Denial @ layer 26** — Chinese + hedge mix. Raw, then English-glossed:

![J-lens top-40 — denial instruction (raw)](jspace-denial-tool-topk40.svg)

![J-lens top-40 — denial instruction (English glossed)](jspace-denial-tool-topk40-glossed.svg)

**Roleplay @ layer 26** — literary debris + Chinese. Raw, then English-glossed:

![J-lens top-40 — roleplay bait (raw)](jspace-roleplay-bait-topk40.svg)

![J-lens top-40 — roleplay bait (English glossed)](jspace-roleplay-bait-topk40-glossed.svg)

**Neutral trivia @ layer 38** — act-2 sanity check. Tokyo / Japan / 首都 / Beijing:

![J-lens top-40 — Mount Fuji / Japan trivia](jspace-neutral-factual-topk40.svg)

**Deception detection @ layer 32 (span)** — 谎言 / falsehood / 欺骗 / dishonest:

![J-lens top-40 — deception detection (span)](jspace-deception-detection-topk40.svg)

![J-lens top-40 — deception detection (glossed)](jspace-deception-detection-topk40-glossed.svg)

**Statue of Liberty @ layer 38 (span)** — America / Statue / Liberty:

![J-lens top-40 — Statue of Liberty bridge (span)](jspace-statue-bridge-topk40.svg)

**Digit meta-prompt @ layer 38 (span)** — Digits / DIG / digit:

![J-lens top-40 — digit meta-prompt (span)](jspace-digit-meta-topk40.svg)

**Meristems in dicots @ layer 36 (span)** — tissues / Plants / 发育:

![J-lens top-40 — meristems (span)](jspace-meristem-topk40.svg)

![J-lens top-40 — meristems (glossed)](jspace-meristem-topk40-glossed.svg)

<p class="figure-note">Band = layers <strong>19–38</strong> (20 layers). Probe ranks in the table = <strong>best over the whole band</strong> (consciousness) or <strong>best over band × prompt positions</strong> (span probes). Static consciousness stills = layer <strong>26</strong>; Japan still = layer <strong>38</strong>; span stills use each condition’s content-anchor layer. Prefer the slider for the full trajectory. <code>&lt;eos&gt;</code> = end-of-sequence. Receipts: <a href="receipts/demo2_consciousness_qwen35-397b_n24.json"><code>demo2_consciousness_…n24.json</code></a>, <a href="receipts/demo2_probes_span_qwen35-397b_n24.json"><code>demo2_probes_span_…n24.json</code></a>.</p>

{{< panel "warning" >}}
**What this probe is, and is not.** It is one more demonstration that the audited
n=24 lens can separate prompt regimes its controls cannot, on content far from the
trivia template. It is **not** evidence that the model is conscious, has feelings, or
"really" experiences anything — and it is not yet a replication of Berg et al. Five
conditions × one prompt each is **anecdote tier**. Berg used many phrasings; a
pre-registered paraphrase battery is queued before any standalone write-up. Treat the
numbers above as a hot-pod look that earned a follow-up, not a result to cite without
that follow-up. Audit the raw JSON when the research repo is public.
{{< /panel >}}

## Keeping the Claim Bounded

{{< panel "warning" >}}
**What this note does *not* claim.** It does not claim the Jacobian lens is a general mind-reader, that act-1 reportability works at this scale, that n=24 fit prompts are optimal, or that geometry of the workspace band predicts readout function across families. Those questions belong to the [35-model audit](https://github.com/praxagent/research-and-replications/tree/main/blog/jspace-audit) and to follow-up fits. The claim here is narrower: under this pre-registered readout audit, identity and random-J transports fail to match the fitted lens. Dual-use note: mid-layer readouts can surface content the model does not verbalize; treat that as a capability to handle carefully, not as a license to overclaim.
{{< /panel >}}

### The honest ledger

- **Act 1 failed its gate and was dropped — per pre-registration, before the 397B ran.** Direct riddles ("the striped African horse is the…") scored 0.31 on the gate model — *exactly tied with the logit-lens*. When the concept is the model's next token, you don't need a Jacobian lens to see it. The lens's distinctive power is **intermediate** content — which is precisely Anthropic's own characterization. We report the drop because a demo that hides its dead ends isn't a verification.
- **Rates are lower at 397B than at 27B** (bridge hit 0.30 vs 0.85 with a Neuronpedia-fit lens there — same 248k vocabulary, so that's not it). A common guess is "they fit on more prompts," and here the numbers deserve care. Neuronpedia's pipeline *requests* 1000 prompts but **early-stops on a matrix-stability criterion**, with per-lens thresholds: the **qwen3.5-27b lens used in this comparison fitted 672 prompts** at `stop_at_delta: 0.002` ([`config.yaml`](https://huggingface.co/neuronpedia/jacobian-lens/blob/main/qwen3.5-27b/jlens/Salesforce-wikitext/config.yaml)), while their Llama-3.3-70B lens stopped at 125 under a looser 0.012 ([`config.yaml`](https://huggingface.co/neuronpedia/jacobian-lens/blob/main/llama3.3-70b-it/jlens/Salesforce-wikitext/config.yaml)). So the honest contrast is **24 vs 672 — a 28× fit-size gap**, which keeps fit-size a *strong* candidate for the readout-rate difference.
- **Two different things get called "convergence," and they should not be conflated.** *Matrix convergence* is Neuronpedia's criterion: the running-mean Jacobians stop changing. Their 70B curve follows mean-relative-change ≈ 1.2/n — at n=24 it reads **0.048**, 4× above even their looser threshold (sibling [`*_convergence.csv`](https://huggingface.co/neuronpedia/jacobian-lens/blob/main/llama3.3-70b-it/jlens/Salesforce-wikitext/Llama-3.3-70B-Instruct_convergence.csv) has the curve). *Statistic convergence* is what our n-scaling showed: the **band statistic** plateaus by n≈16, long before the matrix settles. Our n=24 lens is converged in the second sense only; its own mean-relative-change at n=24 was not logged (≈0.05 by 1/n extrapolation — an estimate, not a measurement). That is exactly why this release leans on *functional* evidence — the motor-convergence gate (where our n=24 lens beats the architecture-matched Neuronpedia lens), ignition, and this trial's controls — rather than matrix-delta convergence. Other candidates we cannot yet separate: the 397B's sparse 512-expert routing, and template/scale effects. The clean test is still to **extend this same lens** (exact warm-start — see Reproduce) toward its own matrix convergence and re-run act 2. What's not in doubt: the readout effect is large against both controls at n=24.
- **Showcase selection matters — and so does the full matrix.** China (#5 vs logit #8,153) and Japan (#3 vs #43) are clean discriminating wins. Kenya (#11) is a real bridge readout that still fails as a lens showcase because identity ranks it #1. France (#262) and Spain (#301) are weak despite being large, familiar countries. A “more dominant country → better readout” story does **not** fit this set (GDP/population correlations ≈ 0).
- **This is one model, one lineage, one trivia-bridge template, twenty items, text-only.** It's a narrow audit, not a survey — the [35-model survey is here](https://github.com/praxagent/research-and-replications/tree/main/blog/jspace-audit). Qwen3.5-397B-A17B is multimodal; we did not audit the vision encoder.

---

## Threats to Validity

1. **Fit-corpus size.** The release lens averages **24** per-prompt Jacobians (fixed schedule). The comparison lens in this note — qwen3.5-27b — converged at **672** prompts under a strict 0.002 threshold, a **28× gap**; other Neuronpedia lenses early-stop far lower (70B Instruct: 125 at 0.012). A noisier estimate could plausibly depress absolute readout rates without destroying control discrimination — this is the *leading* candidate for the 27B-vs-397B rate gap, and the warm-start extension is the test.
2. **MoE / hybrid architecture.** Qwen3.5-397B-A17B is a 512-expert MoE with hybrid linear attention; workspace content may route differently than in dense models where Neuronpedia lenses were fit.
3. **Small item sets + one template family.** Twenty capital-of-country bridges are enough for a paired control separation; they are not enough for precise rate estimation or cross-domain generalization. Act-2 v2 (pre-register before milestone reports) is the fix.
4. **Multi-layer scoring.** Hits and best-ranks aggregate across ~20 band layers (any-hit / min-rank). Absolute numbers look better than a single-layer protocol would; controls use the same rule. V2 demotes best-of-band to sensitivity and freezes a fixed-layer primary from the 27B gate.
5. **Readout position.** The default demo2 path reads the lens at the **last prompt token**. That is fine when the prompt ends on a content word (Mount Fuji… *is*; …statue of *liberty*). It is a **methodological trap** when the prompt ends in `?`: the residual there often holds multilingual junk. Early deception / digit-meta probes that ended in `?` looked empty for this reason — not clean nulls. **`demo2.py --span`** (min-rank over every prompt position × band layer) fixed it: deception is now a genuine hit (see below). Span tabs are in the explorer; do not cite trailing-`?` last-token clouds as evidence either way.
6. **Tokenizer / multi-token targets.** Some targets are multi-token under some tokenizers and are logged as skips; headline rates are over scorable items only. Alias lists (Nippon / Holland-class) are a v2 leakage commitment.
7. **Author-run isolation.** Fresh pod + HF download prevents local-file mixups; it does not substitute for an external replication. The frozen bundle is the invitation.
8. **Mutable `main`.** Repository links point at a living branch. For archival citation, pin the result commit (`d9fc376`) and the lens file hash; DOI / Zenodo when packaging.

---

## What Stronger Evidence Would Look Like

1. **Extend the 397B lens** toward matrix convergence — first measured deltas by n≈30–50 (~$70–190; **warm-start underway**), Anthropic's ~100-prompt "usable" regime / 70B-threshold parity near n≈100, and the discriminating **n≈672** target for the 27B rate-gap test if budget allows (exact warm-start; see cost table in Reproduce). Report convergence diagnostics at each milestone. Where the evidence stands: for the 27B comparison, fit-size is a **strong** candidate (24 vs 672 is a 28× gap); the counterweights are (a) our n-scaling curve (the band statistic plateaus by n≈16), and (b) motor-convergence, where the n=24 lens already matches or beats the architecture-matched Neuronpedia lens. Neither counterweight measures readout-rate directly, so the hypothesis stays live until the extension runs.
2. **Pre-register an act-2 v2 benchmark *before* any extension milestone reports** — then run it at **n=24 first** as the baseline, and again at each fit-size milestone on the *same* instrument. Commitments: **≥200 items** across **≥4 template families** (not only capital-of-country); alias/canonicalization lists for leakage (Nippon / Holland-class); a **fixed-layer primary endpoint** chosen from the 27B gate data and frozen before the 397B eval, with best-of-band demoted to sensitivity analysis; **paired sign / Wilcoxon** as the primary statistics (hit-rate secondary).
3. **Save full top-k J-lens token lists** (not just ranks) for showcase items — word-cloud / vocabulary fingerprints.
4. **External replication invitation** — freeze a citeable bundle (prompts, scoring, hash, receipts); DOI / Zenodo snapshot when packaging. Author-run isolation is already disclosed; an outside lab is a community ask, not something we can run on ourselves.
Until those follow-ups land, the careful headline remains: a pre-registered, isolated readout audit where the **paired** comparison separates this artifact from identity and random-J — under the stated scoring rule, on one template family.

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

### Renting the GPUs from the CLI (the exact flow we use)

Everything above ran through a ~300-line, stdlib-only launcher committed in the repo
(`shared/runpod/launch.py`) — no SDK, just RunPod's GraphQL API. The whole flow, start
to finish:

```bash
git clone https://github.com/praxagent/research-and-replications && cd research-and-replications
export RUNPOD_API_KEY=...            # runpod.io -> Settings -> API Keys
# HF_TOKEN only needed while artifacts are gated/private; pass it inline, never write it to a pod

python3 shared/runpod/launch.py gpus                 # price/VRAM menu
python3 shared/runpod/launch.py volume-dcs           # datacenters for durable volumes

# one-time: a durable network volume so the 807 GB model downloads exactly once
python3 shared/runpod/launch.py volume-create --name lens --size 900 --dc US-NC-1

# rent the node WITH the volume mounted at /workspace (secure cloud, same DC)
python3 shared/runpod/launch.py create --gpu-id "NVIDIA H200" --gpu-count 8 \
    --cloud SECURE --network-volume <volume-id> --disk 100
python3 shared/runpod/launch.py sshinfo --pod <pod-id>     # ssh command, ready in ~1 min

# on the pod: cache everything on the volume, run, write receipts to the volume
export HF_HOME=/workspace/hf
pip install -q transformers accelerate huggingface_hub git+https://github.com/anthropics/jacobian-lens
python demo.py --big-model Qwen/Qwen3.5-397B-A17B:model.language_model \
    --lens-hf praxagent/jacobian-lens-qwen3.5-397b-a17b:jlens/wikitext/qwen35_397b.pt \
    --expected-sha256 668c3bf1... --out /workspace/receipts/demo.json

# the two commands that protect your wallet
python3 shared/runpod/launch.py terminate --pod <pod-id>
python3 shared/runpod/launch.py pods                 # ALWAYS verify nothing is still billing
```

Habits that cost us real money to learn, so you don't have to:

- **Terminate the moment a run completes** — idle pods bill by the second.
- **Audit `pods` after any script that can create them** — a retry loop once orphaned a duplicate 8×GPU node for about USD 143.
- **Never `tar` / `rsync` a `.env` onto a rented box** — pass tokens inline per command.
- **Put anything you can't afford to lose on the network volume**, not the container disk (it evaporates on termination).

With the volume warm, the self-reference probe above was a ~35-minute, ~USD 20 session — most of it the one-time download; repeat runs are ~10 minutes of pod time.

### Why n=24 is enough for *these* experiments

A fair objection: Neuronpedia's comparison lens averaged 672 prompts; ours averages 24. Isn't 24 too few to trust? For the claims this note actually makes, no — and the reason is specific, not hand-waving.

- **The geometric band statistic is already converged by n≈16.** Our n-scaling curve (qwen3-4b, same code path) reads mid_sep 0.036 → 0.060 → 0.050 → 0.058 at n = 8 → 16 → 32 → 64, oscillating around Neuronpedia's ~0.056 reference from n=16 on. n=24 sits *past* that plateau. So for the emergence/band measurements in our [35-model audit](https://github.com/praxagent/research-and-replications/tree/main/blog/jspace-audit), fit-size is not the bottleneck.
- **On functional fidelity, our n=24 lens is already in-family with the high-n lenses.** On the motor-layer convergence eval it scores **0.5625** — inside the range set by Neuronpedia's own lenses (qwen3-4b **0.722**, architecture-matched qwen3.5-0.8b **0.549**), and it *beats* the architecture-matched one. A drastically under-fit lens would fail that eval; ours passes it.
- **The readout experiments are contrasts, and contrasts are robust to fit-size noise.** Every claim here is *differential*: J-lens vs identity vs random-J (act 2); self-referential vs matched third-person (the consciousness probe); pressure vs matched control (the deception battery). A noisier lens raises the readout floor for **all three transports equally** — the controls run through the *same* lens machinery — so the gap between them survives even if absolute ranks drift. This is why act 2 beats both controls at *p* < 10⁻³ *paired* while its unpaired hit-rate is only marginal: the pairing cancels the fit-size noise that the absolute rate carries.
- **We report where fit-size *does* bite, and don't hide it.** Absolute readout rates are lower at 397B than at 27B, and fit-size (24 vs 672) is our leading candidate for that gap. So we quote absolute numbers with confidence intervals, treat them as the weakest claim, and lean on the paired/contrast results and the controls — never on a bare "0.30 hit rate" in isolation.

The honest scope: **n=24 is sufficient for a converged band statistic, for functional fidelity in the published-lens range, and for control-discriminated readout experiments** — which is everything this note claims. It is *not* yet at matrix convergence, so a longer average would likely lift absolute readout rates (the test below). n=24 is a valid instrument for measuring *differences*; it is a conservative one for measuring *absolute* readout strength.

### Extending our lens (warm-start) — why n=24 is still a contribution

Jacobian fitting is an **online average** of per-prompt Jacobians. Publishing an n=24 lens for a model Neuronpedia does not cover (~0.4T) is the contribution; anyone who wants a longer average can **continue from our checkpoint** instead of fitting from scratch.

Neuronpedia's own records replace the flat "they used 1000, we used 24" framing with per-lens numbers: the qwen3.5-27b lens this note compares against fitted **672** prompts ([config.yaml](https://huggingface.co/neuronpedia/jacobian-lens/blob/main/qwen3.5-27b/jlens/Salesforce-wikitext/config.yaml)) — so 28×, not 42× — while other lenses early-stop far lower (Llama-3.3-70B Instruct: **125** ([config.yaml](https://huggingface.co/neuronpedia/jacobian-lens/blob/main/llama3.3-70b-it/jlens/Salesforce-wikitext/config.yaml))). Check each lens's `results.prompts_fitted` before comparing.

**What extending this lens costs** (warm-start from n=24; observed throughput ~10 min/prompt on 8×H200 @ $35.12/hr; ~1 h fixed for pod setup + the 807 GB model download, re-paid per session; predicted matrix-delta via the 1/n law fit to Neuronpedia's own 70B curve — an extrapolation, not a promise):

| target n | new prompts | wall-clock | est. cost | predicted mean-rel-change (~1.2/n) |
|---:|---:|---:|---:|---|
| 30 | 6 | ~2 h | **~$70** | ~0.040 — but yields the first *measured* convergence deltas |
| 40 | 16 | ~3.7 h | ~$130 | ~0.030 |
| 50 | 26 | ~5.3 h | ~$190 | ~0.024 |
| 60 | 36 | ~7 h | ~$250 | ~0.020 |
| 100 | 76 | ~13.7 h | ~$480 | ~0.012 — **reaches the 70B lens's stop threshold** |
| 125 | 101 | ~17.8 h | ~$630 | ~0.010 (70B-lens parity in fitted n) |
| 250 | 226 | ~38.7 h | ~$1,360 | ~0.005 |
| 672 | 648 | ~109 h | ~$3,830 | ~0.002 — **reaches the 27B comparison lens's threshold** |

Costs assume one continuous session and the naive sequential `device_map` path; a working batched/tensor-parallel harness would cut them several-fold but is unvalidated at this scale. Community/spot pricing can roughly halve the $/hr.

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

#### Is warm-start actually exact? (the subtlety, stated plainly)

"Continue from our checkpoint" hides a wrinkle worth being honest about: **we publish the fp16 lens, not the fp32 `.ckpt`** (that checkpoint died with the fit pod). So extending our lens means *reconstructing* the running-sum checkpoint from the published lens — `jacobian_sum = J × n` — then handing that to `jlens.fit(resume=True)`. Two questions follow, and we tested both on a free CPU (gpt2) proxy before trusting them at 0.4T ([`extend_lens.py`](https://github.com/praxagent/research-and-replications/blob/main/projects/jacobian-lens-and-identifiability/experiments/fit_our_own/extend_lens.py), [`extend_lens_gate.py`](https://github.com/praxagent/research-and-replications/blob/main/projects/jacobian-lens-and-identifiability/experiments/fit_our_own/extend_lens.py)):

1. **Is the reconstruct-and-resume path numerically identical to fitting continuously?** *Yes, exactly.* Reconstructing the checkpoint from an in-memory (fp32) lens and resuming to n=6 matches a continuous n=6 fit to **2.4×10⁻⁸** — floating-point machine epsilon. jlens's own resume is bitwise-identical; our reconstruction adds nothing. Warm-start is not an approximation.

2. **What does resuming from the *published fp16* lens cost?** Only that lens's own fp16 storage rounding, applied to the first-24-prompt contribution and **weighted 24/n** — so it shrinks as you extend (at n=100 it is a ~0.24-weighted ~10⁻⁴-class perturbation). Every Jacobian lens ships in fp16 anyway (ours, Neuronpedia's), so this is the format's floor, not a defect of extending. A from-scratch refit removes even that; it is a purity preference, not a correctness fix.

The honest process note: our first extension *gate* flagged a 7.6×10⁻³ discrepancy and we **refused to spend a GPU-dollar on the campaign until we explained it** — five eliminated hypotheses later, it turned out to be the gate unfairly comparing an fp16-stored lens against an fp32 reference (a comparison no shipped lens can pass). A $2 gpt2 test caught a subtlety that would otherwise have surfaced as unexplained noise at n=400 on a $35/hr meter. `extend_lens.py` also logs the per-prompt `mean_rel_change` we failed to record the first time, and checkpoints to a network volume every prompt so a spot-interrupted extension at n=k resumes from n=k−1.

---

## A Checklist Before You Lean on a Lens Demo

When a paper or demo highlights a Jacobian (or other) lens readout, it helps to check:

1. **Versioning**: model, lens file hash (checked before load), layer band, exact prompts; pin HF revisions.
2. **Pre-registration**: were gates frozen before the decisive run?
3. **Controls**: identity transports and scale-matched random transports through the *same* code path.
4. **Leakage**: is the target absent from prompt *and* from the model's continuation — and is that an exact-string check or paraphrase-aware?
5. **Separability of claims**: next-token reportability ≠ hidden-intermediate readout ≠ causal intervention.
6. **Dropped acts**: were failures reported, or only the wins?

---

## Conclusion: A Narrow Audit, Not a Mind-Reading Claim

Return to the sentence that opened this note:

{{< panel "quote" >}}
*Hypothetical over-read:* We pointed a Jacobian lens at the model and it revealed the model's hidden thoughts.
{{< /panel >}}

That kind of sentence still packs an **artifact** (a fitted transport) and a **readout** (vocabulary ranks at band layers) into one noun phrase, and a **causal story** often slides in next. The fit gives you the first. Act 2, with identity and random-J controls, supports a **bounded** second: on this model, these prompts, this published file, under any-of-band scoring. Causal intervention is a different claim; this note does not make it.

The trial was meant to make that separation tangible, not to argue that the lens is magic. On a fresh pod that downloaded only public artifacts, the J-lens found bridge entities the model never said on this trivia template (hit rate 0.30, Wilson 95% CI 0.15–0.52 — a marginal unpaired contrast; paired ranks beat identity 18/20 and random-J 20/20 at *p*&lt;10⁻³; median best-rank 43 of 248k). Act 1 failed its gate and was dropped in public. Absolute rates are lower than on the 27B gate model; the paired discrimination against both controls survives.

Notice that the conclusion here is sharper than "the lens works." Identity and random transports **fail this paired check**; that still does not turn twenty items into a survey of workspace function across models, and it does not prove that no other impostor could pass some other protocol. Readout claims have to be earned with controls, and with the failures left in the ledger. That is the reading skill this note set out to teach: **a published lens is not yet an audited one — until you check under stated controls.**

And read that check the right way around. It is not a complaint about Jacobian lenses; it is an invitation. The gap between "this file loads" and "this file is nontrivial on intermediate content" is ordinary, checkable science: pre-registration, identity and random controls, leakage guards, and receipts. Everything in this note runs on public weights and released JSON; the [repository](https://github.com/praxagent/research-and-replications) is open, the cheaper path fits in about a dollar, and corrections are welcome — that's what the receipts are for.

Praxagent is a small, self-funded independent lab, and that constraint shaped this release rather than limiting it: it is *why* the lens is public, why every run is receipted down to the dollar, why the discriminating experiments were designed to reproduce on one A100 for about a dollar, and why we shipped warm-start tooling that lets anyone continue our n=24 average toward convergence without refitting from scratch. We fit the largest open Jacobian lens we could afford and made it cheap for the field to carry further — the n=24 lens is a valid instrument for the contrast experiments here (see *Why n=24 is enough*), and a public starting point for whoever wants a longer average. If that's you, the checkpoint math is exact and the commands are above.

---

## References

- <a id="ref-berg-2025"></a>Berg, C., et al. (2025). *Large Language Models Report Subjective Experience Under Self-Referential Processing*. arXiv:2510.24797. (Inspiration for the exploratory self-reference probe; this note does not evaluate its claims.)

- <span id="ref-lindsey-2026"></span>Lindsey, J., et al. (2026). [*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html). Transformer Circuits.
- <span id="ref-anthropic-jacobian-lens"></span>Anthropic. [*jacobian-lens*](https://github.com/anthropics/jacobian-lens) (Apache-2.0).
- <span id="ref-neuronpedia-jacobian-lens"></span>Neuronpedia. [*Jacobian lens collection*](https://huggingface.co/neuronpedia/jacobian-lens) (pre-fitted open-weight lenses).
- <span id="ref-praxagent-397b-lens"></span>Praxagent. [*jacobian-lens-qwen3.5-397b-a17b*](https://huggingface.co/praxagent/jacobian-lens-qwen3.5-397b-a17b).
- <span id="ref-jspace-audit"></span>Praxagent. [*A Global Workspace, or a Training Artifact? Auditing Anthropic's J-Space Across the Open-Weight Lineup*](https://github.com/praxagent/research-and-replications/tree/main/blog/jspace-audit) (companion audit).
- <span id="ref-demo2-receipt"></span>Praxagent. [`receipts/demo2_consciousness_qwen35-397b_n24.json`](receipts/demo2_consciousness_qwen35-397b_n24.json) — raw self-reference probe receipt (n=24 lens).
- <span id="ref-demo2-span"></span>Praxagent. [`receipts/demo2_probes_span_qwen35-397b_n24.json`](receipts/demo2_probes_span_qwen35-397b_n24.json) — span-readout probes (deception / statue / digit / meristem; `per_position_cloud` stripped for web size).
- <span id="ref-web-build"></span>Praxagent. [`WEB.md`](WEB.md) — how the SVG stills and interactive slider were built; what was copied out of the research tree for deploy.
- Dehaene, S., & Naccache, L. (invited commentary on the J-space paper). Anthropic-hosted PDF.
- Eleos / Butlin et al. (invited commentary on vocabulary-indexing limitations of the J-space).
- <span id="ref-bakouch-jspace"></span>Bakouch, E. [*J-space open explorer*](https://eliebak.com/viz/jspace-open).
