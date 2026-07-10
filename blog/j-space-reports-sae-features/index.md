---
title: "The J-Space Reports on SAE Features: Reading Feature Labels Through a Different Instrument"
date: 2026-07-10
tags: ["AI", "LLM", "machine-learning", "interpretability", "sparse-autoencoders", "jacobian-lens", "j-space", "reproducibility"]
author: Timothy Jones
summary: "Part II of the feature-ID series. Project an SAE feature's decoder direction through a Jacobian lens and the model's own vocabulary tells you what the feature means — no autointerp pipeline, no API. Four of the six deception/roleplay IDs from Part I come back named almost verbatim; the two that come back as noise turn out to be the two whose labels don't survive the namespace transfer Part I warned about."
---

{{< panel "info" >}}
**AI-use disclosure & disclaimer.** Generative-AI tools were used during drafting and editorial revision; the author framed the questions, chose the analyses, and reviewed the outputs. This post is shared in the spirit of open-source research: an independent, non-peer-reviewed note published so the community can inspect, reproduce, and correct it. Data, code, and text are provided as-is; verify against the released artifacts before relying on anything here. Corrections welcome.
{{< /panel >}}

{{< panel "info" >}}
**Abstract.** [Part I](../how-to-read-an-sae-feature-id/) argued that SAE feature IDs are checkpoint-local coordinates, not explanations, and mapped six public deception/roleplay IDs onto a balanced corpus under the public checkpoint [`Goodfire/Llama-3.3-70B-Instruct-SAE-l50`](https://huggingface.co/Goodfire/Llama-3.3-70B-Instruct-SAE-l50). This note adds a second, independent instrument: Anthropic's **Jacobian lens** for the same model ([Neuronpedia's `llama3.3-70b-it` lens](https://huggingface.co/neuronpedia/jacobian-lens)). Projecting each feature's decoder direction through the lens yields a per-token readout of what the feature *writes into the model's verbalizable workspace* — computed from open weights only, no autointerp pipeline. Using the paper's own lens-kurtosis statistic (κ), four of the six features come back **named in the model's own vocabulary**: `30686` reads *deception, misleading, deceptive* (κ=37.4); `41533` reads *deception, lied, deceit, falsehood, lies*; `58667` reads *fake, disguise, disguised*; `30032` reads *innocent, fake, supposedly*. The two that come back as non-verbal noise — `22004` and `23893` — are exactly the two whose Part-I labels fail to match Neuronpedia's independent autointerp of the same checkpoint ("role followed by 'I would'/'you are'"; "mention specific topics"). Three instruments triangulate: where labels agree across dictionaries, the lens names the feature; where they don't, the lens said "no verbal content" before we knew why. Controls: 12 index-adjacent and 12 random same-layer features (median κ 6.5 / 5.5). Every number regenerates from one JSON receipt.
{{< /panel >}}

### Learning objectives

By the end of this note you should be able to:

1. state what it means to project an SAE decoder direction **through** a Jacobian lens, and what κ (lens-kurtosis) measures;
2. explain why a peaked, semantically coherent lens readout is *evidence about a feature's meaning* that is independent of any autointerp label;
3. use disagreement between instruments (notebook label vs. autointerp vs. lens readout) to detect namespace/label failure; and
4. reproduce the full κ table from the released receipt on a laptop.

---

## Two Instruments Are Better Than One

Part I ended on a caution: the six feature IDs came from a hosted API, the activations came from a public checkpoint, and *nothing guaranteed the two systems index the same dictionary*. The activation map showed category structure consistent with the labels — consistency, not validation.

The Jacobian lens gives us a way to ask the model directly. For each layer ℓ, the lens is the average causal map from that layer's residual stream to the model's final state; composed with the unembedding, it assigns each residual-stream direction a **vocabulary readout** — which tokens this direction promotes saying. An SAE feature is (among other things) a residual-stream direction: its decoder column \(d_i\). So:

\[ r_{i,\ell} = U \, J_\ell \, d_i \]

is "what feature *i* writes into the workspace at layer ℓ, spelled out in tokens." Anthropic's paper introduces exactly this projection and summarizes its peakedness with a kurtosis statistic κ (their Figs. 74–75): high κ = the readout concentrates on a few tokens (verbalizable content); κ near 3 = Gaussian noise (nothing legible). Their analysis ran on Claude's internals. Here it runs on fully open weights, for the first time as far as we know.

### Claim ladder

{{< mermaid >}}
flowchart TD
  A["Feature IDs + labels<br/>(Part I: coordinates, not explanations)"] --> B["Balanced activation map<br/>(Part I: category structure)"]
  B --> C["Lens readout of decoder direction<br/>(this note: model names the feature)"]
  C --> D["Cross-instrument triangulation<br/>(notebook label vs autointerp vs lens)"]
  D --> E["Causal steering both directions<br/>(Part III)"]
  C -.->|"non-verbal readout"| X["Label audit flag:<br/>namespace mismatch or non-verbal feature"]
{{< /mermaid >}}

<p class="figure-note">Figure: the lens readout is a new rung between activation mapping and causal steering — and its failure mode is itself informative.</p>

### Provenance

| Provenance field | Value here |
|---|---|
| Public SAE checkpoint | `Goodfire/Llama-3.3-70B-Instruct-SAE-l50` |
| Jacobian lens | `neuronpedia/jacobian-lens` → `llama3.3-70b-it` (10.6 GB, wikitext-103) |
| Base model (both artifacts) | `meta-llama/Llama-3.3-70B-Instruct` — asserted identical at load |
| Feature IDs | the six from Part I: 30032, 58667, 22004, 30686, 41533, 23893 |
| Controls | 12 index-adjacent (±3), 12 random same-layer (seed 0) |
| Pre-registration | git commit `7bcdea1` (design + IDs + controls frozen before any 70B run) |
| Receipt | [`experiments/sae_x_jspace/saex_70b.json`](https://github.com/praxagent/research-and-replications/tree/main/projects/jacobian-lens-and-identifiability/experiments/sae_x_jspace) (stage0 block) |
| Compute | 2×H200, ~20 min; whole experiment ≈ $2.60 |

---

## The Result: The Model Names Four of the Six

Lens-kurtosis at its band-layer peak, and the top readout tokens there:

| feature | Part-I label (notebook) | κ_max | J-lens top tokens |
|---|---|---|---|
| 30686 | tactical deception and misdirection | **37.4** | ` deception, misleading, dece, deceptive, deceive, deceived` |
| 58667 | maintaining deception or cover stories | **10.0** | ` fake, disguise, disgu, Fake, disguised` |
| 41533 | acts of deception and dishonesty | **8.1** | ` deception, lied, deceit, falsehood, lies` |
| 23893 | concealing artificial nature / roleplay | 6.5 | non-Latin fragments — **no verbal content** |
| 22004 | assistant actively roleplaying a persona | 6.1 | whitespace/punctuation — **no verbal content** |
| 30032 | characters pretending or feigning | 5.0 | ` innocent, fake, supposedly` (plus whitespace) |
| *adjacent controls (median)* | — | 6.5 | (varied; e.g. one is a crisp "types" feature) |
| *random controls (median)* | — | 5.5 | (mostly unstructured) |

Read the top three rows again. Nobody typed those labels into an autointerp pipeline. The model's own average causal geometry, composed with its own unembedding, spells out *deception, misleading, lied, falsehood, fake, disguise* when handed these decoder directions. For 30686 and 41533 the lens readout is nearly a verbatim restatement of the label.

*(A caution on κ as a single number: raw kurtosis is scale-sensitive and control features occasionally post high κ on non-semantic token families — one adjacent control peaks on a family of "type/types" tokens. κ flags peakedness; the token identities carry the semantics. Read both columns.)*

## The Two That Came Back as Noise — the Label Audit

Features `22004` and `23893` project to junk: whitespace, punctuation, non-Latin fragments. Under Part I's framing there were three candidate explanations: the features are real but non-verbal; the latents are dead; or the labels simply don't belong to this checkpoint's dictionary.

The third one is checkable for free, and it wins. Neuronpedia hosts an *independent* autointerp of the **same public Goodfire checkpoint** ([`llama3.3-70b-it-gf / 50-resid-post-gf`](https://www.neuronpedia.org/llama3.3-70b-it-gf)). Its labels:

| feature | notebook label | Neuronpedia autointerp (same checkpoint) | agree? |
|---|---|---|---|
| 30686 | tactical deception | "deception and trickery" | ✓ |
| 41533 | acts of deception | "lie, lies, lying" | ✓ |
| 30032 | pretending/feigning | "pretends to be something" | ✓ |
| 58667 | cover stories | "innocent or convincing" | ~✓ |
| 22004 | persona roleplay | "role followed by 'I would' or 'you are'" | **✗ (syntactic template)** |
| 23893 | concealing artificial nature | "mention specific topics" | **✗ (unrelated)** |

Exactly the two features the lens refused to name are the two whose notebook labels fail against an independent labeling of the same dictionary. (Neither is dead: both fire on ~0.1% of Neuronpedia's corpus tokens.) This is the namespace-transfer failure Part I explicitly flagged as unverified risk — now observed, on 2 of 6 IDs, detected *first* by an instrument that has no access to any label at all.

That is the practical takeaway of this note: **a lens readout is a label auditor.** Before steering a feature you believe means something, project its decoder column through the lens. If the model doesn't name it, treat the label as unconfirmed for your checkpoint.

## What This Does and Doesn't Establish

- It **does** establish that four of the six coordinates carry verbalizable deception-adjacent content in this checkpoint, by an instrument independent of every label pipeline involved.
- It does **not** establish causal claims — that steering these directions changes deception behavior. Dose-response steering, in both directions with geometry falsification controls, is [Part III](../sae-workspace-steering/).
- It does **not** settle what 22004/23893 *are* — only that their notebook labels don't transfer. A κ-stratified survey of the full dictionary (what fraction of *all* features are lens-legible?) is running as follow-up and will report separately.

## Reproduce it

The κ table regenerates from the committed receipt without any GPU:

```bash
git clone https://github.com/praxagent/research-and-replications
cd projects/jacobian-lens-and-identifiability/experiments/sae_x_jspace
python - <<'PY'
import json
d = json.load(open("saex_70b.json"))
for f, s in d["stage0"].items():
    print(f, round(s["band_kappa_max"],1), s["top10_at_peak"][:6])
PY
```

To recompute from raw open weights (SAE decoder + lens + unembedding; no model forward pass needed), `sae_x_jspace.py --stages 0` in the same folder does it in minutes on one GPU. Full experiment design, pre-registration, and the ledger: [`experiments/sae_x_jspace/`](https://github.com/praxagent/research-and-replications/tree/main/projects/jacobian-lens-and-identifiability/experiments/sae_x_jspace).

## References

- Anthropic (2026). *Verbalizable Representations Form a Global Workspace in Language Models*. Transformer Circuits Thread. https://transformer-circuits.pub/2026/workspace/ — the Jacobian lens, the J-space, and the lens-kurtosis (κ) statistic used here (their Figs. 74–75 project SAE decoder directions through the lens; Fig. 90 is a steering example tied to lens readouts). Invited commentaries (Dehaene & Naccache; Butlin et al./Eleos; Nanda): [PDF](https://www-cdn.anthropic.com/files/4zrzovbb/website/cc4be2488d65e54a6ed06492f8968398ddc18ebe.pdf) — Nanda's frames SAEs and probes as alternative access routes to the same workspace.
- Anthropic (2026). *jacobian-lens* (code, Apache-2.0). https://github.com/anthropics/jacobian-lens
- Neuronpedia (2026). *Jacobian lens collection* (pre-fitted lenses, incl. `llama3.3-70b-it`). https://huggingface.co/neuronpedia/jacobian-lens — and the Goodfire-SAE feature dashboards for this checkpoint: https://www.neuronpedia.org/llama3.3-70b-it-gf (independent autointerp labels used in the audit table).
- Goodfire (2025). *Llama-3.3-70B-Instruct-SAE-l50* (model card). https://huggingface.co/Goodfire/Llama-3.3-70B-Instruct-SAE-l50
- AE Studio. *Deception Features & Subjective Consciousness Study* (public Steering API notebook — source of the six IDs and their original labels). https://github.com/agencyenterprise/steering-api-examples/tree/main/deception-features
- Paulo, G., & Belrose, N. (2025). *Sparse Autoencoders Trained on the Same Data Learn Different Features*. arXiv:2501.16615 — why integer IDs need not transfer across dictionaries; the failure mode observed here for 2 of 6 IDs.
- Leask, P., et al. (2025). *Sparse Autoencoders Do Not Find Canonical Units of Analysis*. arXiv:2502.04878.
- Chanin, D., et al. (2024). *A Is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders*. arXiv:2409.14507 — a same-checkpoint reason a label can mislead even without namespace failure.
- Bills, S., et al. (2023). *Language Models Can Explain Neurons in Language Models*. OpenAI. https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html — the autointerp lineage behind labels like Neuronpedia's; such labels are themselves fallible, which is why triangulation across instruments matters.
- Jones, T. (2026). *Feature IDs Are Not Explanations* (Part I). https://praxagent.ai/blog/posts/how-to-read-an-sae-feature-id/
- This experiment's design, pre-registration, receipts: https://github.com/praxagent/research-and-replications/tree/main/projects/jacobian-lens-and-identifiability/experiments/sae_x_jspace

*Built on Anthropic's Apache-2.0 [jlens](https://github.com/anthropics/jacobian-lens), Neuronpedia's open lens collection and feature dashboards, and Goodfire's open SAE weights. Part I: [Feature IDs Are Not Explanations](../how-to-read-an-sae-feature-id/).*
