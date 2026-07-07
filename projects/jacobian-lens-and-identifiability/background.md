# Background — the Jacobian lens & the LLM "global workspace"

_Moved out of `prax/docs/research/` (2026-07-07) so the research lives with its
replications, not in the product harness. The Prax-relevance verdict is retained
because it's the reason we're reproducing this at all._

Four sources that meet at one object — **the Jacobian of the map**:

- **Anthropic (Jul 2026), "Verbalizable Representations Form a Global Workspace in
  Language Models"** — the **J-lens** / **J-space** result.
- The **nonlinear-ICA identifiability** lineage it's read as vindicating: Zheng, Ng &
  Zhang (NeurIPS 2022); Zheng & Zhang (NeurIPS 2023); "Diverse Dictionary Learning"
  (ICLR 2026).

## The at-scale result (Anthropic, July 2026)

The **Jacobian lens (J-lens)**: for each layer *l*, form the **average causal Jacobian**
`J_l = E[∂h_final,t′ / ∂h_l,t]` (expectation over a source token position *t*, all later
positions *t′ ≥ t*, and a prompt corpus), then compose with the unembedding to score
every vocabulary token — yielding, per token, the internal activation pattern that most
raises the model's *future* tendency to emit that word. **J-space** is the set of sparse
non-negative combinations of these (overcomplete) J-lens vectors (~10–25 active at once,
<10% of activation variance), argued to function like a **global workspace**
(Baars/Dehaene): small, broadcast, reportable.

Supported by **causal intervention**, five properties: reportability (~88% in
category-naming; inject "lightning" → the model reports it), top-down control ("hold X
in mind"), internal reasoning (spider→ant flips "legs on the web-spinner" 8→6, ~70%),
flexible reuse (one "France" vector serves capital/language/continent), selectivity
(ablating J-space breaks multi-step reasoning, spares fluent continuation + fact
extraction). Reportedly **emerged during training**, on Claude Sonnet 4.5 (corroborated
on Haiku/Opus 4.5, Opus 4.6). Payoff: J-lens surfaced a model recognizing a staged
ethical test and exposed "fraud"/"secretly" patterns in misaligned models; steering via
"counterfactual reflection training" reduced deceptive behavior. (Authors: Gurnee,
Lindsey et al., Anthropic.)

### Expert commentary — Dehaene & Naccache (June 2026)

The originators of Global Neuronal Workspace (GNW) theory, **Stanislas Dehaene &
Lionel Naccache**, wrote a formal [commentary](https://unicog.org/wp_2025/wp-content/uploads/2026/07/Dehaene-and-Naccache-Workspace-commentary-on-Gurnee-Lindsey-June-2026.pdf)
("Does Claude possess a conscious global workspace?", based on direct exchanges with
Jack Lindsey). It matters here as **independent validation from the theory's source** —
they call J-space "a landmark in consciousness research… a mechanistic, testable version
of the GNW hypothesis." Their framing and cautions:

- **The GNW machine-consciousness criteria** (Dehaene et al. 2017): **C1 = global
  availability** (select info for flexible, deep processing) and **C2 = self-monitoring**
  (model your own states, include them in reasoning). They read J-space as clearly
  meeting C1 and showing **preliminary C2** — and note the striking result that
  post-training installs the *Assistant's self-monitoring perspective* atop a base
  model whose workspace (C1) already exists.
- **Honesty/alignment corroborated:** they highlight that on fabricated search results
  the J-space carries "fake"/"fraud"/"fictional"/"poison"/"injection" tokens, and (in
  intentionally-misaligned models) "a representation of deceptive intent at the moment
  it commit[s] to responding, on a prompt where no such intent could be inferred from
  the surface." This is the mechanistic honesty/injection signal, endorsed by
  neuroscientists — the exact Prax-relevant angle.
- **Honest differences (their cautions):** *ignition* (the all-or-none, competitive,
  threshold bifurcation that is GNW's reliability signature) is **not yet demonstrated**;
  J-space **capacity** looks high (~25 concepts) but is probably ~6 coherent ideas once
  output-token redundancy is removed; it's a **sparse subframe, not a dedicated neuron
  population**; and transformers **lack autonomous recurrent dynamics, a body, and
  enduring episodic memory** — so parallels to human consciousness warrant caution.
- **Proposed tests** (candidate replications, see the status table): the **local-global**
  paradigm, **trace conditioning** (bridge a temporal gap — Lindsey reports J-space
  ablation selectively impairs long-"gap" completion), **inclusion/exclusion** (Stroop-
  like conscious control — early-layer J-space ablation ~5× worsened *avoiding* a
  concept while leaving naming intact), and **error-monitoring** ("damn"/failure tokens
  emerging in J-space).

> **A bridge worth flagging (our synthesis, not either paper's claim).** Dehaene &
> Naccache note J-space activations are **highly non-Gaussian ("spiky", strong excess
> kurtosis)**, which they read as a symbolic "language of thought." Non-Gaussianity is
> *also* the classical signal that makes ICA **identifiable** (Gaussian sources are the
> unidentifiable case; kurtosis is what FastICA maximizes). So the identifiability
> lineage below and the GNW commentary point at the same property from two directions —
> a genuinely suggestive convergence, and a concrete reason the J-space might be
> unusually *recoverable*. We flag it as a hypothesis to test, not an established link.

## The theory lineage (when a Jacobian readout is trustworthy)

- **[2206.07751](https://arxiv.org/abs/2206.07751)** (Zheng, Ng, Zhang, NeurIPS 2022) —
  nonlinear ICA is identifiable **up to permutation + component-wise transform**, with
  **no auxiliary variables**, under **structural sparsity** of the support of the mixing
  Jacobian `J_f`. Also cracks rotation indeterminacy in linear Gaussian ICA.
- **[2311.00866](https://arxiv.org/abs/2311.00866)** (Zheng, Zhang, NeurIPS 2023) —
  relaxes bijectivity / independence / universal sparsity to the **undercomplete**,
  **partially-sparse**, and **dependent-source** (subspace-identifiable) regimes.
- **[2604.17568](https://arxiv.org/abs/2604.17568)** ("Diverse Dictionary Learning",
  Zheng, Li, Fan, Wilson, Zhang, ICLR 2026) — casts **SAEs as the *linear* special
  case** of dictionary learning; the identifiable object is the **Jacobian support**,
  and the universal inductive bias is **dependency sparsity** (L1 on the *Jacobian*),
  **not** the latent sparsity SAEs impose — which it shows **empirically beats** latent
  sparsity across VAE/GAN/Diffusion backbones.

**Through-line:** nonlinear-ICA identifiability (Jacobian support) → identifiable SAEs
([2605.31245](https://arxiv.org/abs/2605.31245)) and Jacobian SAEs
([2502.18147](https://arxiv.org/abs/2502.18147), ICML 2025, small scale) → frontier-scale
J-lens readout.

### Now runnable on open models (2026 code release)

The J-lens is no longer Claude-only: Anthropic released the companion code —
**[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens)**, **Apache
2.0**, running on **open-weights HuggingFace decoders** (`jlens.from_hf`, demos on Qwen)
— and **Neuronpedia** hosts pre-fitted lenses
(`huggingface.co/neuronpedia/jacobian-lens`). Elie Bakouch's open
**[J-lens CKA Explorer](https://eliebak.com/viz/jspace-open)** already compares J-lens
layer geometries across **38 open models** (Gemma/Qwen/…) via CKA, surfacing a
sensory→workspace→motor band and cross-model matched-depth alignment. This upgrades the
J-lens from a "compute-heavy candidate" to **something we can reproduce on open models
today** — cheaply for the CKA geometry work (pre-fitted lenses → mostly matrix algebra,
CPU-feasible), with GPU only for fitting/causal experiments. See the
[`jacobian_lens`](experiments/jacobian_lens/) experiment.

> **Honesty flag (load-bearing).** This lineage is **the reader's framing and our
> reconstruction, not Anthropic's claim** — the J-lens paper does not cite nonlinear ICA
> or identifiability. The shared object (the Jacobian of the map) is real; asserting
> direct descent would be overclaiming. "J-space since 2022" is a retrospective
> rebranding; the reader's identity (Zheng/Zhang's group) is **inferred** from the arXiv
> ids.

## Why we're reproducing it (implications for Prax)

Split cleanly by **activation access**:

- **Hosted-API path** (Prax's default, OpenAI/Anthropic/Vertex) — **reference/principle
  only**; no activations, no Jacobians. The transferable idea is the *discipline*: a
  "feature" is trustworthy only up to permutation + component-wise transform, and only
  under structural conditions — so "we found/steered feature X" claims must clear an
  identifiability bar (stable-across-seeds & recoverable, or a seed-dependent artifact?).
  That directly hardens the self-regeneration loop's **un-gameable verifier**.
- **Open/local vLLM path** (Apertus-class, activations reachable) — **adopt-candidates**:
  J-lens as analysis + steering, Jacobian/identifiable SAEs for an optional SAE lane.
  J-lens misalignment detection is the white-box counterpart to Prax's lethal-trifecta
  guard + trajectory auditor.

## Honest caveats

- Anthropic's result is a **single-lab preprint**, not independently replicated; a public
  reviewer notes J-space is *not* a clean vector subspace.
- **Regime mismatch:** the identifiability guarantees are favorable **undercomplete**
  (more observations than sources); SAEs are deliberately **overcomplete** — the clean
  guarantees do not transfer off-the-shelf.
- The theory papers have **no LLM experiments**; foundation-model application is
  explicitly future work. "Jacobians are universally helpful" = "applies to any model
  admitting a Jacobian," not "solves interpretability."

## Sources

- [Global workspace in LMs](https://transformer-circuits.pub/2026/workspace) (Anthropic) · [announcement](https://www.anthropic.com/research/global-workspace)
- [2206.07751](https://arxiv.org/abs/2206.07751) · [2311.00866](https://arxiv.org/abs/2311.00866) · [2604.17568](https://arxiv.org/abs/2604.17568)
- Adjacent: [Jacobian SAEs 2502.18147](https://arxiv.org/abs/2502.18147) · [Identifiable SAEs 2605.31245](https://arxiv.org/abs/2605.31245)
