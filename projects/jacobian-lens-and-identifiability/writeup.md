# WRITEUP (living blog draft) — Where does the "global workspace" actually emerge?

_Working draft that converts to a public post. **Status: in progress** — results fill in
as the sweep + fitting complete. Every number here is from a run in this repo; nothing is
asserted without a receipt. Companion docs: [background.md](background.md),
[related-work.md](related-work.md), experiment ledgers under `experiments/*/results.md`._

---

## Working title

*"A global workspace, or a mid-sized artifact? Auditing the J-lens across 38 open models."*

## TL;DR (to finalize once data is in)

Anthropic's *Verbalizable Representations Form a Global Workspace in Language Models*
found a reportable, causally-load-bearing "workspace" in the middle layers of frontier
Claude, and framed it in the language of consciousness / Global Workspace Theory. We
replicated the core tool (the Jacobian lens) on **38 open-weight models** (70M→70B, six
families) using the publicly released lenses, and asked the question the framing invites:
**is the "workspace band" universal, and where does it emerge?** Preliminary answer:
[the band is strongly family-dependent and reduced by instruct-tuning; emergence onset
~0.2–0.8B; frontier behavior untested — full numbers below].

## Why this, and what we're *not* claiming

Two narratives dominate the discourse (see [related-work.md](related-work.md)): "a
landmark, models may be conscious" and "it's all PR." Neither is a measurement. We're not
here to dunk or to defend — we're here to **strip it to the linear algebra and run it at
scale**, credit the prior work, and report where the framing holds and where it outruns
the evidence. The core interpretability result is good (everyone, including critics,
agrees). The contestable part is the *universality* and the *framing*.

## Method (reproducible)

- **J-lens** (Anthropic, Apache-2.0): the average causal Jacobian `E[∂h_final/∂h_l]`
  composed with the unembedding — a linear readout of what each layer is disposed to
  make the model say. We use the **pre-fitted lenses on Neuronpedia** (no fitting needed
  for ≤70B), and (tier-1, in progress) **fit our own** on small models to validate + test
  stability.
- **Our statistic — mid-band separation.** For each model we take the J-lens token
  geometry `D_l = U · J_l` over 4096 vocab tokens, compute layer×layer linear CKA, and
  measure how much more self-similar the *middle third* of layers is than its neighbors
  (a "workspace band" shows a distinct mid-block). Full definition + code:
  `experiments/jacobian_lens/`.
- **Confound control (the null).** Replace `J_l` with scale-matched random transports;
  if the band survived that, it'd be an artifact of the shared unembedding. It doesn't —
  the null sits flat near 0 across all scales.
- **Honesty:** `mid_sep` is *our* statistic, not a standard metric; we report it as a
  CKA-block measure with the null floor alongside, and cross-check with [pending: an
  alternative metric + a probe-token bootstrap].

## Results (living — filled from `experiments/*/results.md`)

### 1. Emergence onset
Tiny models (<0.2B: pythia-70m, gpt2) show **no band** (`mid_sep` ≈ 0.015, at the null
floor). By ~0.8B a band appears in some families. [Curve + exact onset once the sweep is
complete.]

### 2. Family dependence (a headline)
At matched size the band varies **~20×** by family — e.g. Qwen3.5-4B ≈ 0.20 vs Gemma-2/3
at 2–4B ≈ 0.006–0.05. "LLMs have a global workspace" is doing heavy lifting as a
*universal* claim. [Full family table.]

### 3. Instruct-tuning reduces the band
Every Gemma base/instruct pair so far shows base > instruct (e.g. gemma-3-4b 0.046 →
-it 0.015). This runs *counter* to a naive reading of "post-training installs the
Assistant into the workspace" — needs careful interpretation (our metric is geometric
distinctness, not workspace *content*). [Full base-vs-instruct panel.]

### 4. Stability (tier-1 fitting, in progress)
Does the band survive re-fitting the lens with a different seed/corpus? [Result from the
qwen3-4b 3-seed fit — high CKA = stable/real; low = fitting-dependent.]

## Interrogation — does this support "oversold"?

[Synthesis once results are complete. Current shape: the *measurement* replicates; the
*universal-workspace framing* is not supported by the open-model evidence — the phenomenon
is family-dependent, instruct-sensitive, and (crucially) **untested at frontier scale**.]

## Limitations (up front, not buried)

- **The sub-frontier-transient risk (the big one).** Our largest point is 70B; the
  ≤70B window may not capture asymptotic behavior. We could be over-claiming from a
  small-scale transient — the exact error we're auditing. Frontier (>70B) fitting is the
  key open question (future work).
- **`mid_sep` is our own statistic** — reported with the null floor; cross-checks pending.
- **Linear readout = lower bound.** The J-lens (and our CKA on it) is linear; brain–LLM
  work (TRIBE) shows linear maps *underestimate* structure. Our band is a lower bound.
- **One lens per model** for the pre-fitted set → the seed-stability tier-1 fit addresses
  this directly.

## Credit

Anthropic (strong core result + open code); the GNW originators **Dehaene & Naccache**
(sympathetic expert commentary); **Jean-Rémi King** and the brain–LLM alignment community
(the "middle layers are special" prior art); the **nonlinear-ICA identifiability** lineage
(Zheng, Zhang et al.) that frames when a Jacobian readout is trustworthy.

## Open data + code

Everything (code, per-model CSVs, figures) is in this repo:
[github.com/praxagent/research-and-replications](https://github.com/praxagent/research-and-replications).

## Future work

Frontier-scale lens fitting (>70B open: Llama-405B, DeepSeek-671B, Qwen-235B) to settle
the transient question; the **brain-alignment bridge** (does our band coincide with peak
brain-alignment across models?); base-vs-instruct as its own study.
