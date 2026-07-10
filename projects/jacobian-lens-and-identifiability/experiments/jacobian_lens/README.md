# jacobian_lens

**Targets:** Gurnee, Lindsey et al. (Anthropic), *Verbalizable Representations Form a
Global Workspace* (2026) — the **J-lens / J-space**; the **Dehaene & Naccache** (2026)
tests; and the open cross-model **CKA explorer** by Elie Bakouch
([eliebak.com/viz/jspace-open](https://eliebak.com/viz/jspace-open)).

## Build ON the reference, don't reimplement it

Anthropic released the companion code — **[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens)**,
**Apache 2.0**, works on **open-weights HuggingFace decoders** (`jlens.from_hf`, demos on
Qwen), ~1700 LOC. It gives us `fit()`, `lens.apply(model, text, positions)`, the exact
estimator (cotangents at all target positions, sum over `p'≥p`, mean over source `p`,
skip first 16 attention-sink positions), `.merge()` for sharded fitting, and — crucially
— **Apache-2.0 prompt sets** for every property (`data/experiments/`: verbal-report,
selectivity, capacity, **ignition**, **dual-task**, top-down-summoning,
flexible-generalization, probe-swap, …) plus lens-quality evals. They also flag it as
*"not maintained, not optimized, not accepting contributions."*

Pre-fitted lenses exist on **Neuronpedia** (`huggingface.co/neuronpedia/jacobian-lens`,
trained on 1000 wikitext prompts) — so we can **apply** a lens with **no fitting**.

**So we depend on `jlens`** (`uv pip install jlens` / from git) and lead where they don't.

## Two sub-experiments

### A. CKA cross-model geometry — partial replication of eliebak (CPU-feasible)

Bakouch's explorer computes **CKA between layers' J-lens token geometries**, within and
across 38 open models, over 4096 shared probe tokens — revealing a
sensory→**workspace**→motor band and cross-model matched-depth alignment. Because it uses
**pre-fitted** lenses, the heavy part (backward-pass fitting) is already done; computing
J-lens vectors for the probe tokens and the CKA matrix is cheap matrix algebra.

**This is our first, cost-free target.** `common/cka.py` (`linear_cka`, `cka_matrix`) is
built and CPU-validated. Plan: pull a pre-fitted lens (Neuronpedia), compute the J-lens
token-direction matrix per layer for the shared probe tokens, build the layer×layer CKA
matrix, and reproduce (a) the diagonal workspace-band block and (b) a cross-model
matched-depth CKA for 2–3 small open models. **CPU-feasible**; larger models on GPU only
if memory forces it.

### B. J-space readout + causal tests on open models (GPU for fitting)

Fit/apply `jlens` on a small open model (Qwen-small / Pythia / GPT-2), then run the
Apache-2.0 experiment sets as evals: verbal-report, selectivity, capacity, ignition,
dual-task, plus the **Dehaene/Naccache** additions not in their data — **local-global**,
**trace-conditioning**, **inclusion/exclusion**. Also the **honesty probe** (fabricated
inputs → do "fake"/"fraud"/"injection" tokens appear in J-space?). Fitting needs GPU
(the first approved, terminated-immediately RunPod spend); applying a pre-fitted lens is
cheaper.

## How we do better (honestly — we will NOT out-scale Anthropic)

We can't match frontier-Claude scale or internal access. We lead where openness and rigor
win:

1. **Identifiability grounding** — the project's thesis: connect the J-lens readout to
   nonlinear-ICA **identifiability** theory (`background.md`), i.e. *when is a readout a
   real recovered factor vs a seed-dependent artifact?* Anthropic's paper doesn't cite
   this; it's our unique contribution and a real question (is J-space stable across lens
   seeds / fitting corpora? — directly testable with `.merge()`-style reruns + CKA).
2. **The neuroscience test battery as reproducible evals** — turn Dehaene & Naccache's
   proposed tests (local-global, trace-conditioning, inclusion/exclusion, ignition) into
   a runnable open suite; they proposed them, nobody has packaged them openly.
3. **Cross-open-model robustness** — extend the CKA study with a *replication lens*: does
   the workspace band appear across families (Qwen/Gemma/Pythia/OLMo/**Apertus**), and
   how stable is it? A generalization result, openly reproducible.
4. **Maintained + optimized** — theirs is explicitly unmaintained/unoptimized; a faster,
   tested, documented fitting path is a concrete improvement.
5. **Safety-harness integration** — wire the honesty/injection signal into Prax's
   open-model path (their code is a research lib, not a deployed guard).

**Status: RUN (complete).** 35-of-38-lens uniform sweep (own-vocab + shared-vocab
probes + null controls + precision A/B) — all numbers and the tokenizer-confound verdict
in [`results.md`](results.md); figures `emergence_curve*.png`. The design text below is
kept as written for the record.
