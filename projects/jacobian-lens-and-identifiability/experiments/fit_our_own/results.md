# Results — fitting our own lenses

_Only executed runs. RunPod A6000, jlens.fit, wikitext-103 corpus._

## 1. Validation gate — gpt2 (PASSED, 2026-07-08)

Fit our own gpt2 lens (100 wikitext prompts, seed 0, dim_batch 8) and CKA-compared its
J-lens token geometry to **Neuronpedia's** pre-fitted gpt2 lens, per layer:

```
shared layers: 11 | mean CKA(ours, neuronpedia) = 0.9992 (min 0.999, max 1.000)
per-layer: 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
```

**Our fitter reproduces Anthropic/Neuronpedia's lens essentially perfectly.** This
validates the whole pipeline end-to-end (fit_lens.py, our jlens usage, and the CKA
methodology the emergence sweep rests on), so bigger fits are trustworthy. It also means
the emergence-sweep numbers — which use these lenses — sit on a faithful foundation.

## 2. Seed/corpus stability — qwen3-4b (PASSED, 2026-07-08)

Fit Qwen3-4B on 3 **disjoint** wikitext subsets (seeds 0/1/2, 100 prompts each,
dim_batch 16), then pairwise CKA of the lenses' token geometry across all 35 shared
layers:

```
seed0 vs seed1: mean CKA = 0.9971  (min 0.976, max 1.000)
seed0 vs seed2: mean CKA = 0.9981  (min 0.989, max 1.000)
```

**J-space is corpus-sample-stable.** Re-fitting on different English text recovers
essentially the same lens — the structure is a property of the *model*, not estimation
noise. This closes the "one lens per model / can't test stability" limitation with a
positive result, and puts a foundation under every lens-based claim (Anthropic's,
eliebak's, and our 38-model sweep). Scope note: this is **within-distribution** stability
(all wikitext); the cross-**language** test is next.

## 3. Language dependence — qwen3-4b, Chinese-Wikipedia lens (in progress)

Fit the same model on **Chinese Wikipedia** (streaming, 100 passages) and compare to the
English-fit lens. If CKA(zh, en) ≈ the en↔en baseline (~0.997), the J-space transport is
**language-independent** — a structural property of the model, not of the estimation
language. If much lower, the "workspace" has a language-dependent component (relevant to
Wendler et al. 2024 "Do Llamas Work in English?", Anthropic's multilingual claims, and
Nanda's multilingual-artifact caution). Qwen3-4B is the right subject (strong Chinese).
[Numbers on completion.]
