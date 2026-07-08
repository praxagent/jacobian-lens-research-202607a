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

## 3. Language dependence — qwen3-4b, Chinese-Wikipedia lens (RESULT, 2026-07-08)

Fit the same model on **Chinese Wikipedia** (streaming, 100 passages; domain ≈ held
constant since wikitext is also encyclopedic) and CKA-compared to the English-fit lenses:

```
zh vs en-seed0: mean CKA = 0.8932  (min 0.706, max 0.986)
zh vs en-seed1: mean CKA = 0.8960  (min 0.725, max 0.986)   <- robustness: not seed-specific
en vs en baseline:        0.9971 / 0.9981  (min 0.976)
per-layer (zh vs en-seed1, layers 0->34):
0.73 0.73 0.73 0.77 0.85 0.85 0.86 0.86 0.89 0.90 0.91 0.91 0.91 0.92 0.92 0.92 0.92
0.92 0.92 0.92 0.92 0.92 0.92 0.92 0.92 0.93 0.93 0.93 0.94 0.94 0.94 0.94 0.94 0.96 0.99
```

**Finding: the J-space transport has a real, layer-structured language-dependent
component.** zh↔en agreement (0.89) sits far below the en↔en noise floor (0.997), so the
estimation language genuinely matters — but the disagreement is **concentrated in the
early layers** (0.73) and **vanishes monotonically with depth** (0.99 by the top). Read:
**early-layer verbalization dispositions are language-bound; the deeper (workspace-band)
transport geometry is close to language-general.** This refines rather than refutes
Anthropic's multilingual claim — the workspace-adjacent layers do look language-general,
with a precise statement of where that breaks (and a nuance for Wendler et al. 2024:
the English-anchoring we detect lives early, not in the deep concept space).

Caveats: one model (qwen3-4b, Chinese-strong Qwen); one language pair; wikipedia-zh vs
wikitext is same-domain but not identical text distribution. Cross-model + French
replication is cheap future work (fit_lens.py --corpus wikipedia-fr is already wired).

**Artifacts:** all five fitted lenses preserved at `artifacts/lenses/` on the dev box
(gitignored; qwen4b seeds 0/1/2 + zh + gpt2). Pod terminated after this run.
