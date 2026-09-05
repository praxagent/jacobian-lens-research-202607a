# Jacobian lens & identifiability

Reproducing the thread that runs from **nonlinear-ICA identifiability** (when is a
learned representation the *real* one?) to Anthropic's frontier-scale **J-lens / J-space**
"global workspace," endorsed as a GNW landmark by **Dehaene & Naccache**.

Start with **[background.md](background.md)** — the literature, the through-line, the
honesty flags, and why any of this matters for Prax. See also
**[related-work.md](related-work.md)** — the brain–LLM alignment / "workspace layers"
lineage (via Jean-Rémi King's thread) that predates and contextualizes J-space.

## Sources reproduced / assessed

| # | Source | Role |
|---|--------|------|
| 1 | Zheng, Ng, Zhang — *Identifiability of Nonlinear ICA: Sparsity and Beyond* (NeurIPS 2022) | structural sparsity of the mixing **Jacobian** ⇒ identifiability |
| 2 | Zheng, Zhang — *Generalizing Nonlinear ICA Beyond Structural Sparsity* (NeurIPS 2023) | undercomplete / dependent / partial-sparsity regimes |
| 3 | Zheng, Li, Fan, Wilson, Zhang — *Diverse Dictionary Learning* (ICLR 2026) | SAEs = linear special case; **dependency sparsity > latent sparsity** |
| 4 | Gurnee, Lindsey et al. (Anthropic) — *Verbalizable Representations Form a Global Workspace* (2026) | the **J-lens / J-space** at-scale result |
| 5 | Dehaene & Naccache — *Workspace commentary on Gurnee/Lindsey* (2026) | GNW originators' validation + proposed tests |

## Replication status

Honest state of each experiment, one row per directory. **CPU** = runs free on the dev box; **GPU** =
needed RunPod. Verdicts are the ledgers' words; every row links its `results.md`.

| Experiment | Question | Compute | Verdict (see ledger) |
|---|---|---|---|
| [`nonlinear_ica_sparsity`](experiments/nonlinear_ica_sparsity/) | structural sparsity of the mixing Jacobian implies identifiability (#1) | CPU | linear mode directionally reproduces; nonlinear path executes; not crisp |
| [`beyond_structural_sparsity`](experiments/beyond_structural_sparsity/) | undercomplete / dependent sources (#2) | CPU | planned, not run |
| [`diverse_dictionary_learning`](experiments/diverse_dictionary_learning/) | dependency vs latent sparsity (#3) | CPU/GPU | planned, not run |
| [`jacobian_lens`](experiments/jacobian_lens/) | CKA sweep over the public lens collection; shared-vocab re-sweep; precision A/B | CPU | run; shared probe overturns the own-vocabulary family split |
| [`behavioral`](experiments/behavioral/) | do lens band statistics predict behaviour (23 models) | GPU | run; correlations survive the shared probe (+0.53) but band is neither necessary nor sufficient |
| [`fit_our_own`](experiments/fit_our_own/) | fit our own lenses; the 397B frontier fit; consumer-path integrity gate | GPU | run; 397B lens released (mid_sep +0.343, reproduced to 2e-8); 24-prompt budget now caveated |
| [`lens_demo`](experiments/lens_demo/) | pre-registered readout audit and steering on the 397B | GPU | run; 32/50 steered swaps flip vs 0/50 controls |
| [`jspace_atlas`](experiments/jspace_atlas/) | 36-lens atlas, cross-model matrix, Tier-1 arms, readout decomposition, identifiability, Toeplitz surrogate | CPU (+ Tier-1 GPU) | run; both cross-model predictions failed; 20/35 lenses have identified boundaries; 80% of the 397B statistic is distance decay |
| [`readout_decomposition`](experiments/readout_decomposition/) | what the invariant gemma-2 readout direction encodes | CPU | run; the model's own prior modulated by writing system, 37% unexplained |
| [`sae_x_jspace`](experiments/sae_x_jspace/) | SAE features read through the J-space | GPU | run; bidirectional on 6 deception features |
| [`block_patching`](experiments/block_patching/) | do band boundaries act as barriers to activation transfer (v1, v2, v2.1) | GPU | run; nulls, one position artifact, one non-replication; design not identified |
| [`geometry_causality`](experiments/geometry_causality/) | equal-norm perturbation: lens direction vs random vs comparator | GPU | run; lens beats random 5x to 11x (float32, prompt intervals); dose caveat |
| [`bands_vs_computation`](experiments/bands_vs_computation/) | Test B (lens vs activation boundaries, 12 models), Test C (corpus instability), identified-only Test B | GPU + CPU | run; readout property in every cell (shared probe); Test C moot after the corpus correction |
| [`corpus_dependence`](experiments/corpus_dependence/) | does the map depend on the fitting corpus; fit budget; 8B extension | GPU | run; map moves 84x to 888x the seed null, boundaries barely; converged from 200 prompts; 25-prompt caveat triggered; 8B pending |
| [`ignition_depth`](experiments/ignition_depth/) | does ignition depth track the late boundary | GPU | run; no verdict (2 usable models; prompt set too hard for half the set) |

### Candidate replications from the Dehaene & Naccache commentary

Cognitive-neuroscience tests they propose for the J-space — the harness now exists
(`experiments/behavioral/`) and the decisive **ignition** test is **DONE** (~24 open
models + the 397B; `experiments/behavioral/results.md`). Still tracked:

- **local-global** (global vs local sequence prediction), **trace conditioning** (bridge
  a temporal gap; ablate J-space, expect long-gap completion to break), **inclusion/
  exclusion** (Stroop-like conscious control), **error-monitoring** (failure tokens in
  J-space).

## Tools

- [`tools/jlens_atlas/`](tools/jlens_atlas/): `python -m jlens_atlas --neuronpedia <slug> --model <hf_id>`
  reproduces the atlas recipe on any Jacobian lens (layer x layer linear CKA of the readout geometry on
  the shared or own probe, fixed-thirds and fitted band statistics, boundary identifiability, the
  random-transport null, receipts). `selftest.py` reproduces the cached gpt2-small maps byte for byte.

## Corrections (2026-09-05)

A fresh review found two analysis defects, both corrected with the superseded numbers kept regenerable:
the corpus-dependence and fit-budget experiments had scored maps with a cosine of self-covariances
that is not CKA (see the correction block at the top of `experiments/corpus_dependence/results.md`),
and the boundary-agreement and ignition tests had used own-vocabulary lens boundaries against
shared-probe pre-registrations (`experiments/bands_vs_computation/results.md`,
`experiments/ignition_depth/results.md`). Verdicts survived; several headline numbers changed.

## Run it

```bash
# from repo root, CPU venv already set up (torch 2.4.1+cpu, numpy, scipy)
cd projects/jacobian-lens-and-identifiability
uv run python experiments/nonlinear_ica_sparsity/train.py --mode linear --n 8
```
