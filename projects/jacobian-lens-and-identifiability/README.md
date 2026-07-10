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

Honest state of each experiment. **CPU** = runs free on the dev box; **GPU** = needs
RunPod (real LLM or scale).

| Experiment | Targets | Compute | Status |
|---|---|---|---|
| [`nonlinear_ica_sparsity`](experiments/nonlinear_ica_sparsity/) | #1 (linear corollary + nonlinear mechanism) | CPU | **run** — linear mode *directionally* reproduces (sparsity+L1 helps only when structure exists); not yet crisp. Nonlinear flow path executes. See its `results.md`. |
| [`beyond_structural_sparsity`](experiments/beyond_structural_sparsity/) | #2 (undercomplete / dependent sources) | CPU | planned — config variations of the flagship (undercomplete `m>n`, dependent subspaces) |
| [`diverse_dictionary_learning`](experiments/diverse_dictionary_learning/) | #3 (dependency vs latent sparsity) | CPU (small) → GPU (scale) | planned — decoder-Jacobian-L1 vs latent-L1 on a small VAE; the SAE-relevant one |
| [`jacobian_lens`](experiments/jacobian_lens/) | #4/#5 (J-lens readout) + eliebak CKA explorer | **CPU** (CKA via pre-fitted lenses) / **GPU** (fitting + causal) | **run** — 35-of-38-lens uniform sweep + nulls + shared-vocab (tokenizer-confound) re-sweep + precision A/B (see its `results.md`); behavioral battery (swap + ignition, 23 models) in [`behavioral/`](experiments/behavioral/); own-fit validation + the **397B frontier fit** (mid_sep 0.343, lens published on HF) in [`fit_our_own/`](experiments/fit_our_own/). Write-up: `blog/jspace-audit/index.md` |

### Candidate replications from the Dehaene & Naccache commentary

Cognitive-neuroscience tests they propose for the J-space — the harness now exists
(`experiments/behavioral/`) and the decisive **ignition** test is **DONE** (~24 open
models + the 397B; `experiments/behavioral/results.md`). Still tracked:

- **local-global** (global vs local sequence prediction), **trace conditioning** (bridge
  a temporal gap; ablate J-space, expect long-gap completion to break), **inclusion/
  exclusion** (Stroop-like conscious control), **error-monitoring** (failure tokens in
  J-space).

## Run it

```bash
# from repo root, CPU venv already set up (torch 2.4.1+cpu, numpy, scipy)
cd projects/jacobian-lens-and-identifiability
uv run python experiments/nonlinear_ica_sparsity/train.py --mode linear --n 8
```
