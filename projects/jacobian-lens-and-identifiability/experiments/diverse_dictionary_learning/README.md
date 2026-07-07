# diverse_dictionary_learning

**Targets:** Zheng, Li, Fan, Wilson & Zhang, *Diverse Dictionary Learning* (ICLR 2026,
arXiv:2604.17568).

**Claim under test (the SAE-relevant one):** the identifiable object of a general latent
model `X = g(Z)` is the **support of the decoder Jacobian** (the "dependency structure"),
and the universal inductive bias that recovers it is **dependency sparsity** — an L1
penalty on the *Jacobian* — which **outperforms latent sparsity** (the sparse-code bias
SAEs use). The paper shows this on VAE/GAN/Diffusion backbones by disentanglement score.

**Compute:** CPU at small scale (a compact VAE on synthetic / low-res factors); GPU only
to scale to their image backbones.

## Plan

1. Small VAE on a synthetic multi-factor dataset (or dSprites-like) with known factors.
2. Three training variants, identical except the sparsity term:
   - **baseline** (no sparsity),
   - **latent-L1** (sparse codes — the SAE-style bias),
   - **dependency-L1** (L1 on the decoder Jacobian `∂g/∂z`, reusing the
     `jacobian_l1`-style `torch.func.jacrev`+`vmap` utility from the flagship).
3. **Metric:** a disentanglement score (MIG / DCI / FactorVAE-score) against the known
   factors, plus `support_iou` (already in `common/metrics.py`) of the recovered vs true
   Jacobian support. Prediction: dependency-L1 ≥ latent-L1 ≥ baseline.

Why it matters for Prax: if dependency sparsity really beats latent sparsity, an SAE lane
on the open-model backend should penalize the *Jacobian*, not just activations (see
`background.md`). This is the experiment that would justify that design choice.

**Status:** planned. `common/metrics.support_iou` is ready; the VAE + training loop is
the next CPU build. No runs yet.
