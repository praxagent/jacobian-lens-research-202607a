# beyond_structural_sparsity

**Targets:** Zheng & Zhang, *Generalizing Nonlinear ICA Beyond Structural Sparsity*
(NeurIPS 2023, arXiv:2311.00866).

**Claim under test:** identifiability survives when the 2022 assumptions are relaxed —
**undercomplete** mixing (`m > n` observations), **dependent** sources (identifiable up
to a *subspace* transform), and **partial** sparsity (only a subset of sources satisfies
it).

**Compute:** CPU (synthetic, small).

## Plan

This reuses the flagship (`../nonlinear_ica_sparsity/train.py`) infrastructure with new
data regimes, so it's mostly a data + evaluation extension, not a new model:

1. **Undercomplete** — add a rectangular mixing `m > n` to `common/synth.py` and confirm
   (their Fig. 5) that the fraction of supports satisfying structural sparsity rises
   toward 1 as `m/n` grows; measure MCC vs `m/n`.
2. **Dependent sources** — generate sources in correlated blocks (irreducible independent
   subspaces); evaluate **subspace-MCC** (best matching of *subspaces*, not components)
   for the dependent block and component-MCC for the independent one.
3. **Partial sparsity** — mix a structurally-sparse subset with a dense subset; expect
   component identifiability for the former, subspace for the latter.

**Metric:** MCC + a subspace-MCC variant (to be added to `common/metrics.py`).

**Status:** planned. No runs yet — this README is the design, not a result.
