# Results — structural sparsity & identifiability

_Ledger of actual runs. Numbers here were produced by running `train.py`; none are
copied from the paper. Honesty rule: if it wasn't run, it's not a result._

## Linear mode (CPU, this box — free)

Command: `python train.py --mode linear --n {N} --samples 20000`
Metric: MCC (mean |corr| after optimal source matching; higher = better recovery).
Estimator: minimal-support over the orthogonal group (min_Q ‖cov^½ Q‖₁, 8 restarts).

| n | mixing | L1 | MCC | note |
|---|--------|----|-----|------|
| 5 | sparse | 0.05 | 0.830 | ← the target condition |
| 5 | sparse | 0.00 | 0.723 | ablation: no sparsity bias |
| 5 | dense  | 0.05 | 0.811 | |
| 5 | dense  | 0.00 | 0.664 | |
| 8 | sparse | 0.05 | 0.741 | ← the target condition |
| 8 | sparse | 0.00 | 0.648 | ablation |
| 8 | dense  | 0.05 | 0.645 | |
| 8 | dense  | 0.00 | 0.627 | |

**Honest read.** The *direction* reproduces the theorem: the L1 (minimal-support) bias
**improves recovery only when the true mixing is structurally sparse** (n=8: 0.648→0.741),
and gives essentially nothing on a dense mixing (0.627→0.645) — sparsity is identifying
*only when there is structure to find*, exactly Prop. 1 / Thm. 2. **But it is NOT a clean
reproduction:** the target condition tops out at ~0.74–0.83, not the ~0.95+ that clean
identification would show, and at n=5 the dense+L1 case is nearly as high as sparse+L1.

**Why (leading hypotheses, for the next iteration):**
1. The orthogonal minimal-support search is non-convex; 8 restarts likely undershoots
   the global minimum at n≥5. → more restarts / basin-hopping / an annealed L0 surrogate.
2. L1 is a loose surrogate for the L0 support the theorem uses. → try a smoothed-L0 or
   reweighted-L1 (log-sum) penalty.
3. MCC's optimal matching is generous, inflating the "chance" floor (n=8 dense+L1-off is
   already 0.63 vs 1/√8≈0.35), which compresses the visible gap. → report a permutation
   null and Amari distance alongside MCC.

**Status: replication-in-progress, directionally confirmed.** This is a real,
CPU-reproducible negative-ish result to iterate on — not a claimed success.

## Nonlinear mode (coupling flow + Jacobian-L1)

From-scratch implementation of the paper's *mechanism* (flow MLE under a factorized
non-Gaussian prior + L1 on the input→output Jacobian via `torch.func.jacrev`+`vmap`), not
the authors' GIN code.

**Executed (CPU smoke, n=3, 3k samples, 15 epochs):** the pipeline **runs end-to-end**
(the per-sample Jacobian penalty works), but MCC is at chance (~0.60, chance≈0.577) in
all conditions — **as expected for a 15-epoch undertrained flow**, so it carries no
signal yet. This confirms the code path only; it is not a result. Real training (hundreds
of epochs, larger n) needs GPU — the first genuine use of RunPod for this project.
**(Update this section only with runs you actually executed.)**

## Next steps

- Sharpen the linear estimator (restarts/L0-surrogate) to test whether clean
  identification is reachable — all CPU, no GPU spend.
- Add Amari distance + a permutation null so the sparse-vs-dense gap is measured against
  a real floor.
- Only after the linear result is crisp, invest GPU time in the nonlinear flow at scale.
