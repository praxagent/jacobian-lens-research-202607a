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

### Multi-seed robustness (30 seeds, n=8 — the definitive linear result)

`python seed_sweep.py --seeds 30 --n 8 --samples 20000` → `seed_sweep_linear.json`. A single
seed is not a result; this reruns every cell across 30 seeds and reports mean ± 95% CI and the
load-bearing interaction (does minimal-support recovery beat an arbitrary rotation MORE when
the mixing is sparse?).

| mixing | estimator | MCC (mean, 95% CI, 30 seeds) |
|--------|-----------|------------------------------|
| sparse | minimal-support (L1) | **0.704** [0.680, 0.727] |
| sparse | random-rotation gauge (L1-off) | 0.648 [0.631, 0.664] |
| dense  | minimal-support (L1) | 0.614 [0.598, 0.630] |
| dense  | random-rotation gauge (L1-off) | 0.653 [0.638, 0.669] |
| — analytic 1/√8 floor | — | 0.354 |

**The interaction is robust; the max-cell is not.** Minimal-support beats the random-rotation
gauge by **+0.056** with structure (sparse) and by **−0.039** (a small loss) without it (dense)
— interaction **+0.095**, positive in **83%** of the 30 seeds, Wilcoxon **p=2.4e-5**. But
"sparse+L1 is the single highest cell" holds in only **67%** of seeds (the two no-structure
cells sit close), so the honest headline is the *interaction*, never the single-seed 0.741 or a
max-cell claim. Absolute recovery (0.704) is still short of clean identification (~0.95).

**Control relabel (was mislabeled).** The L1-off condition in `train.py` (`run_linear`, l1≤0
branch) is a *single random orthogonal rotation* — an arbitrary gauge baseline, NOT the
minimal-support estimator with the penalty removed. Earlier tables calling it an "ablation" were
imprecise; it is a gauge baseline. (This is also why L1-off scores ~0.65, not the 1/√8≈0.354
floor: MCC's optimal matching is a max-over-permutations statistic that lifts even a random
rotation well above the naive floor.)

**Selective-n note.** The single n=5 row above (dense+L1 0.811, moving +0.147) was an
underpowered fluctuation; at n=8 across 30 seeds the dense L1 gain is −0.039. The 30-seed n=8
result is the primary; the single-seed and small-n numbers are exploratory.

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

**Executed (CPU, n=3, 5,000 samples, 300 epochs, 2026-07-13; `.nonlinear_ica_run.log`):** the
pipeline runs end-to-end, but the result is a **null** — every cell near the 1/√3≈0.577 chance
floor, and the sparsity penalty does not separate the conditions:

| mixing | L1 | MCC |
|--------|----|-----|
| sparse | 0.05 | 0.644 |
| sparse | 0.00 | 0.654 |
| dense  | 0.05 | 0.482 |
| dense  | 0.00 | 0.499 |
| — chance 1/√3 | — | 0.577 |

The predicted contrast is absent (sparse+L1 0.644 does not even beat its own L1-off 0.654). This
is an **undertrained flow at small n**, not a reproduction; reaching the regime where the
nonlinear claim could be tested needs real training (hundreds of epochs, larger n) on GPU.
**(Update this section only with runs you actually executed.)**

## Next steps

- Sharpen the linear estimator (restarts/L0-surrogate) to test whether clean
  identification is reachable — all CPU, no GPU spend.
- Add Amari distance + a permutation null so the sparse-vs-dense gap is measured against
  a real floor.
- Only after the linear result is crisp, invest GPU time in the nonlinear flow at scale.
