# nonlinear_ica_sparsity

**Targets:** Zheng, Ng & Zhang, *On the Identifiability of Nonlinear ICA: Sparsity and
Beyond* (NeurIPS 2022, arXiv:2206.07751).

**Claim under test:** an L1 penalty on the *support of the mixing/unmixing Jacobian*
selects the structurally-sparse solution, which breaks the gauge freedom that makes ICA
unidentifiable — so recovery (MCC) should jump **only** when the true mixing is
structurally sparse *and* the sparsity bias is on.

**Metric:** MCC (mean |corr| after optimal source matching) — the paper's own metric.

## Run (CPU, free)

```bash
uv run python train.py --mode linear     --n 8 --samples 20000     # clean core, seconds
uv run python train.py --mode nonlinear  --n 3 --samples 5000 --epochs 300   # flow, minutes
```

Each run sweeps {sparse, dense} mixing × {L1 on, off} and prints an MCC table; the
identifiability claim predicts high MCC **only** for sparse + L1-on.

- **`--mode linear`** — linear-Gaussian corollary (Prop. 1 / Thm. 2). Minimal-support
  estimator over the orthogonal group. Gaussian sources, so identifiability *must* come
  from structural sparsity. CPU-cheap; the clean core result.
- **`--mode nonlinear`** — RealNVP coupling-flow unmixer + factorized non-Gaussian prior
  MLE + L1 on the flow's input→output Jacobian. CPU-runnable small; GPU (`--device cuda`)
  for scale.

## Status

See [`results.md`](results.md). Linear mode **directionally reproduces** (L1 helps only
when structure exists) but is not yet crisp (~0.74–0.83, not ~0.95); next step is
sharpening the non-convex orthogonal search (restarts / smoothed-L0) — all on CPU. The
nonlinear path executes end-to-end; real training is the first GPU use.

**Honesty:** the linear mode is a faithful self-contained reproduction of the 2022 linear
corollary; the nonlinear mode implements the paper's *mechanism*, not the authors' GIN
code — treat its numbers as a replication attempt to validate, never as the paper's.
