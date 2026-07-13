"""Replication — structural sparsity of the mixing Jacobian yields identifiability.

Target: Zheng, Ng & Zhang, "On the Identifiability of Nonlinear ICA: Sparsity
and Beyond" (NeurIPS 2022), arXiv:2206.07751.

Two configs, sharing the metric (MCC) and the central idea (an L1 penalty on the
unmixing's Jacobian selects the structurally-sparse solution, breaking the gauge
freedom that makes ICA unidentifiable):

  --mode linear     Linear-Gaussian corollary (Prop. 1 / Thm. 2). Learn an
                    unmixing W by decorrelation + L1(W). CPU-cheap, runs in
                    seconds; the clean core result. Gaussian sources, so
                    identifiability MUST come from structural sparsity.

  --mode nonlinear  Full nonlinear case. A RealNVP-style coupling flow unmixer
                    trained by maximum likelihood under a factorized non-Gaussian
                    prior, plus an L1 penalty on the flow's input->output
                    Jacobian ("structural sparsity" bias). Heavier; CPU-runnable
                    at small n, GPU for scale.

Ablation baked in: each mode runs {sparse, dense} mixing x {L1 on, L1 off} and
prints an MCC table. The identifiability claim predicts high MCC ONLY for
sparse-mixing + L1-on.

Honesty: the linear mode is a faithful, self-contained reproduction of the 2022
linear corollary. The nonlinear mode implements the paper's mechanism (flow MLE
+ Jacobian-L1) but is a from-scratch implementation, not the authors' code —
treat its numbers as a replication attempt to be validated, not the paper's.

Run (CPU):
    uv run python train.py --mode linear
    uv run python train.py --mode nonlinear --n 3 --samples 5000 --epochs 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# make `common` importable whether run from repo root or the experiment dir
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import synth  # noqa: E402
from common.metrics import mcc  # noqa: E402


# --------------------------------------------------------------------------- #
# Linear mode — the clean, CPU-cheap core result
# --------------------------------------------------------------------------- #
def run_linear(x: np.ndarray, s: np.ndarray, l1: float, device: str,
               steps: int = 3000, lr: float = 2e-2) -> float:
    """Minimal-support unmixing over the orthogonal group (Thm. 2).

    Exactly whiten x (ZCA): x_w = x @ Wz, cov(x_w)=I. For Gaussian sources the
    only remaining gauge freedom is an orthogonal rotation Q — every Q keeps the
    output white, so decorrelation cannot identify anything. The identifying
    step is choosing the Q whose implied unmixing U = Wz @ Q is SPARSEST
    (minimal L1 = surrogate for minimal support). When the true mixing is
    structurally sparse, that sparse rotation recovers the sources (MCC->1);
    when it is dense, no sparse rotation exists and MCC stays near chance.

    Q is parametrized as matrix_exp(P - P^T) so it stays exactly orthogonal and
    whitening is preserved by construction. l1=0 disables the identifying term
    (ablation): the rotation is then unconstrained -> chance-level MCC.
    """
    n = x.shape[1]
    X = torch.tensor(x, dtype=torch.float32, device=device)
    X = X - X.mean(0, keepdim=True)
    cov = (X.T @ X) / X.shape[0]
    evals, evecs = torch.linalg.eigh(cov)
    Csqrt = evecs @ torch.diag(torch.sqrt(evals.clamp_min(1e-8))) @ evecs.T  # cov^{1/2}

    def recover(Q):
        A_hat = Csqrt @ Q                       # candidate mixing, A_hat A_hat^T = cov
        S_hat = X @ torch.linalg.inv(A_hat).T   # s = A^{-1} x  (row-vector form)
        return A_hat, S_hat

    if l1 <= 0:
        # Ablation: no identifying signal -> an arbitrary rotation (chance gauge).
        with torch.no_grad():
            P = torch.randn(n, n, device=device)
            _, S_hat = recover(torch.matrix_exp(P - P.T))
        return mcc(s, S_hat.cpu().numpy())

    # Minimal-support estimator (Prop. 1 / Thm. 2): min_Q ||cov^{1/2} Q||_1.
    # The orthogonal L1 landscape is non-convex, so use random restarts and keep
    # the lowest-L1 rotation (the actual global-minimum-support estimate).
    best_l1_val, best_P = float("inf"), None
    for restart in range(8):
        P = torch.nn.Parameter(0.3 * torch.randn(n, n, device=device))
        opt = torch.optim.Adam([P], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            A_hat = Csqrt @ torch.matrix_exp(P - P.T)
            (A_hat.abs().sum()).backward()
            opt.step()
        with torch.no_grad():
            val = (Csqrt @ torch.matrix_exp(P - P.T)).abs().sum().item()
        if val < best_l1_val:
            best_l1_val, best_P = val, P.detach().clone()
    with torch.no_grad():
        _, S_hat = recover(torch.matrix_exp(best_P - best_P.T))
    return mcc(s, S_hat.cpu().numpy())


# --------------------------------------------------------------------------- #
# Nonlinear mode — coupling-flow unmixer + Jacobian-L1
# --------------------------------------------------------------------------- #
class AffineCoupling(nn.Module):
    """RealNVP-style affine coupling; splits dims by a fixed binary mask."""

    def __init__(self, n: int, mask: torch.Tensor, hidden: int = 64):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(n, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, 2 * n),
        )

    def forward(self, x):
        xm = x * self.mask
        st = self.net(xm)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * (1 - self.mask)  # scale acts on the complement
        t = t * (1 - self.mask)
        y = xm + (1 - self.mask) * (x * torch.exp(s) + t)
        logdet = s.sum(-1)
        return y, logdet


class Flow(nn.Module):
    """A small stack of alternating-mask affine couplings (invertible)."""

    def __init__(self, n: int, n_couplings: int = 6, hidden: int = 64):
        super().__init__()
        masks = []
        for i in range(n_couplings):
            m = torch.zeros(n)
            m[i % 2 :: 2] = 1.0  # alternate which half is passed through
            masks.append(m)
        self.layers = nn.ModuleList(AffineCoupling(n, m, hidden) for m in masks)

    def forward(self, x):
        logdet = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, ld = layer(z)
            logdet = logdet + ld
        return z, logdet


def laplace_log_prob(z: torch.Tensor) -> torch.Tensor:
    """Factorized standard-Laplace prior: sum_i -|z_i| - log 2."""
    return (-z.abs() - np.log(2.0)).sum(-1)


def jacobian_l1(flow: Flow, x: torch.Tensor, n_probe: int = 64) -> torch.Tensor:
    """Mean L1 of the flow's input->output Jacobian over a probe minibatch.

    Uses torch.func.jacrev + vmap for a per-sample Jacobian; averaged. This is
    the "structural sparsity" inductive bias (penalize dense dependencies).
    """
    from torch.func import jacrev, vmap
    probe = x[:n_probe]

    def single(inp):
        z, _ = flow(inp.unsqueeze(0))
        return z.squeeze(0)

    J = vmap(jacrev(single))(probe)  # (n_probe, n, n)
    return J.abs().mean()


def run_nonlinear(x: np.ndarray, s: np.ndarray, l1: float, device: str,
                  epochs: int = 300, batch: int = 512, lr: float = 1e-3) -> float:
    n = x.shape[1]
    X = torch.tensor(x, dtype=torch.float32, device=device)
    X = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-6)
    flow = Flow(n).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    N = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch):
            idx = perm[i : i + batch]
            xb = X[idx]
            opt.zero_grad()
            z, logdet = flow(xb)
            nll = -(laplace_log_prob(z) + logdet).mean()
            loss = nll + l1 * jacobian_l1(flow, xb)
            loss.backward()
            opt.step()
    with torch.no_grad():
        z, _ = flow(X)
    return mcc(s, z.cpu().numpy())


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["linear", "nonlinear"], default="linear")
    p.add_argument("--n", type=int, default=4, help="number of sources")
    p.add_argument("--samples", type=int, default=20000)
    p.add_argument("--epochs", type=int, default=300, help="nonlinear mode only")
    p.add_argument("--l1", type=float, default=0.05, help="Jacobian/unmixing L1 weight")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu | cuda")
    args = p.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)

    src_dist = "gaussian" if args.mode == "linear" else "laplace"
    print(f"mode={args.mode}  n={args.n}  samples={args.samples}  "
          f"sources={src_dist}  device={device}")
    print(f"epochs={args.epochs}  l1={args.l1}  seed={args.seed}  "
          f"argv={' '.join(sys.argv[1:])}")
    print("(identifiability predicts high MCC ONLY for sparse mixing + L1 on)\n")

    rows = []
    for support in ("sparse", "dense"):
        for l1 in (args.l1, 0.0):
            if args.mode == "linear":
                d = synth.make_linear(args.samples, args.n, support, src_dist, args.seed)
                score = run_linear(d["x"], d["s"], l1, device)
            else:
                d = synth.make_nonlinear(args.samples, args.n, 3, support, src_dist, args.seed)
                score = run_nonlinear(d["x"], d["s"], l1, device, epochs=args.epochs)
            rows.append((support, l1, score))

    print(f"{'mixing support':16s} {'L1':>8s} {'MCC':>8s}")
    print("-" * 34)
    for support, l1, score in rows:
        tag = "  <- expect high" if (support == "sparse" and l1 > 0) else ""
        print(f"{support:16s} {l1:8.3f} {score:8.3f}{tag}")
    chance = 1.0 / np.sqrt(args.n)
    print(f"\nchance MCC ~ {chance:.3f} (1/sqrt(n))")


if __name__ == "__main__":
    main()
