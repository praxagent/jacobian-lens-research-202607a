"""Synthetic data for nonlinear-ICA identifiability experiments. NumPy only.

The identifying object across this whole line of work is the SUPPORT of the
mixing function's Jacobian. These generators let you dial that support between
DENSE (full — the classically unidentifiable case) and STRUCTURALLY SPARSE
(each source is the sole driver of some observation group — the paper's
identifiability condition), so an experiment can show recovery flipping from
chance to near-perfect as the structure appears.

Kept NumPy-only so data generation + the metric harness are CPU-validatable
before any GPU spend.
"""
from __future__ import annotations

import numpy as np


def sample_sources(n_samples: int, n_sources: int, dist: str = "laplace",
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Independent latent sources s ~ prod_i p(s_i). (N, n)."""
    rng = rng or np.random.default_rng(0)
    if dist == "laplace":
        return rng.laplace(size=(n_samples, n_sources))
    if dist == "uniform":
        return rng.uniform(-1.0, 1.0, size=(n_samples, n_sources))
    if dist == "gaussian":
        # Gaussian sources are the HARD case: independence alone leaves a
        # rotation gauge, so identifiability MUST come from structure/sparsity.
        return rng.standard_normal((n_samples, n_sources))
    raise ValueError(f"unknown source dist {dist!r}")


def structural_support(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    """An (n, n) boolean support mask for a mixing matrix / layer.

    - "dense": full support (every source touches every observation).
    - "sparse": lower-triangular — source k drives observations k..n, so the
      row-intersection singling out {k} exists (structural sparsity holds).
    - "banded": a sparser band, a stricter structural-sparsity instance.
    """
    if kind == "dense":
        return np.ones((n, n), dtype=bool)
    if kind == "sparse":
        return np.tril(np.ones((n, n), dtype=bool))
    if kind == "banded":
        M = np.eye(n, dtype=bool)
        for k in range(1, max(1, n // 2)):
            M |= np.eye(n, k=-k, dtype=bool)
        return M
    raise ValueError(f"unknown support kind {kind!r}")


def linear_mixing(n: int, support: np.ndarray, rng: np.random.Generator,
                  cond_clip: float = 8.0) -> np.ndarray:
    """A well-conditioned invertible mixing matrix A respecting `support`.

    Off-support entries are exactly zero (so the support is the real Jacobian
    support). Diagonal entries are random nonzero (|.| in [0.7, 1.5], random
    sign) to keep triangular/banded supports invertible WITHOUT an identity
    boost that would wash out the sparse-vs-dense contrast.
    """
    for _ in range(500):
        A = rng.normal(size=(n, n))
        A = np.where(support, A, 0.0)
        diag = (rng.uniform(0.7, 1.5, n)) * rng.choice([-1.0, 1.0], n)
        np.fill_diagonal(A, diag)
        sv = np.linalg.svd(A, compute_uv=False)
        if sv[-1] > 1e-2 and sv[0] / sv[-1] < cond_clip:
            return A
    return A  # best effort


def make_linear(n_samples: int, n: int, support_kind: str = "sparse",
                source_dist: str = "gaussian", seed: int = 0) -> dict:
    """Linear ICA problem x = A s. Returns sources, obs, mixing, support.

    Defaults to Gaussian sources so identifiability must come from the mixing's
    structural sparsity, not from non-Gaussianity — the cleanest test of the
    2022 linear corollary (Prop. 1 / Thm. 2), and CPU-cheap.
    """
    rng = np.random.default_rng(seed)
    s = sample_sources(n_samples, n, source_dist, rng)
    support = structural_support(n, support_kind, rng)
    A = linear_mixing(n, support, rng)
    x = s @ A.T
    return {"s": s, "x": x, "A": A, "support": support}


def make_nonlinear(n_samples: int, n: int, n_layers: int = 3,
                   support_kind: str = "sparse", source_dist: str = "laplace",
                   seed: int = 0) -> dict:
    """Nonlinear ICA problem x = f(s), f = composition of (LeakyReLU o Linear).

    Each linear layer respects `support` (a masked, near-triangular invertible
    matrix), so the composite mixing Jacobian inherits structural sparsity. This
    is the standard synthetic nonlinear-ICA mixing (Hyvarinen TCL / Zheng et
    al.). Non-Gaussian sources by default (classic nonlinear ICA), with the
    Jacobian structure as the identifying signal the unmixer must exploit.
    """
    rng = np.random.default_rng(seed)
    s = sample_sources(n_samples, n, source_dist, rng)
    support = structural_support(n, support_kind, rng)
    h = s
    weights = []
    for _ in range(n_layers):
        W = linear_mixing(n, support, rng)
        weights.append(W)
        h = h @ W.T
        h = np.where(h > 0, h, 0.1 * h)  # LeakyReLU(0.1), invertible
    return {"s": s, "x": h, "weights": weights, "support": support}
