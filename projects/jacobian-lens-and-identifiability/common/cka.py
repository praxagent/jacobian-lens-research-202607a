"""Centered Kernel Alignment (CKA) — layer/model geometry comparison. NumPy only.

Used to partially replicate the eliebak "J-lens CKA Explorer"
(https://eliebak.com/viz/jspace-open): compare the geometry of J-lens token
directions between layers of one model, or across models at matched depth. CKA
is cheap (Gram-matrix contractions), so with *pre-fitted* lenses (Neuronpedia)
this whole comparison runs on CPU — the expensive part (fitting the lens) is
already done.

Reference: Kornblith et al., "Similarity of Neural Network Representations
Revisited" (ICML 2019).
"""
from __future__ import annotations

import numpy as np


def _center_gram(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between feature matrices X (n,p) and Y (n,q).

    n = number of probe tokens (shared across the two layers/models); columns
    are feature dimensions. Invariant to orthogonal transforms and isotropic
    scaling; 1.0 = identical geometry, ~0 = unrelated. Uses the efficient
    ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F) form on column-centered features.
    """
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xty = np.linalg.norm(Y.T @ X, "fro") ** 2
    xx = np.linalg.norm(X.T @ X, "fro")
    yy = np.linalg.norm(Y.T @ Y, "fro")
    denom = xx * yy
    return float(xty / denom) if denom > 0 else 0.0


def kernel_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """RBF-kernel CKA (median-heuristic bandwidth). Captures nonlinear geometry
    similarity; linear_cka is usually enough and much cheaper."""
    def rbf(Z):
        sq = np.sum(Z ** 2, 1)
        d = sq[:, None] + sq[None, :] - 2 * Z @ Z.T
        med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        return np.exp(-d / (2 * med + 1e-12))
    Kx, Ky = _center_gram(rbf(X)), _center_gram(rbf(Y))
    hsic = np.sum(Kx * Ky)
    denom = np.sqrt(np.sum(Kx * Kx) * np.sum(Ky * Ky))
    return float(hsic / denom) if denom > 0 else 0.0


def cka_matrix(layer_features: list[np.ndarray]) -> np.ndarray:
    """Pairwise linear-CKA matrix over a list of (n, d_l) layer feature blocks
    (each row = the same shared probe token). This is the eliebak explorer's
    core object: the layer x layer geometry-similarity heatmap."""
    L = len(layer_features)
    M = np.eye(L)
    for i in range(L):
        for j in range(i + 1, L):
            M[i, j] = M[j, i] = linear_cka(layer_features[i], layer_features[j])
    return M
