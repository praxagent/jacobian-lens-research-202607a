"""Evaluation metrics for identifiability replications. NumPy only (CPU-safe).

Kept torch-free on purpose so the smoke test validates the metric harness on
this (GPU-less) box before any RunPod spend.
"""
from __future__ import annotations

import itertools

import numpy as np


def _correlation_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|Pearson corr| between every column of a and every column of b."""
    na, nb = a.shape[1], b.shape[1]
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    a_std = a.std(0, keepdims=True) + 1e-12
    b_std = b.std(0, keepdims=True) + 1e-12
    a = a / a_std
    b = b / b_std
    corr = np.abs(a.T @ b) / a.shape[0]
    return corr  # (na, nb)


def _best_assignment(corr: np.ndarray) -> tuple[float, list[int]]:
    """Max mean over the optimal one-to-one column matching.

    Uses scipy's Hungarian if available (n can be large); falls back to
    brute-force permutations for small n (keeps the CPU smoke test dep-free).
    """
    n = corr.shape[0]
    try:
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(-corr)
        return float(corr[row, col].mean()), list(col)
    except Exception:
        if n > 8:
            # Greedy fallback for large n without scipy (approximate).
            used, cols, total = set(), [], 0.0
            for i in range(n):
                order = np.argsort(-corr[i])
                pick = next(j for j in order if j not in used)
                used.add(pick); cols.append(pick); total += corr[i, pick]
            return total / n, cols
        best, best_perm = -1.0, list(range(n))
        for perm in itertools.permutations(range(n)):
            score = float(np.mean([corr[i, perm[i]] for i in range(n)]))
            if score > best:
                best, best_perm = score, list(perm)
        return best, best_perm


def mcc(true_s: np.ndarray, est_s: np.ndarray) -> float:
    """Mean Correlation Coefficient — the identifiability literature's metric.

    Mean absolute Pearson correlation between matched true/estimated sources
    under the optimal permutation. Invariant to permutation and sign/scale —
    exactly the indeterminacy ICA identifiability tolerates. 1.0 = perfect
    recovery up to the benign gauge; ~1/sqrt(n) = chance.
    """
    corr = _correlation_matrix(np.asarray(true_s), np.asarray(est_s))
    # Square if rectangular (unequal source counts): pad to square with zeros.
    n = max(corr.shape)
    sq = np.zeros((n, n))
    sq[: corr.shape[0], : corr.shape[1]] = corr
    score, _ = _best_assignment(sq)
    return score


def support_iou(true_support: np.ndarray, est_support: np.ndarray,
                thresh: float = 1e-2) -> float:
    """IoU of two Jacobian supports (boolean masks) after optimal column match.

    The identifiability object in Diverse Dictionary Learning is the SUPPORT of
    the Jacobian, recoverable up to a column permutation. This scores how well a
    recovered dependency structure matches the true one.
    """
    t = (np.abs(true_support) > thresh)
    e = (np.abs(est_support) > thresh)
    # match estimated columns to true columns by overlap
    n = t.shape[1]
    overlap = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = np.logical_and(t[:, i], e[:, j]).sum()
            union = np.logical_or(t[:, i], e[:, j]).sum()
            overlap[i, j] = inter / union if union else 1.0
    score, _ = _best_assignment(overlap)
    return score
