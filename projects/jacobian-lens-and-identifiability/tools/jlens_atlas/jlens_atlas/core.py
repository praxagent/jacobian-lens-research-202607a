"""Numerics. numpy + torch only. Vendored from the campaign repository with the source named
on each function; the point of vendoring is that a reader can check this file against the
experiment code that produced the published numbers."""
from __future__ import annotations
import numpy as np
import torch


# ---- common/cka.py -------------------------------------------------------------------------
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between feature matrices X (n, p) and Y (n, q); rows are the same n probe
    tokens. ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F) on column-centered features. Invariant to
    orthogonal transforms and isotropic scaling; NOT to per-dimension scaling."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xty = np.linalg.norm(Y.T @ X, "fro") ** 2
    denom = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(xty / denom) if denom > 0 else 0.0


def cka_matrix(geoms: list[np.ndarray]) -> np.ndarray:
    """Pairwise linear CKA over a list of (n, d) layer geometries."""
    L = len(geoms)
    M = np.eye(L)
    for i in range(L):
        for j in range(i + 1, L):
            M[i, j] = M[j, i] = linear_cka(geoms[i], geoms[j])
    return M


# ---- experiments/jacobian_lens/cka_layers.py ------------------------------------------------
def band_stats(M: np.ndarray) -> dict:
    """Fixed index-thirds block means and mid_sep = within_mid - mean(early_mid, mid_late)."""
    L = M.shape[0]
    thirds = np.array_split(np.arange(L), 3)

    def block_mean(a, b):
        vals = [M[i, j] for i in a for j in b if i != j]
        return float(np.mean(vals)) if vals else 1.0
    early, mid, late = thirds
    wm = block_mean(mid, mid)
    return {"within_early": block_mean(early, early), "within_mid": wm,
            "within_late": block_mean(late, late), "early_mid": block_mean(early, mid),
            "mid_late": block_mean(mid, late), "early_late": block_mean(early, late),
            "mid_sep": wm - 0.5 * (block_mean(early, mid) + block_mean(mid, late)),
            "thirds": [[int(t[0]), int(t[-1])] for t in thirds]}


# ---- experiments/jspace_atlas/atlas_stage_a.py ----------------------------------------------
def _block_scores(M: np.ndarray):
    L = M.shape[0]
    S = M.cumsum(0).cumsum(1)

    def block_sum(a, b):
        t = S[b - 1, b - 1]
        if a > 0:
            t = t - S[a - 1, b - 1] - S[b - 1, a - 1] + S[a - 1, a - 1]
        return t
    out = {}
    for b1 in range(2, L - 3):
        for b2 in range(b1 + 2, L - 1):
            s = 0.0
            for a, b in ((0, b1), (b1, b2), (b2, L)):
                n = b - a
                s += (block_sum(a, b) - n) / max(n * n - n, 1)
            out[(b1, b2)] = s
    return out


def fitted_seg(M: np.ndarray):
    """(b1, b2, fitted_sep): the 3 contiguous segments maximising the sum over blocks of mean
    off-diagonal within-block CKA; fitted_sep is within-mean minus between-mean at that split.
    Returns (None, None, None) when L < 8 (no legal segmentation with the margins used)."""
    sc = _block_scores(M)
    if not sc:
        return None, None, None
    (b1, b2) = max(sc, key=sc.get)
    L = M.shape[0]
    mask = np.ones((L, L), bool)
    np.fill_diagonal(mask, False)
    seg_id = np.zeros(L, int)
    seg_id[b1:b2] = 1
    seg_id[b2:] = 2
    same = seg_id[:, None] == seg_id[None, :]
    return int(b1), int(b2), float(M[mask & same].mean() - M[mask & ~same].mean())


# ---- experiments/jspace_atlas/boundary_identifiability.py -----------------------------------
def segmentation_scores(M: np.ndarray) -> dict:
    """{(b1, b2): objective} for every legal 3-segmentation (same objective as fitted_seg)."""
    return _block_scores(M)


def near_optimal_spread(M: np.ndarray, tol: float = 0.05):
    """(best_seg, n_near, spread_b1, spread_b2, objective_range) or None if L is too small.
    Near-optimal set: segmentations within tol * (best - worst) of the best. Spread: (max - min)
    of a boundary over that set, as a fraction of L."""
    sc = segmentation_scores(M)
    if not sc:
        return None
    L = M.shape[0]
    keys = list(sc.keys()); v = np.array([sc[k] for k in keys])
    best, worst = v.max(), v.min(); rng = best - worst
    near = [keys[i] for i in range(len(keys)) if v[i] >= best - tol * rng]
    b1 = [k[0] for k in near]; b2 = [k[1] for k in near]
    return (keys[int(v.argmax())], len(near), (max(b1) - min(b1)) / L,
            (max(b2) - min(b2)) / L, float(rng))


# ---- experiments/jspace_atlas/atlas_stage_a.py (pr_of) --------------------------------------
def participation_ratio(D: torch.Tensor) -> float:
    """(sum s^2)^2 / sum s^4 over the singular values of the (n, d) geometry."""
    s = torch.linalg.svdvals(D.float())
    s2 = s ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


# ---- experiments/fit_our_own/cka_heatmap_397b.py (null) -------------------------------------
def random_transport(J: torch.Tensor, gen: torch.Generator, match: str = "frob") -> torch.Tensor:
    """Random matrix with the shape of J. 'frob' matches the Frobenius norm (the convention of
    the 397B release and the atlas note); 'std' matches the entry standard deviation (the
    convention of atlas_stage_a's participation-ratio null)."""
    R = torch.randn(J.shape, generator=gen)
    if match == "frob":
        return R * (J.norm() / R.norm())
    if match == "std":
        return R * J.float().std()
    raise ValueError(match)
