"""Block structure beyond distance decay: the 397B map minus its Toeplitz (distance-only) surrogate.

A random-transport null says the blocks are not an artifact of the shared unembedding; it does not
say they are more than smooth decay of similarity with layer distance, because a Toeplitz map
(every cell replaced by the mean CKA at that layer distance) also yields a positive fixed-thirds
mid_sep. This figure shows the released 397B map, the surrogate with the same distance profile, and
their difference, and the receipt records how much of the band statistic the surrogate explains.
Companion table for the whole zoo: jspace_atlas/atlas_out/toeplitz_surrogate.json.

    build_toeplitz_fig.py            build + write receipt
    build_toeplitz_fig.py --verify   rebuild and assert byte identity
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
NPZ = HERE.parents[1] / "artifacts/lenses-397b/cka_397b.npz"
ZOO = HERE.parent / "jspace_atlas/atlas_out/toeplitz_surrogate.json"
POST = Path("/home/ubuntu/PRAX/pre-blog/blog-source/content/posts/2026/07/jlens-cka-397b")
STEM = "block-excess-397b"
sys.path.insert(0, str(HERE.parent / "jspace_atlas"))
from atlas_stage_a import fitted_seg  # noqa: E402

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Inter", "Arial", "DejaVu Sans"],
                     "figure.facecolor": "#F7F4F0", "axes.facecolor": "#F7F4F0",
                     "savefig.facecolor": "#F7F4F0", "text.color": "#2C2924",
                     "axes.edgecolor": "#A89B8C", "xtick.color": "#5A544C", "ytick.color": "#5A544C",
                     "axes.labelcolor": "#2C2924", "svg.hashsalt": "prax-toeplitz"})


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mid_sep(M):
    L = M.shape[0]; th = np.array_split(np.arange(L), 3)
    blk = lambda a, b: float(np.mean([M[i, j] for i in a for j in b if i != j]))
    e, m, l = th
    return blk(m, m) - 0.5 * (blk(e, m) + blk(m, l))


def toeplitz(M):
    L = M.shape[0]; T = np.eye(L)
    for k in range(1, L):
        v = float(np.mean([M[i, i + k] for i in range(L - k)]))
        for i in range(L - k):
            T[i, i + k] = T[i + k, i] = v
    return T


def build(verify=False):
    z = np.load(NPZ); M = z["cka"]; L = M.shape[0]
    T = toeplitz(M); R = M - T
    ms_real, ms_t = mid_sep(M), mid_sep(T)
    b1, b2, fs_real = fitted_seg(M); tb1, tb2, fs_t = fitted_seg(T)
    zoo = json.loads(ZOO.read_text()) if ZOO.exists() else {}
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.3))
    im0 = axes[0].imshow(M, origin="lower", cmap="magma", vmin=0, vmax=1)
    axes[0].set_title(f"released map, mid_sep {ms_real:+.3f}", fontsize=9.8, fontweight="bold", loc="left")
    axes[1].imshow(T, origin="lower", cmap="magma", vmin=0, vmax=1)
    axes[1].set_title(f"distance-only surrogate, mid_sep {ms_t:+.3f}", fontsize=9.8, fontweight="bold", loc="left")
    lim = float(np.abs(R).max())
    im2 = axes[2].imshow(R, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[2].set_title("difference: block structure beyond distance", fontsize=9.8, fontweight="bold", loc="left")
    for ax in axes:
        ax.set_xlabel("source layer", fontsize=8.5); ax.set_ylabel("source layer", fontsize=8.5)
        for b in (b1, b2):
            ax.axhline(b - 0.5, color="white", lw=0.8, ls="--", alpha=0.8); ax.axvline(b - 0.5, color="white", lw=0.8, ls="--", alpha=0.8)
    fig.colorbar(im0, ax=axes[:2].tolist(), fraction=0.025, pad=0.02, label="linear CKA")
    fig.colorbar(im2, ax=axes[2], fraction=0.05, pad=0.02, label="real minus surrogate")
    frac = ms_t / ms_real if ms_real else float("nan")
    fig.suptitle(f"Qwen3.5-397B lens: a distance-only surrogate reproduces {100*frac:.0f}% of the released mid_sep; "
                 f"the fitted separation exceeds it by {fs_real - fs_t:+.3f}", fontsize=10.6, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    svg = POST / f"{STEM}.svg"
    old = svg.read_bytes() if (verify and svg.exists()) else None
    fig.savefig(svg, format="svg", metadata={"Date": None}); fig.savefig(POST / f"{STEM}.png", dpi=170); plt.close(fig)
    if verify and old is not None and old != svg.read_bytes():
        svg.write_bytes(old); sys.exit("VERIFY FAILED: svg drifted")
    exc = [v["excess_fitted"] for v in zoo.values()] if zoo else []
    alt = (f"Three heatmaps of the 59 by 59 Qwen3.5-397B layer-by-layer CKA map. Left: the released map, mid_sep "
           f"{ms_real:+.3f}, with dashed lines at the fitted boundaries {b1} and {b2}. Middle: a Toeplitz surrogate in "
           f"which every cell holds the mean CKA at that layer distance, mid_sep {ms_t:+.3f}. Right: their difference "
           f"on a red-blue scale, showing the early block, the broad mid band and the late blocks as departures from "
           f"smooth decay. Fitted band separation {fs_real:.3f} real against {fs_t:.3f} surrogate."
           + (f" Across the 36-lens zoo the excess of real over surrogate fitted separation has median {np.median(exc):+.3f} and is positive for {sum(e > 0 for e in exc)} of {len(exc)} lenses." if exc else ""))
    (POST / f"{STEM}.receipt.json").write_text(json.dumps({
        "figure_id": STEM, "title": "Block structure beyond distance decay, 397B lens",
        "alt_text": alt, "description": alt,
        "data_source": [{"receipt": "artifacts/lenses-397b/cka_397b.npz (mirrored in the HF release under cka/)", "sha256": sha(NPZ)},
                        {"receipt": "jspace_atlas/atlas_out/toeplitz_surrogate.json", "sha256": sha(ZOO) if ZOO.exists() else None}],
        "provenance": {"generator": "fit_our_own/build_toeplitz_fig.py", "svg_sha256": sha(svg)},
        "interval_semantics": "deterministic functions of the released matrix; no sampling",
        "plotted_values": {"mid_sep_real": ms_real, "mid_sep_toeplitz": ms_t, "fraction_explained_by_distance": frac,
                           "fitted_sep_real": float(fs_real), "fitted_sep_toeplitz": float(fs_t),
                           "fitted_seg_real": [int(b1), int(b2)], "fitted_seg_toeplitz": [int(tb1), int(tb2)],
                           "zoo_excess_median": float(np.median(exc)) if exc else None,
                           "zoo_excess_positive": int(sum(e > 0 for e in exc)) if exc else None, "zoo_n": len(exc)},
        "accessibility": {"color_only_channel": False, "text_equivalent": "plotted_values"}}, indent=1))
    print(("VERIFY OK " if verify else "built ") + STEM + f"  mid_sep real {ms_real:+.4f} toeplitz {ms_t:+.4f} ({100*frac:.0f}%), fitted {fs_real:.4f} vs {fs_t:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    build(ap.parse_args().verify)
