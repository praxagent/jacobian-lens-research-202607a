"""The recipe, end to end: lens + unembedding rows -> geometries -> map -> statistics -> files."""
from __future__ import annotations
import datetime, json, platform, sys
from pathlib import Path
import numpy as np
import torch

from . import __version__
from .core import (cka_matrix, band_stats, fitted_seg, near_optimal_spread,
                   participation_ratio, random_transport)
from .io import (load_lens_file, resolve_neuronpedia, download_hf_file, build_probe, sha256)

IDENT_TOL, IDENT_SPREAD_MAX = 0.05, 0.25


def geometries(J: dict, layers, U: np.ndarray, geometry_dtype: str = "fp32"):
    """D_l = U J_l for each layer, float32 numpy. geometry_dtype='fp16' rounds each D through
    float16 first, which is exactly what the cached atlas maps did (they stored D in fp16 and
    ran CKA in fp32 on the stored copy); use it to reproduce those maps to machine precision."""
    Ut = torch.from_numpy(U).float()
    out = []
    for l in layers:
        D = Ut @ J[l].float()
        if geometry_dtype == "fp16":
            D = D.to(torch.float16).float()
        out.append(D)
    return out


def run_atlas(*, lens_path=None, neuronpedia=None, hf_repo=None, hf_file=None, hf_revision=None,
              model=None, probe="shared", n_probe=4096, seed=0, geometry_dtype="fp32",
              null="frob", pr=True, out_dir="atlas_out", shared_tokens=None, title=None,
              argv=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = datetime.datetime.now(datetime.timezone.utc)
    lens_meta = {}
    if neuronpedia:
        lens_path, model_from_lens, rev = resolve_neuronpedia(neuronpedia)
        model = model or model_from_lens
        lens_meta.update({"source": f"hf:{'neuronpedia/jacobian-lens'}", "slug": neuronpedia,
                          "revision": rev})
    elif hf_repo:
        lens_path, rev = download_hf_file(hf_repo, hf_file, hf_revision)
        lens_meta.update({"source": f"hf:{hf_repo}", "file": hf_file, "revision": rev})
    else:
        lens_meta.update({"source": "local"})
    if not lens_path or not model:
        raise SystemExit("need a lens (--lens / --neuronpedia / --hf + --file) and --model")
    lens_meta["path"] = str(lens_path); lens_meta["sha256"] = sha256(lens_path)

    J, jmeta = load_lens_file(lens_path)
    layers = sorted(J)
    U, pinfo = build_probe(probe, model, n_probe=n_probe, seed=seed,
                           shared_tokens_path=shared_tokens)
    if U.shape[1] != jmeta["d_final"]:
        raise SystemExit(f"unembedding d={U.shape[1]} but lens d_final={jmeta['d_final']}: "
                         f"wrong --model for this lens?")
    geoms = geometries(J, layers, U, geometry_dtype)
    M = cka_matrix([g.numpy() for g in geoms])
    bs = band_stats(M)
    b1, b2, fsep = fitted_seg(M)
    ident = near_optimal_spread(M, IDENT_TOL)
    prs = [participation_ratio(g) for g in geoms] if pr else None

    gen = torch.Generator().manual_seed(seed)
    Ut = torch.from_numpy(U).float()
    null_geoms = []
    for l in layers:
        R = random_transport(J[l].float(), gen, null)
        D = Ut @ R
        if geometry_dtype == "fp16":
            D = D.to(torch.float16).float()
        null_geoms.append(D.numpy())
    Mn = cka_matrix(null_geoms)
    bsn = band_stats(Mn)

    L = len(layers)
    summary = {
        "model": model, "lens": lens_meta, "lens_meta": jmeta,
        "probe": pinfo, "geometry_dtype": geometry_dtype, "n_layers": L, "layers": layers,
        "mid_sep": bs["mid_sep"], "band_stats": bs,
        "fitted_seg": [b1, b2] if b1 is not None else None, "fitted_sep": fsep,
        "boundary_identifiability": (None if ident is None else {
            "tol": IDENT_TOL, "spread_max": IDENT_SPREAD_MAX, "n_near_optimal": ident[1],
            "spread_b1": ident[2], "spread_b2": ident[3], "objective_range": ident[4],
            "identified": bool(ident[2] <= IDENT_SPREAD_MAX and ident[3] <= IDENT_SPREAD_MAX)}),
        "participation_ratio": prs,
        "pr_over_d_median": (float(np.median(prs) / jmeta["d_final"]) if prs else None),
        "null": {"kind": f"random transport, {null}-matched, seed {seed}",
                 "mid_sep": bsn["mid_sep"], "offdiag_mean": float(Mn[~np.eye(L, dtype=bool)].mean())},
        "offdiag_min": float(M[~np.eye(L, dtype=bool)].min()),
        "offdiag_max": float(M[~np.eye(L, dtype=bool)].max()),
    }
    np.savez_compressed(out_dir / "cka.npz", cka=M, cka_null=Mn, layers=np.array(layers),
                        mid_sep=np.float64(bs["mid_sep"]),
                        fitted_sep=np.float64(fsep if fsep is not None else np.nan),
                        seg=np.array([b1, b2] if b1 is not None else [-1, -1]),
                        null_mid_sep=np.float64(bsn["mid_sep"]),
                        probe=str(pinfo["probe"]), n_probe=np.int64(pinfo["n_probe"]),
                        seed=np.int64(seed), geometry_dtype=str(geometry_dtype))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    receipt = {
        "tool": "jlens_atlas", "version": __version__,
        "command": " ".join(argv) if argv else None,
        "timestamp_utc": t0.isoformat(timespec="seconds"),
        "model": model, "lens": lens_meta,
        "probe": pinfo, "seed": seed, "geometry_dtype": geometry_dtype, "null_match": null,
        "outputs": {"cka.npz": sha256(out_dir / "cka.npz"),
                    "summary.json": sha256(out_dir / "summary.json")},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "torch": torch.__version__},
        "definitions": {
            "geometry": "D_l = U_probe @ J_l  (rows: probe tokens; J_l: d_final x d_layer)",
            "cka": "linear CKA, column-centered, ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)",
            "mid_sep": "mean within-mid-third CKA minus mean of early-mid and mid-late block means, "
                       "fixed index thirds, diagonal excluded",
            "fitted_seg": "argmax over (b1, b2) of the sum over 3 blocks of mean off-diagonal "
                          "within-block CKA; fitted_sep = within-mean minus between-mean there",
            "identifiability": f"spread of each boundary over segmentations within {IDENT_TOL} of "
                               f"the objective range; identified if both <= {IDENT_SPREAD_MAX}",
        },
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=1))
    try:
        _heatmap(M, out_dir, title or f"{model}  J-lens  layer x layer CKA", probe=pinfo["probe"],
                 mid_sep=bs["mid_sep"], seg=(b1, b2))
        receipt["outputs"]["heatmap.png"] = sha256(out_dir / "heatmap.png")
        (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=1))
    except ImportError:
        print("matplotlib not installed: no heatmap written", file=sys.stderr)
    return summary


def _heatmap(M, out_dir, title, probe, mid_sep, seg):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = M.shape[0]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(M, origin="lower", cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xlabel("source layer"); ax.set_ylabel("source layer")
    ax.set_title(f"{title}\n{probe} probe, mid_sep {mid_sep:+.3f}"
                 + (f", fitted boundaries {seg[0]}, {seg[1]}" if seg[0] is not None else ""),
                 fontsize=9.5, loc="left")
    if seg[0] is not None:
        for b in seg:
            ax.axhline(b - 0.5, color="white", lw=0.8, ls="--", alpha=0.8)
            ax.axvline(b - 0.5, color="white", lw=0.8, ls="--", alpha=0.8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="linear CKA of readout geometry")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap.png", dpi=160)
    fig.savefig(out_dir / "heatmap.svg", format="svg", metadata={"Date": None})
    plt.close(fig)
