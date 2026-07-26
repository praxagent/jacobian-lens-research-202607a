"""Test B runner: activation-CKA map + fitted boundaries, for comparison against the lens map.

Design frozen in PREREG.md. Forward passes only, no fitting, no jlens.

For each model: capture raw hidden states over WikiText prompts, compute the layer x layer
linear CKA of the activations, fit the same three-segmentation used everywhere in this campaign,
and write both the map and the boundaries. The lens boundaries come from the already-cached
shared-vocabulary maps and are NOT recomputed here.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ATLAS = ROOT / "experiments/jspace_atlas"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ATLAS))
from common.cka import linear_cka                    # noqa: E402


def band_sep(M):
    """Fixed index-thirds band separation, identical to the atlas definition."""
    L = M.shape[0]
    e, mid, la = np.array_split(np.arange(L), 3)
    def blk(a_, b_):
        v = [M[i, j] for i in a_ for j in b_ if i != j]
        return float(np.mean(v)) if v else 1.0
    return blk(mid, mid) - 0.5 * (blk(e, mid) + blk(mid, la))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--prompts", default=str(HERE / "prompts_frozen.json"))
    ap.add_argument("--n-prompts", type=int, default=48)
    ap.add_argument("--n-tok", type=int, default=4096)
    ap.add_argument("--lens-npz", required=True,
                    help="cached shared-vocab lens map; its `layers` array fixes the index set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import transformers
    from atlas_stage_a import fitted_seg

    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, output_hidden_states=True).to(a.device).eval()

    # same frozen prose prompts Test C uses, so the two tests share an input distribution
    texts = json.load(open(a.prompts))["prose"][:a.n_prompts]
    acc = None
    with torch.no_grad():
        for p in texts:
            ids = tok(p, return_tensors="pt").input_ids.to(a.device)
            hs = model(ids).hidden_states               # tuple[L+1] of [1, seq, d]
            hh = [h[0].float().cpu() for h in hs]
            if acc is None:
                acc = [[] for _ in hh]
            for k, h in enumerate(hh):
                acc[k].append(h)

    stacked = [torch.cat(v, 0).numpy() for v in acc]
    T = stacked[0].shape[0]
    rng = np.random.default_rng(0)
    idx = rng.choice(T, size=min(a.n_tok, T), replace=False)

    # Align to EXACTLY the lens's source layers so boundary indices live in one index space.
    # hidden_states[0] is the embedding, so block l's output is hidden_states[l + 1].
    lz = np.load(a.lens_npz)
    lens_layers = [int(v) for v in lz["layers"]]
    lens_b = [int(v) for v in lz["seg"]]
    missing = [l for l in lens_layers if l + 1 >= len(stacked)]
    if missing:
        raise SystemExit(f"lens layers {missing} exceed captured hidden states "
                         f"({len(stacked)-1} blocks)")
    X = [stacked[l + 1][idx].astype(np.float32) for l in lens_layers]
    L = len(X)

    M = np.eye(L)
    for i in range(L):
        for j in range(i + 1, L):
            M[i, j] = M[j, i] = linear_cka(X[i], X[j])
    tri = M[np.triu_indices_from(M, 1)]
    b1, b2, sep = fitted_seg(M)

    res = {"slug": a.slug, "model": a.model, "n_layers": L, "n_tokens": int(len(idx)),
           "n_prompts": len(texts),
           "lens_layers": lens_layers, "lens_boundaries": lens_b,
           "alignment": "activation position k == lens source layer lens_layers[k]; "
                        "block l output taken as hidden_states[l+1]",
           "boundary_shift_lens_vs_act": int(abs(lens_b[0] - int(b1))
                                             + abs(lens_b[1] - int(b2))),
           "act_offdiag_median": float(np.median(tri)),
           "act_offdiag_min": float(tri.min()),
           "act_mid_sep": float(band_sep(M)),
           "act_boundaries": [int(b1), int(b2)],
           "act_fitted_sep": float(sep),
           "diag_ok": bool(np.allclose(np.diag(M), 1.0, atol=1e-4)),
           "degenerate": bool(np.median(tri) >= 0.999),
           "transformers": transformers.__version__, "torch": torch.__version__}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out + ".npz", cka=M)
    Path(a.out + ".json").write_text(json.dumps(res, indent=1))
    print(f"ACTB {a.slug}: L={L} lens_b={lens_b} act_b={res['act_boundaries']} "
          f"shift={res['boundary_shift_lens_vs_act']} "
          f"mid_sep={res['act_mid_sep']:+.4f} offdiag_median={res['act_offdiag_median']:.4f} "
          f"diag_ok={res['diag_ok']} degenerate={res['degenerate']}", flush=True)


if __name__ == "__main__":
    main()
