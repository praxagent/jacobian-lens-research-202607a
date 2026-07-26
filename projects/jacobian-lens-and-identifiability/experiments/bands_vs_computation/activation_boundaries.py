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
sys.path.insert(0, str(ROOT))
from common.cka import linear_cka                    # noqa: E402


def fitted_seg(M):
    """(b1, b2, within_mean - between_mean) for the best 3 contiguous segments.

    Vendored verbatim from jspace_atlas/atlas_stage_a.py so this runner ships with no atlas
    dependency chain: importing that module pulls in cka_layers, which is not needed here and
    broke a pod run. Any change there must be mirrored here; the two are asserted equal by
    test_fitted_seg_matches_atlas() below when both are importable.
    """
    L = M.shape[0]
    S = M.cumsum(0).cumsum(1)

    def block_sum(a, b):
        t = S[b - 1, b - 1]
        if a > 0:
            t = t - S[a - 1, b - 1] - S[b - 1, a - 1] + S[a - 1, a - 1]
        return t

    best = (-1e9, 1, 2)
    for b1 in range(2, L - 3):
        for b2 in range(b1 + 2, L - 1):
            score = 0.0
            for a, b in ((0, b1), (b1, b2), (b2, L)):
                nn = b - a
                score += (block_sum(a, b) - nn) / max(nn * nn - nn, 1)
            if score > best[0]:
                best = (score, b1, b2)
    _, b1, b2 = best
    mask = np.ones((L, L), bool)
    np.fill_diagonal(mask, False)
    seg_id = np.zeros(L, int)
    seg_id[b1:b2] = 1
    seg_id[b2:] = 2
    same = seg_id[:, None] == seg_id[None, :]
    return b1, b2, float(M[mask & same].mean()) - float(M[mask & ~same].mean())


def test_fitted_seg_matches_atlas(M):
    """Assert the vendored copy agrees with the atlas original, when it can be imported."""
    try:
        sys.path.insert(0, str(ROOT / "experiments/jspace_atlas"))
        from atlas_stage_a import fitted_seg as orig
    except Exception as e:
        return f"skipped ({type(e).__name__})"
    a, b = fitted_seg(M), orig(M)
    assert a[0] == b[0] and a[1] == b[1] and abs(a[2] - b[2]) < 1e-9, (a, b)
    return "PASS"


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

    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16).to(a.device).eval()

    # same frozen prose prompts Test C uses, so the two tests share an input distribution
    texts = json.load(open(a.prompts))["prose"][:a.n_prompts]
    acc = None
    with torch.no_grad():
        for p in texts:
            ids = tok(p, return_tensors="pt").input_ids.to(a.device)
            # request on the FORWARD call, not the constructor: some architectures
            # (qwen3.5's hybrid attention) silently ignore the config-time flag and return None
            hs = model(ids, output_hidden_states=True).hidden_states
            if hs is None:
                raise SystemExit("hidden_states is None; this architecture needs another route")
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
    vendor_check = test_fitted_seg_matches_atlas(M)

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
           "vendored_fitted_seg_check": vendor_check,
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
