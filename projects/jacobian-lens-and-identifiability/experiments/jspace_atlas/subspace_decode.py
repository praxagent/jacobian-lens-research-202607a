"""#1-followup: WHAT does gemma-2-9b's invariant readout subspace decode to?

We showed the flatness is confined to the unembedding-weighted (readout) subspace. Now
identify that subspace and read it out in tokens. Uses the real unembedding U (tied
embed_tokens, fetched separately) + the cached lens J.

  M = U^T U                                  (readout second moment, d x d)
  A = mean_l J_l^T M J_l                      (avg readout energy operator, d x d, PSD)
  w = top eigenvector(s) of A                 (invariant readout direction in residual space)
  logit_l = U @ (J_l @ w)                      (what layer l's readout of w favors, per token)

If the per-layer logit_l are near-identical across layers (high cosine), that IS the
invariant readout, and its top +/- tokens name what gemma-2 holds constant through depth.
We also report the eigenvalue participation ratio (how low-dim the invariant subspace is)
and the cross-layer cosine of the readout.

CPU-only. Needs decompose_out/gemma2_9b_embed.npy + the lens .pt + gemma tokenizer.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEC = HERE / "decompose_out"
LENS = "/home/ubuntu/hf-lenses-tier1/gemma2_9b_wiki.pt"
CHUNK = 8192
TOPK = 25


def gram_UtU(U: np.ndarray) -> np.ndarray:
    d = U.shape[1]
    M = np.zeros((d, d), dtype=np.float64)
    for i in range(0, U.shape[0], CHUNK):
        c = U[i:i+CHUNK].astype(np.float32)
        M += c.T @ c
    return M.astype(np.float32)


def band_sep(Mx):
    L = Mx.shape[0]; th = np.array_split(np.arange(L), 3)
    blk = lambda a, b: float(np.mean([Mx[i, j] for i in a for j in b if i != j]) or 1.0)
    e, m, l = th
    return blk(m, m) - 0.5*(blk(e, m) + blk(m, l))


def main():
    print("loading U (unembedding)...", flush=True)
    U = np.load(DEC / "gemma2_9b_embed.npy")            # (vocab, d) fp16
    V, d = U.shape
    print(f"U {U.shape} {U.dtype}", flush=True)
    print("computing M = U^T U (chunked)...", flush=True)
    M = gram_UtU(U)                                     # (d, d) f32

    dd = torch.load(LENS, map_location="cpu", weights_only=False)
    Jd = dd["J"]; layers = sorted(Jd.keys())
    Js = [Jd[l].float().numpy() for l in layers]        # list of (d, d)

    # A = mean_l J_l^T M J_l ; also readout-CKA sanity from the M-weighted grams
    print("building A = mean J^T M J ...", flush=True)
    A = np.zeros((d, d), dtype=np.float64)
    Gs = []                                             # M-weighted gram per layer for CKA sanity
    for J in Js:
        MJ = M @ J                                      # (d,d)
        A += (J.T @ MJ)
        Gs.append(J.T @ MJ)                             # = J^T M J (the readout second moment at layer l)
    A = (A / len(Js)).astype(np.float32)

    # readout-CKA sanity: CKA between the per-layer J^T M J should reproduce the flat map
    n = len(Gs); norm = [np.linalg.norm(g, "fro") for g in Gs]; Mck = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            Mck[i, j] = Mck[j, i] = float(np.sum(Gs[i]*Gs[j])/(norm[i]*norm[j]))
    readout_sanity = band_sep(Mck)

    # invariant readout directions = top eigenvectors of A
    evals, evecs = np.linalg.eigh(A)
    order = np.argsort(evals)[::-1]
    evals = evals[order]; evecs = evecs[:, order]
    pr = float((evals.sum()**2) / (np.square(evals).sum() + 1e-12))
    w = evecs[:, 0]                                     # top invariant readout direction

    # per-layer logit vector for w; cross-layer cosine (is the readout the same at every layer?)
    logits = np.stack([U.astype(np.float32) @ (J @ w) for J in Js])   # (L, vocab)
    ln = logits / (np.linalg.norm(logits, axis=1, keepdims=True) + 1e-9)
    cross = ln @ ln.T
    cross_cos = float(cross[np.triu_indices(n, 1)].mean())
    mean_logit = logits.mean(0)

    # sign convention: point w so the strongest tokens are the positive tail
    if abs(mean_logit.min()) > abs(mean_logit.max()):
        mean_logit = -mean_logit; logits = -logits
    top_pos = np.argsort(mean_logit)[::-1][:TOPK].tolist()
    top_neg = np.argsort(mean_logit)[:TOPK].tolist()

    # decode with the gemma tokenizer (tiny download; HF_TOKEN in env)
    toks_pos = toks_neg = None
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained("google/gemma-2-9b")
        toks_pos = [tk.convert_ids_to_tokens(i) for i in top_pos]
        toks_neg = [tk.convert_ids_to_tokens(i) for i in top_neg]
    except Exception as e:
        print("tokenizer decode skipped:", e, flush=True)

    # is the invariant readout the "unigram/frequency prior"? proxy: cosine of mean_logit
    # with U's per-token norm (embedding norm tracks token frequency/confidence in tied models)
    unorm = np.linalg.norm(U.astype(np.float32), axis=1)
    freq_align = float(np.corrcoef(mean_logit, unorm)[0, 1])

    res = {
        "model": "gemma-2-9b", "vocab": int(V), "d": int(d), "n_layers": n,
        "readout_cka_bandsep_sanity": round(readout_sanity, 4),
        "invariant_subspace_participation_ratio": round(pr, 2),
        "top_eig_share": round(float(evals[0]/evals.sum()), 4),
        "top5_eig_share": round(float(evals[:5].sum()/evals.sum()), 4),
        "cross_layer_readout_cosine": round(cross_cos, 4),
        "freq_prior_alignment_corr": round(freq_align, 4),
        "top_pos_tokens": toks_pos, "top_neg_tokens": toks_neg,
        "top_pos_ids": top_pos, "top_neg_ids": top_neg,
    }
    (DEC / "subspace_decode.json").write_text(json.dumps(res, indent=1))
    print("\n=== INVARIANT READOUT DECODE ===", flush=True)
    print(f"readout-CKA sanity band-sep (should be ~+0.005 flat): {readout_sanity:+.4f}")
    print(f"invariant subspace participation ratio: {pr:.1f}  (top eig {res['top_eig_share']:.2%}, top5 {res['top5_eig_share']:.2%})")
    print(f"cross-layer readout cosine (1.0 = identical readout every layer): {cross_cos:.4f}")
    print(f"freq-prior alignment (corr of readout with token embed-norm): {freq_align:+.3f}")
    if toks_pos:
        print("top + tokens:", " ".join(repr(t) for t in toks_pos[:20]))
        print("top - tokens:", " ".join(repr(t) for t in toks_neg[:20]))
    print("SUBSPACE_DECODE_DONE", flush=True)


if __name__ == "__main__":
    main()
