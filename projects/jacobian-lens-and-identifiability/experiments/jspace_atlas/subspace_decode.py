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

RAM-FRUGAL (this box is 7.6GB and Lightsail freezes on RAM overage): U is memory-mapped
off disk and only ever touched in vocab chunks; Jacobians are converted to float32 one
layer at a time, never all 41 at once; U is never materialized as a full float32 array.
Peak resident stays under ~2GB. Reboot-resumable: caches M and A to disk.

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
CHUNK = 8192      # vocab chunk (rows of U) -> ~235MB float32 temp at 3584 dims
TOPK = 25


def band_sep(Mx):
    L = Mx.shape[0]; th = np.array_split(np.arange(L), 3)
    blk = lambda a, b: float(np.mean([Mx[i, j] for i in a for j in b if i != j]) or 1.0)
    e, m, l = th
    return blk(m, m) - 0.5 * (blk(e, m) + blk(m, l))


def gram_UtU(U):
    """M = U^T U, reading U (memmap) in chunks so it is never fully float32-resident."""
    d = U.shape[1]; M = np.zeros((d, d), dtype=np.float64)
    for i in range(0, U.shape[0], CHUNK):
        c = np.asarray(U[i:i+CHUNK], dtype=np.float32)
        M += c.T @ c
    return M.astype(np.float32)


def U_project_and_norm(U, Vmat):
    """Single pass over U (read once): logits = U @ Vmat^T  and per-row norm of U.
    Vmat is (n_layers, d). Returns logits (n_layers, vocab) and unorm (vocab,)."""
    n = Vmat.shape[0]; Vt = Vmat.T.astype(np.float32)
    logits = np.empty((n, U.shape[0]), dtype=np.float32)
    unorm = np.empty(U.shape[0], dtype=np.float32)
    for i in range(0, U.shape[0], CHUNK):
        c = np.asarray(U[i:i+CHUNK], dtype=np.float32)      # (chunk, d) -- one disk read of U total
        logits[:, i:i+CHUNK] = (c @ Vt).T
        unorm[i:i+CHUNK] = np.sqrt((c*c).sum(1))
    return logits, unorm


def main():
    print("mmap U (unembedding)...", flush=True)
    U = np.load(DEC / "gemma2_9b_embed.npy", mmap_mode="r")   # (vocab, d) fp16, ON DISK
    V, d = U.shape
    print(f"U {U.shape} {U.dtype} (memmap)", flush=True)

    dd = torch.load(LENS, map_location="cpu", weights_only=False)
    Jd = dd["J"]; layers = sorted(Jd.keys()); n = len(layers)
    Jf = lambda l: Jd[l].float().numpy()                     # one layer float32 on demand (51MB)

    A_cache = DEC / "gemma2_9b_A.npy"; M_cache = DEC / "gemma2_9b_M.npy"
    if A_cache.exists():
        print("loaded cached A", flush=True)
        A = np.load(A_cache); readout_sanity = None
    else:
        if M_cache.exists():
            print("loaded cached M", flush=True); M = np.load(M_cache)
        else:
            print("computing M = U^T U (chunked)...", flush=True); M = gram_UtU(U); np.save(M_cache, M)
        print("building A = mean J^T M J (one layer at a time)...", flush=True)
        A = np.zeros((d, d), dtype=np.float64); Gs = []
        for l in layers:
            J = Jf(l); MJ = M @ J; G = J.T @ MJ
            A += G; Gs.append(G.astype(np.float32))          # G is 51MB; 41 of them = 2.1GB -- acceptable, freed after CKA
        A = (A / n).astype(np.float32); np.save(A_cache, A)
        norm = [np.linalg.norm(g, "fro") for g in Gs]; Mck = np.eye(n)
        for i in range(n):
            for j in range(i+1, n):
                Mck[i, j] = Mck[j, i] = float(np.sum(Gs[i]*Gs[j])/(norm[i]*norm[j]))
        readout_sanity = round(band_sep(Mck), 4); del Gs

    # invariant readout directions = top eigenvectors of A
    evals, evecs = np.linalg.eigh(A)
    order = np.argsort(evals)[::-1]; evals = evals[order]; evecs = evecs[:, order]
    pr = float((evals.sum()**2) / (np.square(evals).sum() + 1e-12))
    w = evecs[:, 0]

    # project w through every layer first (tiny), then read U ONCE (single pass, chunked)
    print("decoding invariant readout (single pass over U)...", flush=True)
    Vmat = np.stack([Jf(l) @ w for l in layers]).astype(np.float32)   # (n, d)
    logits, unorm = U_project_and_norm(U, Vmat)              # (n, vocab), (vocab,)
    ln = logits / (np.linalg.norm(logits, axis=1, keepdims=True) + 1e-9)
    cross = ln @ ln.T
    cross_cos = float(cross[np.triu_indices(n, 1)].mean())
    mean_logit = logits.mean(0)

    if abs(mean_logit.min()) > abs(mean_logit.max()):        # sign: strongest tokens on the + tail
        mean_logit = -mean_logit; logits = -logits
    top_pos = np.argsort(mean_logit)[::-1][:TOPK].tolist()
    top_neg = np.argsort(mean_logit)[:TOPK].tolist()

    toks_pos = toks_neg = None
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained("google/gemma-2-9b")
        toks_pos = [tk.convert_ids_to_tokens(i) for i in top_pos]
        toks_neg = [tk.convert_ids_to_tokens(i) for i in top_neg]
    except Exception as e:
        print("tokenizer decode skipped:", e, flush=True)

    # is the invariant readout the unigram/frequency prior? proxy: corr of readout with U row-norm
    freq_align = float(np.corrcoef(mean_logit, unorm)[0, 1])   # unorm from the single pass above

    res = {
        "model": "gemma-2-9b", "vocab": int(V), "d": int(d), "n_layers": n,
        "readout_cka_bandsep_sanity": readout_sanity,        # None if resumed from A cache
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
    print(f"readout-CKA sanity band-sep (should be ~+0.005 flat): "
          f"{readout_sanity if readout_sanity is not None else 'cached-skip'}")
    print(f"invariant subspace participation ratio: {pr:.1f}  (top eig {res['top_eig_share']:.2%}, top5 {res['top5_eig_share']:.2%})")
    print(f"cross-layer readout cosine (1.0 = identical readout every layer): {cross_cos:.4f}")
    print(f"freq-prior alignment (corr of readout with token embed-norm): {freq_align:+.3f}")
    if toks_pos:
        print("top + tokens:", " ".join(repr(t) for t in toks_pos[:20]))
        print("top - tokens:", " ".join(repr(t) for t in toks_neg[:20]))
    print("SUBSPACE_DECODE_DONE", flush=True)


if __name__ == "__main__":
    main()
