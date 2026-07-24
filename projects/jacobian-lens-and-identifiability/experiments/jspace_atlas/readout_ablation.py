"""FINAL PROBE: is gemma-2-small's extreme base flatness caused by its readout covariance
being dominated by a few directions (the frequency-prior axis)?

Setup. The shared-vocab readout geometry is D_l = U_s J_l (4096 probe tokens x d). Column
centering gives D_c = U_c J_l with U_c the column-centered probe embedding, so the ENTIRE
readout CKA is a function of the centered readout covariance M_c = U_c^T U_c:

    CKA(D_i, D_j) = ||J_j^T M_c J_i||_F^2 / (||J_i^T M_c J_i||_F ||J_j^T M_c J_j||_F)

Write M_c = Q S^2 Q^T (SVD of U_c). Then with B_l = S Q^T J_l and C_l = B_l B_l^T,
CKA_ij = <C_i, C_j>_F / (||C_i||_F ||C_j||_F) exactly. Ablating the top-k readout
eigendirections is then just dropping the first k rows/cols: C_l[k:, k:]. Cheap and exact.

Two questions:
  (a) CONCENTRATION: how much of the readout energy sits in the top few eigendirections
      (participation ratio of S^2)? Flat-lens models should be far more concentrated.
  (b) CAUSAL-ISH ABLATION: does band separation RETURN when the top-k directions are
      removed? If gemma's flatness is the prior axis, removing a handful of directions
      should restore normal depth structure; the structured controls should barely move.

Models: gemma-2-2b, gemma-2-9b (flat) vs gemma-2-27b (STRUCTURED, same family, same 256k
tied vocab, same soft-capping) and qwen3-4b (structured, other family). gemma-2-27b is the
control that isolates "gemma-2 small-model recipe" from "gemma-2 family/vocab".

Validation: k=0 must reproduce the atlas shared-vocab mid_sep for each model.
RAM-frugal: U memory-mapped, only 4096 probe rows read; one Jacobian in float32 at a time.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEC = HERE / "decompose_out"
SM = HERE / "atlas_out/shared_maps"
STRINGS = HERE / "atlas_out/gemma-2-9b.strings.json"      # the shared 4096-token probe
RANK = 1024                                               # top readout directions retained
KS = [0, 1, 2, 4, 8, 16, 32, 64, 128]

MODELS = {                       # slug -> (hf repo for tokenizer, flat?)
    "gemma-2-2b":  ("google/gemma-2-2b", True),
    "gemma-2-9b":  ("google/gemma-2-9b", True),
    "gemma-2-27b": ("google/gemma-2-27b", False),
    "qwen3-4b":    ("Qwen/Qwen3-4B", False),
}


def band_sep(M):
    L = M.shape[0]; th = np.array_split(np.arange(L), 3)
    blk = lambda a, b: float(np.mean([M[i, j] for i in a for j in b if i != j]) or 1.0)
    e, m, l = th
    return blk(m, m) - 0.5 * (blk(e, m) + blk(m, l))


def load_lens(slug):
    """Neuronpedia / our lens -> (sorted layer list, getter returning float32 (d,d))."""
    p = DEC / f"{slug}_np_lens.pt"
    d = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(d, dict) and "J" in d:
        Jd = d["J"]
    elif isinstance(d, dict) and "jacobians" in d:
        Jd = d["jacobians"]
    elif isinstance(d, dict) and all(isinstance(k, int) for k in d):
        Jd = d
    else:                                   # last resort: first dict-of-tensors value
        Jd = next(v for v in d.values() if isinstance(v, dict))
    layers = sorted(Jd.keys())
    return layers, (lambda l: np.asarray(Jd[l].float().numpy(), dtype=np.float32))


def shared_ids(repo, strings):
    """The shared list holds TOKEN STRINGS, so resolve by vocabulary lookup. Re-encoding
    the string instead drops ~25-50% of the probe (and caps the covariance rank at the
    probe size, which silently biases a cross-model comparison). Falls back to encode()
    only for entries the vocab lookup cannot resolve."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(repo)
    unk = tk.unk_token_id
    ids, kept = [], []
    for s in strings:
        i = tk.convert_tokens_to_ids(s)
        if i is None or i == unk:                       # try the space-prefixed variants
            for v in ("▁" + s, "Ġ" + s):
                j = tk.convert_tokens_to_ids(v)
                if j is not None and j != unk:
                    i = j; break
        if i is None or i == unk:
            t = tk.encode(s, add_special_tokens=False)
            i = t[0] if len(t) == 1 else None
        if i is not None and i != unk:
            ids.append(i); kept.append(s)
    return np.array(ids), kept


def run_model(slug):
    repo, is_flat = MODELS[slug]
    strings = json.load(open(STRINGS))
    ids, kept = shared_ids(repo, strings)
    U = np.load(DEC / f"{slug}_embed.npy", mmap_mode="r")          # (vocab, d) fp16 on disk
    Us = np.asarray(U[ids], dtype=np.float32)                      # (n, d) -- only 4096 rows
    Uc = Us - Us.mean(0, keepdims=True)                            # column-centered
    # SVD -> readout eigenbasis: M_c = Q S^2 Q^T
    _, S, Qt = np.linalg.svd(Uc, full_matrices=False)              # Qt: (k, d)
    ev = S**2
    r = min(RANK, (ev > ev.max()*1e-12).sum())
    pr = float((ev.sum()**2) / (np.square(ev).sum() + 1e-12))      # participation ratio of M_c
    shares = {f"top{k}_share": round(float(ev[:k].sum()/ev.sum()), 4) for k in (1, 4, 16, 64)}
    kept_share = float(ev[:r].sum()/ev.sum())

    layers, Jget = load_lens(slug)
    Sr = S[:r][:, None]; Qr = Qt[:r]                               # (r,1), (r,d)
    Cs = []
    for l in layers:
        B = Sr * (Qr @ Jget(l))                                    # (r,d)  = S Q^T J
        Cs.append((B @ B.T).astype(np.float32))                    # (r,r)
    n = len(Cs)

    out_k = {}
    for k in KS:
        if k >= r: continue
        sub = [C[k:, k:] for C in Cs]
        nrm = [np.linalg.norm(c, "fro") for c in sub]
        M = np.eye(n)
        for i in range(n):
            for j in range(i+1, n):
                M[i, j] = M[j, i] = float(np.sum(sub[i]*sub[j])/(nrm[i]*nrm[j]))
        out_k[k] = round(band_sep(M), 4)

    atlas = None
    ap = SM / f"{slug}.npz"
    if ap.exists():
        atlas = round(float(np.load(ap, allow_pickle=True)["mid_sep"]), 4)
    res = {"slug": slug, "flat_lens": is_flat, "n_layers": n, "d": int(Uc.shape[1]),
           "n_probe": int(len(ids)), "rank_used": int(r), "rank_energy_share": round(kept_share, 5),
           "readout_participation_ratio": round(pr, 2), **shares,
           "band_sep_by_k": out_k, "atlas_mid_sep": atlas}
    print(f"[{slug:12s}] PR={pr:7.2f} top1={shares['top1_share']:.3f} top4={shares['top4_share']:.3f} "
          f"| k=0 {out_k.get(0):+.4f} (atlas {atlas:+.4f}) -> k=4 {out_k.get(4):+.4f} "
          f"k=16 {out_k.get(16):+.4f} k=64 {out_k.get(64):+.4f}", flush=True)
    return res


def main():
    global RANK
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--rank", type=int, default=RANK)
    a = ap.parse_args(); RANK = a.rank
    out = DEC / f"readout_ablation_r{RANK}.json"
    res = json.loads(out.read_text()) if out.exists() else {}      # merge, do not clobber
    for slug in a.models:
        try:
            res[slug] = run_model(slug)
        except Exception as e:
            print(f"[{slug}] FAILED: {type(e).__name__}: {e}", flush=True)
        out.write_text(json.dumps(res, indent=1))                  # save after each model
    print("\n=== READOUT CONCENTRATION vs FLATNESS ===")
    for s, r in res.items():
        print(f"  {s:12s} flat={str(r['flat_lens']):5s} PR={r['readout_participation_ratio']:8.2f} "
              f"band_sep k=0 {r['band_sep_by_k'].get(0):+.4f} -> k=16 {r['band_sep_by_k'].get(16):+.4f}")
    print("READOUT_ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
