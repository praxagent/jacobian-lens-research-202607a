"""Activation-CKA: does gemma-2-9b's flatness live in the model or only in the lens?

For each model, run a forward pass over WikiText prompts, capture the RAW hidden states
at every layer, and compute the layer x layer CKA of the activations themselves (not the
lens). Compare to the J-lens CKA:
  - if activations show clear depth structure where the lens is flat -> the flatness is a
    property of the first-order OUTPUT linearization (the lens), not the representation.
  - if activations are ALSO flat -> gemma-2-9b is a genuine one-geometry network.
Structured control: qwen3-4b (its lens is structured; activations should be too).

Inference only, no fitting, no jlens. Run on a small GPU.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from common.cka import linear_cka   # noqa: E402


def prompts(n, tok):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    out, i = [], 0
    for t in ds["text"]:
        if len(t) >= 300:
            out.append(t)
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-prompts", type=int, default=80)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--n-tok", type=int, default=4096)   # subsample tokens for CKA
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, output_hidden_states=True).to("cuda").eval()
    ps = prompts(a.n_prompts, tok)
    # collect per-layer hidden states over all tokens
    acc = None
    with torch.no_grad():
        for p in ps:
            ids = tok(p, return_tensors="pt", truncation=True,
                      max_length=a.max_len).input_ids.to("cuda")
            hs = model(ids).hidden_states           # tuple[L+1] of [1, seq, d]
            layers = [h[0].float().cpu() for h in hs]   # drop embedding layer? keep all
            if acc is None:
                acc = [[] for _ in layers]
            for k, h in enumerate(layers):
                acc[k].append(h)
    # [n_tokens, d] per layer, subsample same token indices across layers
    stacked = [torch.cat(a_k, 0).numpy() for a_k in acc]   # each [T, d]
    T = stacked[0].shape[0]
    rng = np.random.default_rng(0)
    idx = rng.choice(T, size=min(a.n_tok, T), replace=False)
    X = [s[idx].astype(np.float32) for s in stacked[1:]]    # skip embedding layer (index 0)
    L = len(X)
    M = np.eye(L)
    for i in range(L):
        for j in range(i+1, L):
            M[i, j] = M[j, i] = linear_cka(X[i], X[j])
    tri = M[np.triu_indices_from(M, 1)]
    # band stat (fixed thirds)
    th = np.array_split(np.arange(L), 3)
    def blk(a_, b_):
        v = [M[i, j] for i in a_ for j in b_ if i != j]
        return float(np.mean(v)) if v else 1.0
    e, mid, la = th
    mid_sep = blk(mid, mid) - 0.5*(blk(e, mid) + blk(mid, la))
    res = {"model": a.model, "tag": a.tag, "n_layers": L, "n_tokens": len(idx),
           "activation_cka_offdiag_median": float(np.median(tri)),
           "activation_cka_offdiag_min": float(tri.min()),
           "activation_cka_mid_sep": float(mid_sep)}
    np.savez_compressed(a.out + ".npz", cka=M)
    Path(a.out + ".json").write_text(json.dumps(res, indent=1))
    print(f"ACTCKA {a.tag}: offdiag median={np.median(tri):.4f} min={tri.min():.4f} "
          f"mid_sep={mid_sep:+.4f} L={L}", flush=True)


if __name__ == "__main__":
    main()
