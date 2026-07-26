"""Compute a model's own marginal output distribution over the fitting corpus.

This is the primary feature in PREREG.md: what the model says by default, averaged over
contexts, as opposed to an external corpus unigram count. Forward passes only.

    p_marginal[t] = mean over all corpus positions of softmax(logits)[t]

Accumulated in float64 over the vocabulary so the average is stable across many chunks.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-docs", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=2)
    a = ap.parse_args()
    import transformers
    from datasets import load_dataset

    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16).to("cuda").eval()
    V = model.config.vocab_size
    print(f"vocab {V}", flush=True)

    # same corpus family the lenses were fitted on
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    texts, i = [], 0
    for t in ds["text"]:
        if len(t) >= 300:
            texts.append(t)
        if len(texts) >= a.n_docs:
            break

    acc = np.zeros(V, dtype=np.float64)
    npos = 0
    for i in range(0, len(texts), a.chunk):
        enc = tok(texts[i:i+a.chunk], return_tensors="pt", truncation=True,
                  max_length=a.seq_len, padding="max_length").to("cuda")
        with torch.no_grad():
            lg = model(**enc).logits.float()                    # (b, T, V)
        pr = torch.softmax(lg, -1)
        mask = enc["attention_mask"].bool()                     # ignore padding positions
        sel = pr[mask]                                          # (n_valid, V)
        acc += sel.sum(0).double().cpu().numpy()
        npos += int(sel.shape[0])
        del lg, pr, sel
        if (i // a.chunk) % 20 == 0:
            print(f"  {i}/{len(texts)} docs, {npos} positions", flush=True)

    p = acc / max(npos, 1)
    s = float(p.sum())
    print(f"positions={npos}  sum(p)={s:.6f}  (gate: ~1.0)", flush=True)
    np.save(a.out, p.astype(np.float32))
    Path(a.out + ".meta.json").write_text(json.dumps(
        {"model": a.model, "n_docs": len(texts), "seq_len": a.seq_len,
         "n_positions": npos, "sum_p": s, "vocab": int(V),
         "transformers": transformers.__version__, "torch": torch.__version__}, indent=1))
    print("MODEL_PRIOR_DONE", flush=True)


if __name__ == "__main__":
    main()
