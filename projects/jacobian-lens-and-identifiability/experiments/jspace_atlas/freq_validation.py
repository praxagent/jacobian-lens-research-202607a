"""Does the invariant readout direction actually track TOKEN FREQUENCY?

The note currently says the direction is "consistent with" a unigram/frequency axis on the
strength of a 0.48 correlation with **embedding norm**, which is only a proxy for frequency. We
flagged that calling it "the unigram prior" needs a real frequency test. This is that test.

  1. recover the invariant readout direction w (top eigenvector of A = mean_l J_l^T M J_l)
  2. mean_logit[t] = mean_l (U @ (J_l @ w))[t]      the direction's effect on each token
  3. empirical unigram log-frequency over WikiText, in this model's own tokenizer
  4. correlate. Compare against the embedding-norm proxy we already reported.

RAM-frugal: U memory-mapped, single chunked pass. CPU only, free.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DEC = HERE / "decompose_out"
LENS = "/home/ubuntu/hf-lenses-tier1/gemma2_9b_wiki.pt"
MODEL = "google/gemma-2-9b"
CHUNK = 8192
N_FREQ_DOCS = 20000


def main():
    import torch
    from transformers import AutoTokenizer
    from datasets import load_dataset

    A = np.load(DEC / "gemma2_9b_A.npy")
    ev, evec = np.linalg.eigh(A)
    w = evec[:, int(np.argmax(ev))]
    w = w / np.linalg.norm(w)
    print(f"invariant direction recovered (d={w.size}, top eig share "
          f"{ev.max()/ev.sum():.3f})", flush=True)

    d = torch.load(LENS, map_location="cpu", weights_only=False)
    Jd = d["J"]; layers = sorted(Jd.keys())
    V = np.stack([Jd[l].float().numpy() @ w for l in layers]).astype(np.float32)  # (L, d)

    U = np.load(DEC / "gemma2_9b_embed.npy", mmap_mode="r")
    vocab, dim = U.shape
    mean_logit = np.zeros(vocab, dtype=np.float64)
    unorm = np.zeros(vocab, dtype=np.float32)
    Vt = V.T.astype(np.float32)
    for i in range(0, vocab, CHUNK):
        c = np.asarray(U[i:i+CHUNK], dtype=np.float32)
        mean_logit[i:i+CHUNK] = (c @ Vt).mean(1)
        unorm[i:i+CHUNK] = np.sqrt((c*c).sum(1))
    print("per-token readout computed", flush=True)

    # empirical unigram frequency in this model's own tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    cnt = Counter()
    n = 0
    for t in ds["text"]:
        if len(t) < 50:
            continue
        cnt.update(tok.encode(t, add_special_tokens=False))
        n += 1
        if n >= N_FREQ_DOCS:
            break
    freq = np.zeros(vocab, dtype=np.float64)
    for k, v in cnt.items():
        if k < vocab:
            freq[k] = v
    seen = freq > 0
    print(f"unigram table: {n} docs, {int(freq.sum())} tokens, "
          f"{int(seen.sum())}/{vocab} vocab items seen", flush=True)

    logf = np.log10(freq + 1.0)
    # sign convention identical to the decode: put the strongest tokens on the + tail
    ml = mean_logit if abs(mean_logit.max()) >= abs(mean_logit.min()) else -mean_logit

    def corr(a, b, m=None):
        if m is not None: a, b = a[m], b[m]
        return float(np.corrcoef(a, b)[0, 1])

    res = {
        "n_docs": n, "n_tokens": int(freq.sum()), "vocab": int(vocab),
        "vocab_seen": int(seen.sum()),
        "corr_readout_vs_log_freq_all": corr(ml, logf),
        "corr_readout_vs_log_freq_seen_only": corr(ml, logf, seen),
        "corr_readout_vs_embed_norm_all": corr(ml, unorm.astype(np.float64)),
        "corr_readout_vs_embed_norm_seen_only": corr(ml, unorm.astype(np.float64), seen),
        "corr_logfreq_vs_embed_norm_seen_only": corr(logf, unorm.astype(np.float64), seen),
    }
    # is the readout better explained by frequency than by embedding norm?
    res["frequency_beats_embed_norm"] = (abs(res["corr_readout_vs_log_freq_seen_only"])
                                         > abs(res["corr_readout_vs_embed_norm_seen_only"]))
    (DEC / "freq_validation.json").write_text(json.dumps(res, indent=1))
    print("\n=== FREQUENCY VALIDATION ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("FREQ_VALIDATION_DONE", flush=True)


if __name__ == "__main__":
    main()
