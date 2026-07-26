"""v2 decomposition: is direction 0's residual a missing feature or a missing functional form?

Design frozen in PREREG_V2.md. Four nested models:
  M0 = the v1 seven features (must reproduce adj R^2 0.572 or VOID)
  M1 = M0 + script/byte features (H1)
  M2 = M0 + model_prior entered as 20 equal-count bins (H2)
  M3 = M0 + H1 + H2

CPU only, RAM-frugal: the embedding matrix is memory-mapped and swept once in chunks.
"""
from __future__ import annotations
import argparse, json, string, unicodedata
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ATLAS = HERE.parents[0].parent / "experiments/jspace_atlas"
DEC = ATLAS / "decompose_out"
LENS = "/home/ubuntu/hf-lenses-tier1/gemma2_9b_wiki.pt"
MODEL = "google/gemma-2-9b"
CHUNK = 8192
N_FREQ_DOCS = 20000
V1_FEATURES = ["model_prior", "log_unigram", "is_unused", "embed_norm",
               "word_initial", "tok_len", "is_punct"]
H1_FEATURES = ["is_non_latin", "is_cjk", "is_cyrillic_or_greek", "bytes_per_char",
               "is_continuation"]
N_BINS = 20
V1_TARGET_R2, V1_TOL = 0.5717, 0.01
GATE_CORR, GATE_TOL = -0.591, 0.02


def adj_r2(y, X):
    Xz = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    A = np.column_stack([np.ones(len(y)), Xz])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    r2 = 1 - float(resid @ resid) / max(float(((y - y.mean()) ** 2).sum()), 1e-30)
    n, p = len(y), X.shape[1]
    return 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1), beta[1:]


def script_flags(t: str):
    """(non_latin, cjk, cyrillic_or_greek) for one token string."""
    s = (t or "").replace("▁", "")
    nl = cjk = cg = 0
    for ch in s:
        o = ord(ch)
        if o > 0x7F:
            nl = 1
        if (0x4E00 <= o <= 0x9FFF or 0x3040 <= o <= 0x30FF or 0xAC00 <= o <= 0xD7AF
                or 0x3400 <= o <= 0x4DBF):
            cjk = 1
        if 0x0400 <= o <= 0x04FF or 0x0370 <= o <= 0x03FF:
            cg = 1
    return nl, cjk, cg


def binned(x, nbins, seen_mask=None):
    """Equal-count bins as dummy columns, first bin dropped as the reference."""
    q = np.quantile(x, np.linspace(0, 1, nbins + 1)[1:-1])
    idx = np.searchsorted(q, x, side="right")
    cols = np.zeros((len(x), nbins - 1))
    for b in range(1, nbins):
        cols[:, b - 1] = (idx == b).astype(float)
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True)
    ap.add_argument("--out", default=str(HERE / "results_v2.json"))
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    from datasets import load_dataset

    A_ = np.load(DEC / "gemma2_9b_A.npy")
    ev, evec = np.linalg.eigh(A_)
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    w = evec[:, 0] / np.linalg.norm(evec[:, 0])

    d = torch.load(LENS, map_location="cpu", weights_only=False)
    Jd = d["J"]; layers = sorted(Jd.keys())
    V = np.stack([Jd[l].float().numpy() @ w for l in layers]).astype(np.float32)
    del d, Jd
    print(f"direction 0 recovered (energy share {ev[0]/ev.sum():.4f})", flush=True)

    U = np.load(DEC / "gemma2_9b_embed.npy", mmap_mode="r")
    vocab, dim = U.shape
    eff = np.zeros(vocab, dtype=np.float64)
    embed_norm = np.zeros(vocab, dtype=np.float32)
    Vt = V.T.astype(np.float32)
    for i in range(0, vocab, CHUNK):
        c = np.asarray(U[i:i+CHUNK], dtype=np.float32)
        embed_norm[i:i+CHUNK] = np.sqrt((c*c).sum(1))
        eff[i:i+CHUNK] = (c @ Vt).mean(1)
    del U
    print("per-token effects computed", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    cnt, n = Counter(), 0
    for t in ds["text"]:
        if len(t) < 50: continue
        cnt.update(tok.encode(t, add_special_tokens=False)); n += 1
        if n >= N_FREQ_DOCS: break
    freq = np.zeros(vocab)
    for k, v in cnt.items():
        if k < vocab: freq[k] = v
    seen = freq > 0
    log_unigram = np.log10(freq + 1.0)

    prior = np.load(a.prior)
    model_prior = np.log10(prior.astype(np.float64) + 1e-12)

    toks = tok.convert_ids_to_tokens(list(range(vocab)))
    is_unused = np.array([1.0 if (t is None or "unused" in t.lower() or
                                  (t.startswith("<") and t.endswith(">"))) else 0.0
                          for t in toks])
    word_initial = np.array([1.0 if (t or "").startswith("▁") else 0.0 for t in toks])
    tok_len = np.array([len((t or "").replace("▁", "")) for t in toks], float)
    punct = set(string.punctuation)
    is_punct = np.array([1.0 if (t and all(ch in punct for ch in t.replace("▁", ""))
                                 and t.replace("▁", "")) else 0.0 for t in toks])
    sf = np.array([script_flags(t) for t in toks], float)
    bpc = np.array([len((t or "").replace("▁", "").encode("utf-8"))
                    / max(len((t or "").replace("▁", "")), 1) for t in toks], float)

    F = {"model_prior": model_prior, "log_unigram": log_unigram, "is_unused": is_unused,
         "embed_norm": embed_norm.astype(np.float64), "word_initial": word_initial,
         "tok_len": tok_len, "is_punct": is_punct,
         "is_non_latin": sf[:, 0], "is_cjk": sf[:, 1], "is_cyrillic_or_greek": sf[:, 2],
         "bytes_per_char": bpc, "is_continuation": 1.0 - word_initial}

    y = eff[seen]
    y = y if abs(y.max()) >= abs(y.min()) else -y
    corr = float(np.corrcoef(y, log_unigram[seen])[0, 1])
    gate_corr = abs(corr - GATE_CORR) <= GATE_TOL
    print(f"GATE corr(dir0, log_unigram) = {corr:+.4f} (expect {GATE_CORR}) -> "
          f"{'PASS' if gate_corr else 'FAIL'}", flush=True)

    X0 = np.column_stack([F[f][seen] for f in V1_FEATURES])
    X1 = np.column_stack([X0] + [F[f][seen] for f in H1_FEATURES])
    Xb = binned(F["model_prior"][seen], N_BINS)
    X2 = np.column_stack([X0, Xb])
    X3 = np.column_stack([X1, Xb])

    r = {}
    for name, X in (("M0", X0), ("M1", X1), ("M2", X2), ("M3", X3)):
        v, betas = adj_r2(y, X)
        r[name] = {"adj_r2": float(v), "n_features": int(X.shape[1])}
        print(f"  {name}: adj R2 = {v:.4f}  ({X.shape[1]} features)", flush=True)
    for f, b in zip(H1_FEATURES, adj_r2(y, X1)[1][len(V1_FEATURES):]):
        r.setdefault("H1_betas", {})[f] = float(b)

    gate_m0 = abs(r["M0"]["adj_r2"] - V1_TARGET_R2) <= V1_TOL
    print(f"GATE M0 reproduces v1 ({V1_TARGET_R2}) -> {'PASS' if gate_m0 else 'FAIL'}", flush=True)

    h1 = r["M1"]["adj_r2"] - r["M0"]["adj_r2"]
    h2 = r["M2"]["adj_r2"] - r["M0"]["adj_r2"]
    overlap = r["M1"]["adj_r2"] + r["M2"]["adj_r2"] - r["M0"]["adj_r2"] - r["M3"]["adj_r2"]
    top = r["M3"]["adj_r2"]
    out = {"gates": {"corr": corr, "corr_pass": bool(gate_corr),
                     "m0_reproduces_v1": bool(gate_m0), "void": not (gate_corr and gate_m0)},
           "models": r, "n_seen": int(seen.sum()),
           "H1_gain": float(h1), "H1_supported": bool(h1 >= 0.05),
           "H2_gain": float(h2), "H2_supported": bool(h2 >= 0.05),
           "overlap": float(overlap),
           "verdict": ("SETTLED" if top >= 0.80 else
                       "PARTLY SETTLED" if top >= 0.50 else "UNEXPLAINED")}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nH1 (script/byte features) gain = {h1:+.4f} -> "
          f"{'SUPPORTED' if out['H1_supported'] else 'not supported'}")
    print(f"H2 (nonlinear model_prior) gain = {h2:+.4f} -> "
          f"{'SUPPORTED' if out['H2_supported'] else 'not supported'}")
    print(f"overlap between them = {overlap:+.4f}")
    print(f"VERDICT (M3 adj R2 = {top:.4f}): {out['verdict']}"
          + ("   [VOID: a gate failed]" if out["gates"]["void"] else ""))
    print("DECOMPOSE_V2_DONE", flush=True)


if __name__ == "__main__":
    main()
