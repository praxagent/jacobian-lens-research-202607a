"""What does the invariant readout subspace encode? Frozen design in PREREG.md.

Per-token effect of eigendirection k:  e_k[t] = mean_l (U (J_l w_k))[t]
Regressed on seven pre-named features. Adjusted R^2 decides SETTLED / PARTLY / UNEXPLAINED
against thresholds frozen before any run.
"""
from __future__ import annotations
import argparse, json, string
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
K_DIRECTIONS = 4
FEATURES = ["model_prior", "log_unigram", "is_unused", "embed_norm",
            "word_initial", "tok_len", "is_punct"]


def adj_r2(y, X):
    """OLS with intercept; returns adjusted R^2 and standardised betas."""
    Xz = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    A = np.column_stack([np.ones(len(y)), Xz])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_res = float(resid @ resid); ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / max(ss_tot, 1e-30)
    n, p = len(y), X.shape[1]
    return 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1), beta[1:]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prior", required=True)
    ap.add_argument("--out", default=str(HERE / "results.json"))
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    from datasets import load_dataset

    # ---- directions ----
    A_ = np.load(DEC / "gemma2_9b_A.npy")
    ev, evec = np.linalg.eigh(A_)
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    shares = (ev / ev.sum())[:K_DIRECTIONS]
    print(f"top-{K_DIRECTIONS} energy shares: {np.round(shares,4).tolist()} "
          f"(sum {shares.sum():.3f})", flush=True)

    d = torch.load(LENS, map_location="cpu", weights_only=False)
    Jd = d["J"]; layers = sorted(Jd.keys())
    Vmat = {}
    for k in range(K_DIRECTIONS):
        w = evec[:, k] / np.linalg.norm(evec[:, k])
        Vmat[k] = np.stack([Jd[l].float().numpy() @ w for l in layers]).astype(np.float32)

    # ---- per-token effects + embedding norm, one pass over U ----
    U = np.load(DEC / "gemma2_9b_embed.npy", mmap_mode="r")
    vocab, dim = U.shape
    eff = {k: np.zeros(vocab, dtype=np.float64) for k in range(K_DIRECTIONS)}
    embed_norm = np.zeros(vocab, dtype=np.float32)
    for i in range(0, vocab, CHUNK):
        c = np.asarray(U[i:i+CHUNK], dtype=np.float32)
        embed_norm[i:i+CHUNK] = np.sqrt((c*c).sum(1))
        for k in range(K_DIRECTIONS):
            eff[k][i:i+CHUNK] = (c @ Vmat[k].T).mean(1)
    print("per-token effects computed", flush=True)

    # ---- features ----
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    cnt, n = Counter(), 0
    for t in ds["text"]:
        if len(t) < 50: continue
        cnt.update(tok.encode(t, add_special_tokens=False)); n += 1
        if n >= N_FREQ_DOCS: break
    freq = np.zeros(vocab);
    for kk, vv in cnt.items():
        if kk < vocab: freq[kk] = vv
    seen = freq > 0
    log_unigram = np.log10(freq + 1.0)

    prior = np.load(a.prior)
    if prior.size != vocab:
        raise SystemExit(f"prior size {prior.size} != vocab {vocab}")
    model_prior = np.log10(prior.astype(np.float64) + 1e-12)

    toks = tok.convert_ids_to_tokens(list(range(vocab)))
    is_unused = np.array([1.0 if (t is None or "unused" in t.lower() or
                                  (t.startswith("<") and t.endswith(">"))) else 0.0
                          for t in toks])
    word_initial = np.array([1.0 if (t or "").startswith("▁") else 0.0 for t in toks])
    tok_len = np.array([len((t or "").replace("▁", "")) for t in toks], dtype=float)
    punct = set(string.punctuation)
    is_punct = np.array([1.0 if (t and all(ch in punct for ch in t.replace("▁", "")) and
                                 t.replace("▁", "")) else 0.0 for t in toks])
    feats = {"model_prior": model_prior, "log_unigram": log_unigram, "is_unused": is_unused,
             "embed_norm": embed_norm.astype(np.float64), "word_initial": word_initial,
             "tok_len": tok_len, "is_punct": is_punct}
    print(f"features built; {int(seen.sum())}/{vocab} tokens seen, "
          f"{int(is_unused.sum())} unused/special", flush=True)

    # ---- gate: reproduce the previously reported correlation ----
    e0 = eff[0]
    e0s = e0 if abs(e0.max()) >= abs(e0.min()) else -e0
    prev = float(np.corrcoef(e0s[seen], log_unigram[seen])[0, 1])
    gate = abs(prev - (-0.591)) <= 0.02
    print(f"GATE reproduce corr(top dir, log_unigram) = {prev:+.4f} "
          f"(expected -0.591) -> {'PASS' if gate else 'FAIL'}", flush=True)

    out = {"energy_shares": shares.tolist(), "n_seen": int(seen.sum()),
           "gate_reproduce_corr": prev, "gate_pass": bool(gate), "directions": {}}
    for k in range(K_DIRECTIONS):
        y = eff[k][seen]
        y = y if abs(y.max()) >= abs(y.min()) else -y
        X = np.column_stack([feats[f][seen] for f in FEATURES])
        r2, betas = adj_r2(y, X)
        # single-feature reference: model prior alone vs unigram alone
        r2_prior, _ = adj_r2(y, feats["model_prior"][seen][:, None])
        r2_uni, _ = adj_r2(y, feats["log_unigram"][seen][:, None])
        out["directions"][str(k)] = {
            "energy_share": float(shares[k]), "adj_r2_full": float(r2),
            "adj_r2_model_prior_alone": float(r2_prior),
            "adj_r2_log_unigram_alone": float(r2_uni),
            "betas": {f: float(b) for f, b in zip(FEATURES, betas)}}
        print(f"  dir {k} (share {shares[k]:.3f}): adj R2 = {r2:.4f}  "
              f"[prior alone {r2_prior:.3f}, unigram alone {r2_uni:.3f}]", flush=True)
        print(f"      betas: " + "  ".join(f"{f}={b:+.3f}" for f, b in zip(FEATURES, betas)), flush=True)

    top = out["directions"]["0"]["adj_r2_full"]
    out["verdict"] = ("SETTLED" if top >= 0.80 else
                      "PARTLY SETTLED" if top >= 0.50 else "UNEXPLAINED")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nVERDICT (top direction adj R2 = {top:.4f}): {out['verdict']}")
    print("DECOMPOSE_DONE", flush=True)


if __name__ == "__main__":
    main()
