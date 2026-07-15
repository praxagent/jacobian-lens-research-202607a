"""Tier-1 runner: probe calibration + four-reader comparison on gold-labeled entity spans.

  load LongFact++ gold spans -> align to model tokens -> cache residual-stream activations
  -> train the supervised attention probe (train split) -> score the four readers on test
  (attention probe / logit lens / fitted J-lens / SAE latent) + random-transport null
  -> AUROC table (completion-clustered bootstrap CI) -> receipt JSON.

CPU smoke (proves the whole path on real data, no GPU):
  .venv/bin/python experiments/probe_calibration/run.py --smoke
Real Tier-1 (on a GPU pod, once frozen):
  python run.py --model meta-llama/Llama-3.1-8B-Instruct --config Meta-Llama-3.1-8B-Instruct \
                --layers 12 16 20 --primary-layer 16 --device cuda --out receipts/llama31_8b.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "common"))
import readers as R      # noqa: E402
import data as D         # noqa: E402
import acts as A         # noqa: E402


def _hf_token():
    env = os.environ.get("HF_TOKEN")
    if env:
        return env
    p = HERE.parents[4] / "shared/runpod/.env"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def pad_spans(records, layer):
    """[N,Smax,d] float32 + [N,Smax] mask from cached per-span slices at `layer`."""
    hs = [A.span_hidden(r, layer) for r in records]
    S = max(h.shape[0] for h in hs)
    d = hs[0].shape[1]
    X = torch.zeros(len(hs), S, d)
    M = torch.zeros(len(hs), S)
    for i, h in enumerate(hs):
        X[i, : h.shape[0]] = h
        M[i, : h.shape[0]] = 1
    return X, M


def cluster_bootstrap_auroc(scores, labels, cids, n=2000, seed=0):
    """95% CI resampling COMPLETIONS (the independent unit), not spans."""
    rng = np.random.default_rng(seed)
    by = {}
    for i, c in enumerate(cids):
        by.setdefault(c, []).append(i)
    groups = list(by.values())
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = [i for g in pick for i in groups[g]]
        a = R.auroc(np.asarray(scores)[idx], np.asarray(labels)[idx])
        if not np.isnan(a):
            vals.append(a)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"), float("nan"))
    return float(lo), float(hi)


def run(args):
    tok_name = "gpt2" if args.smoke else args.model
    model_name = "gpt2" if args.smoke else args.model
    config = args.config or "Meta-Llama-3.1-8B-Instruct"
    limit = args.limit if args.limit else (12 if args.smoke else None)
    layers = args.layers or ([6] if args.smoke else [12, 16, 20])
    primary = args.primary_layer if args.primary_layer is not None else layers[len(layers) // 2]
    tok_hf = _hf_token()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(tok_name, token=tok_hf)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=tok_hf,
        torch_dtype=torch.float32 if args.device == "cpu" else torch.bfloat16,
        output_hidden_states=True).to(args.device)
    unembed, final_norm = A.model_readout_parts(model)
    unembed = unembed.to(args.device).float()

    splits = {}
    for split in ("train", "test"):
        exs = D.load_examples(config, split if not args.smoke else "validation",
                              limit=limit, hf_token=tok_hf)
        exs = [D.align_spans(e, tokenizer) for e in exs]
        max_len_eff = 512 if args.smoke else args.max_len  # gpt2 caps at 1024 positions
        recs = A.cache_spans(model, tokenizer, exs, layers, device=args.device,
                             max_len=max_len_eff)
        splits[split] = recs
        print(f"  {split}: {len(exs)} completions, {len(recs)} spans, "
              f"balance={D.label_balance(exs)['prevalence']:.3f}")

    tr, te = splits["train"], splits["test"]
    y_te = np.array([r["label"] for r in te])
    cid_te = [r["cid"] for r in te]

    # --- supervised attention probe: mean of 3 independently-trained seeds (I06) ---
    # A single seed can be atypically (un)favorable for a small attention probe; the
    # primary score is the MEAN of independently trained probes (never best-seed), so the
    # CI reflects optimization variability, not just the completion bootstrap.
    Xtr, Mtr = pad_spans(tr, primary)
    ytr = np.array([r["label"] for r in tr])
    Xte, Mte = pad_spans(te, primary)
    seed_scores = []
    for s in args.probe_seeds:
        probe, _ = R.train_probe(Xtr, Mtr, ytr, d_model=Xtr.shape[-1], n_heads=args.heads,
                                 epochs=args.epochs, lr=args.lr, seed=s)
        with torch.no_grad():
            seed_scores.append(torch.sigmoid(probe(Xte, Mte)).numpy())
    probe_scores = np.mean(seed_scores, axis=0)
    probe_seed_aurocs = [round(float(R.auroc(ss, y_te)), 4) for ss in seed_scores]

    # --- non-neural heuristic baseline (I04): entity token-count + in-fold unigram
    # log-frequency, logistic, fit on TRAIN only. Frequency corpus = the train-split
    # tokens (self-contained; pin an external corpus at freeze). ---
    from collections import Counter
    freq = Counter(int(t) for r in tr for t in r["tok_ids"].tolist())
    total = sum(freq.values()) or 1

    def _heur_feats(recs):
        import math
        F = []
        for r in recs:
            ids = r["tok_ids"].tolist()
            n = len(ids)
            lf = np.mean([math.log((freq.get(i, 0) + 1) / total) for i in ids]) if ids else 0.0
            F.append([n, lf])
        return np.array(F, dtype=float)

    Htr, Hte = _heur_feats(tr), _heur_feats(te)
    mu, sd = Htr.mean(0), Htr.std(0) + 1e-9
    w = np.zeros(2); b = 0.0
    Xh = (Htr - mu) / sd
    for _ in range(300):                                   # tiny logistic GD, in-fold
        p = 1 / (1 + np.exp(-(Xh @ w + b)))
        g = Xh.T @ (p - ytr) / len(ytr)
        w -= 0.1 * g; b -= 0.1 * (p - ytr).mean()
    heur_scores = 1 / (1 + np.exp(-(((Hte - mu) / sd) @ w + b)))

    # --- unsupervised / sparse readers on test ---
    # NOTE: lens readout runs on CPU here (fine for the smoke's small d). For a real run
    # at large hidden dim (e.g. 70B d=8192) do the transport on GPU (GPU playbook §2
    # corollary: naive CPU transport of d^2 matrices is pathological). A --device cuda
    # path moves unembed/hid/transport to the accelerator before the matmul.
    dev = args.device
    ue = unembed.to(dev).float()
    fnorm = final_norm  # module already on `dev`

    def _fn(h):
        return fnorm(h.to(dev)).to(dev) if fnorm is not None else h.to(dev)

    def lens_scores(transport=None):
        T = transport.to(dev).float() if transport is not None else None
        s = []
        for r in te:
            hid = {primary: A.span_hidden(r, primary).to(dev)}
            if T is None:
                s.append(R.logit_lens_score(hid, ue, _fn, slice(None), r["tok_ids"].to(dev), primary))
            else:
                s.append(R.jlens_score(hid, ue, _fn, slice(None), r["tok_ids"].to(dev), primary, transport=T))
        return np.array(s)

    d = Xtr.shape[-1]
    logit = lens_scores(None)
    rng = torch.Generator().manual_seed(0)
    randT = torch.randn(d, d, generator=rng)
    randT *= (torch.eye(d).norm() / randT.norm())          # Frobenius-matched null
    randnull = lens_scores(randT)

    readers = {
        "attention_probe": (probe_scores, "supervised"),
        "logit_lens": (logit, "unsupervised"),
        "heuristic_len_freq": (heur_scores, "heuristic"),
        "random_transport_null": (randnull, "null"),
    }

    table = {}
    for name, (sc, kind) in readers.items():
        sc = np.asarray(sc, dtype=float)
        if kind == "supervised":
            signed = sc                                    # probe has an a-priori sign
        else:
            signed = sc if R.auroc(sc, y_te) >= 0.5 else -sc  # sign fixed once
        a = R.auroc(signed, y_te)
        lo, hi = cluster_bootstrap_auroc(signed, y_te, cid_te,
                                         n=(200 if args.smoke else 2000))
        table[name] = {"kind": kind, "auroc": round(float(a), 4),
                       "ci95": [round(lo, 4), round(hi, 4)]}

    receipt = {
        "what": "Tier-1 probe-calibration + reader comparison (smoke)" if args.smoke
                else "Tier-1 probe-calibration + reader comparison",
        "model": model_name, "dataset_config": config, "layers": layers,
        "primary_layer": primary, "n_train_spans": len(tr), "n_test_spans": len(te),
        "test_prevalence": round(float(y_te.mean()), 4),
        "probe_seed_aurocs": probe_seed_aurocs,
        "readers": table,
        "per_span_test": [{"cid": r["cid"], "label": int(r["label"]),
                           "probe": float(probe_scores[i]), "logit_lens": float(logit[i]),
                           "random_null": float(randnull[i])} for i, r in enumerate(te)],
        "args": vars(args), "seconds": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(receipt, indent=1, default=float))
    print(f"\n  reader AUROC (test, n={len(te)} spans):")
    for k, v in table.items():
        print(f"    {k:24s} {v['auroc']:.3f}  CI{v['ci95']}  [{v['kind']}]")
    print(f"  receipt -> {args.out}  ({receipt['seconds']}s)")
    return receipt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="gpt2 + tiny real slice on CPU")
    ap.add_argument("--model", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--primary-layer", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--probe-seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--out", default="projects/features-as-rewards-replication/"
                                     "experiments/probe_calibration/receipts/smoke.json")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
