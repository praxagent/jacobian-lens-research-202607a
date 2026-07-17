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


def load_jury_examples(jury_path, cache_path, tokenizer, limit=None):
    """Amendment 6 (pre-registered before the GPU run, 2026-07-17): jury-labeled
    Gemma-3-12B arm. Examples = arm-4's cached greedy completions + the jury receipt's
    definitive labels (majority+ tier; split-tier and Insufficient rows dropped).
    Labels are jury agreement (kappa .598 majority vs public gold), NOT ground truth.
    Split rule, fixed before results: completion index % 5 -> {0,1,2} train,
    {3} validation, {4} test (B09 roles unchanged).
    """
    from collections import defaultdict
    comps = {}
    for line in Path(cache_path).read_text().splitlines():
        o = json.loads(line)
        comps[o["i"]] = o["c"]
    by_comp = defaultdict(list)
    for r in json.loads(Path(jury_path).read_text())["rows"]:
        if r["jury_label"] is None or r["tier"] == "split":
            continue
        by_comp[int(r["cid"].split(":")[1])].append(r)
    roles = {0: "train", 1: "train", 2: "train", 3: "validation", 4: "test"}
    out = {"train": [], "validation": [], "test": []}
    dropped = 0
    for i in sorted(by_comp)[: limit if limit else None]:
        c = comps.get(i)
        if c is None:
            continue
        ex = D.Example(f"g3:{i}", "", c, [])
        for r in by_comp[i]:
            e = r["entity"].strip()
            pos = c.find(e)
            if pos <= 0:          # same locate rule as arm-4's extraction
                dropped += 1
                continue
            ex.spans.append(D.Span(pos, pos + len(e), e,
                                   1 if r["jury_label"] == "Not Supported" else 0))
        if ex.spans:
            out[roles[i % 5]].append(D.align_spans(ex, tokenizer))
    print(f"  jury examples: "
          + ", ".join(f"{k}={len(v)} comps" for k, v in out.items())
          + f", dropped(unlocatable)={dropped}", flush=True)
    return out


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


def paired_reader_contrasts(readers_signed, probe_name, labels, cids, margin=0.05,
                            n=2000, seed=0):
    """B06: paired completion-clustered bootstrap of (reader - probe) AUROC differences,
    SHARED resamples across readers. Frozen decision rules on the paired 95% CI:
      equivalent  : CI entirely within [-margin, +margin]
      noninferior : CI lower bound > -margin
      worse       : CI upper bound < -margin
      better      : CI lower bound > +margin
    """
    rng = np.random.default_rng(seed)
    by = {}
    for i, c in enumerate(cids):
        by.setdefault(c, []).append(i)
    groups = list(by.values())
    labels = np.asarray(labels)
    names = [k for k in readers_signed if k != probe_name]
    diffs = {k: [] for k in names}
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = [i for g in pick for i in groups[g]]
        yb = labels[idx]
        pa = _np_auroc(np.asarray(readers_signed[probe_name])[idx], yb)
        if np.isnan(pa):
            continue
        for k in names:
            a = _np_auroc(np.asarray(readers_signed[k])[idx], yb)
            if not np.isnan(a):
                diffs[k].append(a - pa)
    out = {}
    for k, v in diffs.items():
        if not v:
            out[k] = {"verdict": "undefined"}
            continue
        lo, hi = np.percentile(v, [2.5, 97.5])
        verdict = ("equivalent" if (lo > -margin and hi < margin)
                   else "better" if lo > margin
                   else "worse" if hi < -margin
                   else "noninferior" if lo > -margin
                   else "inconclusive")
        out[k] = {"mean_diff": round(float(np.mean(v)), 4),
                  "ci95": [round(float(lo), 4), round(float(hi), 4)],
                  "margin": margin, "verdict": verdict}
    return out


def _np_auroc(s, y):
    return R.auroc(s, y)


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
    if args.device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, token=tok_hf, torch_dtype=torch.bfloat16,
            device_map="auto", output_hidden_states=True)
        args.device = "cuda:0"   # readers/probe device; hiddens are CPU-offloaded anyway
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, token=tok_hf,
            torch_dtype=torch.float32 if args.device == "cpu" else torch.bfloat16,
            output_hidden_states=True).to(args.device)
    unembed, final_norm = A.model_readout_parts(model)
    n_layers = getattr(model.config, "num_hidden_layers", None) or \
        getattr(model.config, "n_layer", None) or \
        model.config.text_config.num_hidden_layers   # Gemma-3 nests under text_config
    if n_layers not in layers:
        layers = sorted(layers + [n_layers])   # head state for the native reader (I02)
    unembed = unembed.to(args.device).float()

    # B09 split roles: train = parameter fitting; validation = ALL choices (sign, SAE
    # latent, epoch); test = one final evaluation. In smoke, one split is partitioned 3-way.
    splits = {}
    if args.smoke:
        exs = D.load_examples(config, "validation", limit=limit, hf_token=tok_hf)
        exs = [D.align_spans(e, tokenizer) for e in exs]
        recs = A.cache_spans(model, tokenizer, exs, layers, device=args.device, max_len=512)
        third = max(1, len(recs) // 3)
        splits = {"train": recs[:third], "validation": recs[third:2 * third],
                  "test": recs[2 * third:]}
        print(f"  smoke 3-way partition: {[len(splits[s]) for s in ('train','validation','test')]} spans")
    elif args.jury_labels:
        exs_by_split = load_jury_examples(args.jury_labels, args.completions_cache,
                                          tokenizer, limit=limit)
        config = "jury:gemma3"
        # Amendment 6b (numeric bug fix, 2026-07-17): Gemma-3 residuals reach |h| >3e5
        # at layer 24 — over float16 max (65504) — so the default fp16 activation cache
        # clips to inf and the probe trains to NaN (first pod run: all seeds AUROC
        # exactly .5, receipt receipt_gemma3_jury.json kept as the honest record).
        cache_dt = getattr(torch, args.cache_dtype)
        for split, exs in exs_by_split.items():
            splits[split] = A.cache_spans(model, tokenizer, exs, layers,
                                          device=args.device, max_len=args.max_len,
                                          dtype=cache_dt)
            bad = sum((~torch.isfinite(r["hid"][L])).sum().item()
                      for r in splits[split] for L in r["hid"])
            assert bad == 0, f"{bad} non-finite cached activations in {split} " \
                             f"(cache dtype {args.cache_dtype} too narrow for this model)"
            print(f"  {split}: {len(exs)} completions, {len(splits[split])} spans, "
                  f"balance={D.label_balance(exs)['prevalence']:.3f}", flush=True)
    else:
        for split in ("train", "validation", "test"):
            exs = D.load_examples(config, split, limit=limit, hf_token=tok_hf)
            exs = [D.align_spans(e, tokenizer) for e in exs]
            recs = A.cache_spans(model, tokenizer, exs, layers, device=args.device,
                                 max_len=args.max_len)
            splits[split] = recs
            print(f"  {split}: {len(exs)} completions, {len(recs)} spans, "
                  f"balance={D.label_balance(exs)['prevalence']:.3f}")

    tr, va, te = splits["train"], splits["validation"], splits["test"]
    y_va = np.array([r["label"] for r in va])
    y_te = np.array([r["label"] for r in te])
    cid_te = [r["cid"] for r in te]

    # --- supervised attention probe: mean of 3 independently-trained seeds (I06) ---
    # A single seed can be atypically (un)favorable for a small attention probe; the
    # primary score is the MEAN of independently trained probes (never best-seed), so the
    # CI reflects optimization variability, not just the completion bootstrap.
    Xtr, Mtr = pad_spans(tr, primary)
    ytr = np.array([r["label"] for r in tr])
    Xte, Mte = pad_spans(te, primary)
    # Amendment 4 (pre-outcome): mini-batched GPU probe train/eval; tensors stay on CPU,
    # batches stream to the device (a 41.7GB full-tensor .to(cuda) OOMed 3x). Free the
    # base model first -- only unembed + final_norm are needed after activation caching
    # (nn.Module.to is in-place, so the final_norm reference stays valid).
    import gc
    model.cpu(); del model; gc.collect()
    if args.device != "cpu":
        torch.cuda.empty_cache()
        if final_norm is not None:
            final_norm.to(args.device)
    print("  [stage] probe training", flush=True)
    seed_scores = []
    for s in args.probe_seeds:
        probe, _ = R.train_probe(Xtr, Mtr, ytr, d_model=Xtr.shape[-1], n_heads=args.heads,
                                 epochs=args.epochs, lr=args.lr, seed=s, device=args.device)
        seed_scores.append(R.eval_probe(probe, Xte, Mte, device=args.device))
    probe_scores = np.mean(seed_scores, axis=0)
    probe_seed_aurocs = [round(float(R.auroc(ss, y_te)), 4) for ss in seed_scores]

    # EXPLORATORY (post-outcome amendment, TJ 2026-07-16): paper-capacity probe --
    # 16 heads, 400 epochs. Reported separately; never replaces the frozen primary.
    exploratory_capacity = None
    if args.exploratory_capacity:
        print("  [stage] exploratory capacity probe", flush=True)
        xs = []
        for s in args.probe_seeds:
            p2, _ = R.train_probe(Xtr, Mtr, ytr, d_model=Xtr.shape[-1], n_heads=16,
                                  epochs=400, lr=args.lr, seed=s, device=args.device)
            xs.append(R.eval_probe(p2, Xte, Mte, device=args.device))
        xsc = np.mean(xs, axis=0)
        lo_x, hi_x = cluster_bootstrap_auroc(xsc, y_te, cid_te, n=(200 if args.smoke else 2000))
        exploratory_capacity = {"label": "exploratory (post-outcome capacity test)",
                                "n_heads": 16, "epochs": 400,
                                "seed_aurocs": [round(float(R.auroc(x, y_te)), 4) for x in xs],
                                "auroc": round(float(R.auroc(xsc, y_te)), 4),
                                "ci95": [round(lo_x, 4), round(hi_x, 4)]}

    # --- non-neural heuristic baseline (I04): entity token-count + in-fold unigram
    # log-frequency, logistic, fit on TRAIN only. Frequency corpus = the train-split
    # tokens (self-contained; pin an external corpus at freeze). ---
    print("  [stage] heuristic + lens readers", flush=True)
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
    # Functional final-norm (amendment 5): under device_map=auto + accelerate hooks the
    # norm module resists .to() (weight pinned to its shard's GPU -> cross-device crash).
    # Snapshot weight/bias/eps once and apply functionally on the span's device.
    if final_norm is not None:
        _w = final_norm.weight.detach().float().to(dev)
        _b = getattr(final_norm, "bias", None)
        _b = _b.detach().float().to(dev) if _b is not None else None
        _eps = getattr(final_norm, "variance_epsilon", getattr(final_norm, "eps", 1e-6))
        if _b is None and not isinstance(final_norm, torch.nn.LayerNorm):   # RMSNorm
            def _fn(h):
                h = h.to(dev)
                return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + _eps) * _w
        else:
            import torch.nn.functional as _F

            def _fn(h):
                h = h.to(dev)
                return _F.layer_norm(h, (h.shape[-1],), _w, _b, _eps)
    else:
        def _fn(h):
            return h.to(dev)

    d = Xtr.shape[-1]

    def lens_on(recs, transport=None):
        T = transport.to(dev).float() if transport is not None else None
        out = []
        for r in recs:
            hid = {primary: A.span_hidden(r, primary).to(dev)}
            fn = _fn
            out.append(R.jlens_score(hid, ue, fn, slice(None), r["tok_ids"].to(dev),
                                     primary, transport=T) if T is not None
                       else R.logit_lens_score(hid, ue, fn, slice(None),
                                               r["tok_ids"].to(dev), primary))
        return np.array(out)

    rng = torch.Generator().manual_seed(0)
    randT = torch.randn(d, d, generator=rng)
    randT *= (torch.eye(d).norm() / randT.norm())          # Frobenius-matched null

    # native output-token surprisal (I02): read at the model's ACTUAL last hidden state
    # (post-final-norm in HF, so head_layer skips re-norm -- gate G1). The head index is
    # the layer count, NOT max(layers); it is force-included in the cached layers below.
    head_idx = n_layers

    def native_on(recs):
        return np.array([R.logit_lens_score(
            {head_idx: A.span_hidden(r, head_idx).to(dev)}, ue, _fn, slice(None),
            r["tok_ids"].to(dev), head_idx, head_layer=head_idx) for r in recs])

    # readers scored on BOTH validation (for sign/selection) and test (final)
    readers = {"attention_probe": ("supervised", None, probe_scores)}
    for name, T in (("logit_lens", None), ("random_transport_null", randT)):
        readers[name] = ("unsupervised" if T is None else "null",
                         lens_on(va, T), lens_on(te, T))
    readers["native_head_surprisal"] = ("unsupervised", native_on(va), native_on(te))
    readers["heuristic_len_freq"] = ("heuristic", None, heur_scores)  # fit in-fold already

    # J-lens reader (fitted transport) if provided
    if args.jlens_path:
        J = torch.load(args.jlens_path, map_location="cpu", weights_only=True)
        J = J[primary] if isinstance(J, dict) else J
        readers["jacobian_lens"] = ("unsupervised", lens_on(va, J.float()), lens_on(te, J.float()))

    # label-selected SAE reader (B05: selection on VALIDATION labels only)
    print("  [stage] SAE selection", flush=True)
    sae_info = None
    if args.sae_path:
        import sae as S
        sae_r = (S.load_gemmascope(args.sae_path) if args.sae_format == "gemmascope"
                 else S.load_goodfire(args.sae_path, d))
        va_h = [A.span_hidden(r, primary) for r in va]
        te_h = [A.span_hidden(r, primary) for r in te]
        li, sgn, va_a = S.select_latent(sae_r, va_h, y_va, R.auroc)
        sae_te = np.array([sgn * sae_r.encode(h)[:, li].mean().item() for h in te_h])
        readers["sae_latent_label_selected"] = ("label-selected", None, sae_te)
        sae_info = {"latent": int(li), "sign": int(sgn), "val_auroc": round(float(va_a), 4),
                    "n_latents": sae_r.n_latents, "format": args.sae_format}

    table = {}
    signed_all = {}
    for name, (kind, sc_va, sc_te) in readers.items():
        sc_te = np.asarray(sc_te, dtype=float)
        if kind in ("supervised", "heuristic", "label-selected"):
            signed = sc_te                       # sign fixed by training/selection fold
        else:
            # B08 mechanical sign rule: direction fixed on VALIDATION, never test
            flip = R.auroc(np.asarray(sc_va, dtype=float), y_va) < 0.5
            signed = -sc_te if flip else sc_te
        signed_all[name] = signed
        a = R.auroc(signed, y_te)
        lo, hi = cluster_bootstrap_auroc(signed, y_te, cid_te,
                                         n=(200 if args.smoke else 2000))
        table[name] = {"kind": kind, "auroc": round(float(a), 4),
                       "ci95": [round(lo, 4), round(hi, 4)]}

    # B06 primary contrasts: paired reader-minus-probe diffs, shared cluster resamples,
    # frozen +/-0.05 equivalence margin
    contrasts = paired_reader_contrasts(signed_all, "attention_probe", y_te, cid_te,
                                        margin=0.05, n=(200 if args.smoke else 2000))

    receipt = {
        "what": "Tier-1 probe-calibration + reader comparison (smoke)" if args.smoke
                else "Tier-1 probe-calibration + reader comparison",
        "model": model_name, "dataset_config": config, "layers": layers,
        "primary_layer": primary, "n_train_spans": len(tr), "n_test_spans": len(te),
        "test_prevalence": round(float(y_te.mean()), 4),
        "probe_seed_aurocs": probe_seed_aurocs,
        "readers": table,
        "paired_contrasts_vs_probe": contrasts,
        "exploratory_probe_capacity": exploratory_capacity,
        "sae_selection": sae_info,
        "per_span_test": [dict({"cid": r["cid"], "label": int(r["label"])},
                               **{name: float(np.asarray(readers[name][2])[i])
                                  for name in readers}) for i, r in enumerate(te)],
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
    ap.add_argument("--exploratory-capacity", action="store_true")
    ap.add_argument("--sae-path", default=None, help=".pth (goodfire) or params.safetensors (gemmascope)")
    ap.add_argument("--sae-format", choices=["goodfire", "gemmascope"], default="goodfire")
    ap.add_argument("--jlens-path", default=None, help="transport matrix .pt (or {layer: J} dict)")
    ap.add_argument("--jury-labels", default=None,
                    help="jury receipt JSON (amendment 6: jury-labeled Gemma-3 arm)")
    ap.add_argument("--completions-cache", default=None,
                    help="completions_cache.jsonl matching --jury-labels")
    ap.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16",
                    help="activation-cache dtype; float32 for models whose residuals "
                         "exceed fp16 range (Gemma-3: |h|>3e5 at layer 24)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--out", default="projects/features-as-rewards-replication/"
                                     "experiments/probe_calibration/receipts/smoke.json")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
