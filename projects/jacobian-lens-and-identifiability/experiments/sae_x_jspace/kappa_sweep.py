"""kappa-stratified sweep of the FULL Goodfire 70B dictionary (base-rate question):
what fraction of all 65,536 features are lens-legible, and where do the six
deception-cluster features sit in that distribution?

No model forward passes — just U (partial shard load), the band J_l's, and W_dec.
For a seed-1 sample of N features (+ the 6 targets + 24 controls), compute
kappa_max over band layers of r = U @ (J_l @ d_f), plus the top readout tokens for
the highest-kappa features (qualitative legibility check).

Run (one 24GB GPU):  python kappa_sweep.py --n 2048 --out kappa_sweep_70b.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "jacobian_lens"))

SPEC = json.load(open(HERE / "features.json"))
HF_ID = "meta-llama/Llama-3.3-70B-Instruct"
SAE = ("Goodfire/Llama-3.3-70B-Instruct-SAE-l50", "Llama-3.3-70B-Instruct-SAE-l50.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", default=str(HERE / "kappa_sweep_70b.json"))
    args = ap.parse_args()

    import jlens
    import transformers
    from huggingface_hub import hf_hub_download
    from cka_layers import REPO, resolve
    from unembed import load_unembedding

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, lens_file = resolve("llama3.3-70b-it")
    lens = jlens.JacobianLens.load(hf_hub_download(REPO, filename=lens_file))
    src = lens.source_layers
    band = src[len(src) // 3: 2 * len(src) // 3]
    tok = transformers.AutoTokenizer.from_pretrained(HF_ID)
    U = load_unembedding(HF_ID).to(dev)          # fp32 [V, d] straight to GPU
    sd = torch.load(hf_hub_download(*SAE), map_location="cpu", weights_only=True)
    W_dec = sd["decoder_linear.weight"].float().clone()  # [d, F]; drop encoder + bias
    del sd
    F = W_dec.shape[1]

    targets = [f["id"] for f in SPEC["targets"]]
    controls = SPEC["index_adjacent_controls"]
    g = torch.Generator().manual_seed(1)
    sample = sorted(set(torch.randperm(F, generator=g)[: args.n].tolist())
                    | set(targets) | set(controls))
    print(f"features: {len(sample)} (sample {args.n} + targets + controls); "
          f"band {band[0]}..{band[-1]} ({len(band)} layers); V={U.shape[0]}")

    kmax = torch.zeros(len(sample))
    top1_at_peak = [""] * len(sample)
    D = W_dec[:, sample].to(dev)                 # [d, S]
    for l in band:
        J = lens.jacobians[l].to(dev).float()    # [d, d]
        JD = J @ D                               # [d, S]
        for i0 in range(0, D.shape[1], args.batch):
            R = U @ JD[:, i0:i0 + args.batch]    # [V, b]
            mu = R.mean(0, keepdim=True)
            sdv = R.std(0, keepdim=True) + 1e-9
            k = (((R - mu) / sdv) ** 4).mean(0).cpu()  # [b]
            better = k > kmax[i0:i0 + args.batch]
            if better.any():
                tops = R.argmax(0).cpu()
                for j in torch.nonzero(better).flatten().tolist():
                    top1_at_peak[i0 + j] = tok.decode([int(tops[j])])
            kmax[i0:i0 + args.batch] = torch.maximum(kmax[i0:i0 + args.batch], k)
        del J, JD
        lens.jacobians[l] = None  # free as we go — RAM headroom
        print(f"  layer {l} done")

    ks = kmax.tolist()
    order = sorted(range(len(sample)), key=lambda i: -ks[i])
    qs = torch.tensor(ks).quantile(torch.tensor([.5, .75, .9, .95, .99])).tolist()
    tgt_pct = {t: float((kmax < ks[sample.index(t)]).float().mean()) for t in targets}
    out = {
        "n_features": len(sample), "band": band,
        "kappa_quantiles": dict(zip(["p50", "p75", "p90", "p95", "p99"], qs)),
        "frac_above_10": float((kmax > 10).float().mean()),
        "frac_above_20": float((kmax > 20).float().mean()),
        "targets": {t: {"kappa": ks[sample.index(t)], "pctile": tgt_pct[t],
                        "top1": top1_at_peak[sample.index(t)]} for t in targets},
        "top50": [{"f": sample[i], "kappa": round(ks[i], 1),
                   "top1": top1_at_peak[i]} for i in order[:50]],
        "per_feature": {sample[i]: round(ks[i], 2) for i in range(len(sample))},
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"quantiles {out['kappa_quantiles']}")
    print(f"frac kappa>10: {out['frac_above_10']:.3f}  >20: {out['frac_above_20']:.3f}")
    for t, v in out["targets"].items():
        print(f"  target f{t}: kappa={v['kappa']:.1f} pct={v['pctile']:.2f} top1={v['top1']!r}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
