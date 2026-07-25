"""P4: is the lens direction's advantage dose-dependent? Design frozen in PREREG.md addendum.

Sweeps A(eps) = log(KL_aligned / mean KL_random) across the full doubling ladder, and fits the
scaling exponent k of each arm (log KL = k log eps + c). k ~ 2 is the first-order signature.

Separates the two explanations for the observed dose-dependence:
  - random arm keeps k ~ 2  ->  the advantage genuinely grows with dose (non-first-order)
  - random arm's k collapses ->  it is a numerical floor artifact, no claim

Reuses the frozen prompts and the aligned-direction construction from run_geometry_causality.py.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DOSE_FRACS = [0.0015625, 0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2]
N_RANDOM = 4
KEY_SUFFIXES = ("embed_tokens.weight", "wte.weight", "embed_in.weight", "tok_embeddings.weight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--slug", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=25); ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"],
                    help="bf16 has ~8 mantissa bits, so a perturbation at ~0.3%% of the residual "
                         "norm is comparable to the rounding error of storing the residual "
                         "itself, which pins both arms on a precision floor at small doses "
                         "(see results.md P4). float32 lowers that floor by orders of magnitude.")
    a = ap.parse_args()
    import transformers
    import run_geometry_causality as rg          # reuse probe + lens loading, no duplication

    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dtype = ({"float32": torch.float32, "bfloat16": torch.bfloat16}[a.dtype] if a.dtype != "auto"
             else (torch.float32 if a.device == "cpu" else torch.bfloat16))
    print(f"dtype={dtype}", flush=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(a.device).eval()
    layers = rg.get_layers(model); d = model.config.hidden_size

    blob = json.load(open(HERE / "prompts_frozen.json"))
    texts, idxs = blob["texts"][:a.n_prompts], blob["indices"][:a.n_prompts]
    enc = tok(texts, return_tensors="pt", truncation=True, max_length=a.seq_len,
              padding="max_length", padding_side="left").to(a.device)
    B = enc.input_ids.shape[0]
    print(f"{B} frozen prompts, d={d}", flush=True)

    Jd, lens_layers = rg.load_lens_J(a.slug)
    M, n_probe = rg.probe_M(a.slug, a.model, d)

    def fwd(chunk_patch=None):
        outs = []
        for i in range(0, B, a.chunk):
            sub = {k: v[i:i+a.chunk] for k, v in enc.items()}
            hs = []
            if chunk_patch is not None:
                l, vec = chunk_patch
                hs.append(layers[l].register_forward_hook(rg_patch(vec)))
            with torch.no_grad():
                outs.append(model(**sub).logits[:, -1, :].float())
            for h in hs: h.remove()
        return torch.cat(outs, 0)

    def rg_patch(vec):
        def fn(mod, inp, out):
            if isinstance(out, tuple):
                return (out[0] + vec.to(dtype),) + out[1:]
            return out + vec.to(dtype)
        return fn

    clean_lp = torch.log_softmax(fwd(), -1)

    def kl(l, v):
        lp = torch.log_softmax(fwd((l, v.view(1, 1, -1))), -1)
        return float((lp.exp() * (lp - clean_lp)).sum(-1).mean())

    # residual norms + aligned directions
    store, resid, aligned = {}, {}, {}
    for l in lens_layers:
        hh = layers[l].register_forward_hook(rg.cap_hook(store) if hasattr(rg, "cap_hook")
                                             else (lambda m, i, o: store.__setitem__("h", o[0] if isinstance(o, tuple) else o)))
        with torch.no_grad():
            model(**{k: v[:a.chunk] for k, v in enc.items()})
        hh.remove()
        resid[l] = float(store["h"].float().norm(dim=-1).median())
        J = Jd[l].float().numpy()
        w = np.linalg.eigh(J.T @ M @ J)[1][:, -1]
        aligned[l] = torch.tensor(w / np.linalg.norm(w), dtype=torch.float32, device=a.device)

    rng = np.random.default_rng(0)
    rows = {}
    t0 = time.time()
    for frac in DOSE_FRACS:
        A_layers, R_layers = [], []
        for l in lens_layers:
            eps = frac * resid[l]
            ka = kl(l, aligned[l] * eps)
            kr = []
            for _ in range(N_RANDOM):
                v = torch.tensor(rng.standard_normal(d), dtype=torch.float32, device=a.device)
                kr.append(kl(l, (v / v.norm()) * eps))
            A_layers.append(ka); R_layers.append(float(np.mean(kr)))
        rows[frac] = {"aligned": A_layers, "random": R_layers,
                      "A": float(np.median(np.log(np.array(A_layers) / np.maximum(np.array(R_layers), 1e-30))))}
        print(f"  frac={frac:<9} A={rows[frac]['A']:+.4f}  aligned={np.median(A_layers):.3e} "
              f"random={np.median(R_layers):.3e}  ({time.time()-t0:.0f}s)", flush=True)

    # scaling exponents: log KL = k log eps + c, fitted per arm on median-over-layers
    lf = np.log(np.array(DOSE_FRACS))
    ka = np.polyfit(lf, np.log([np.median(rows[f]["aligned"]) for f in DOSE_FRACS]), 1)[0]
    kr = np.polyfit(lf, np.log([np.median(rows[f]["random"]) for f in DOSE_FRACS]), 1)[0]
    # small-dose-only exponents (first four rungs), where a floor would bite
    ka_s = np.polyfit(lf[:4], np.log([np.median(rows[f]["aligned"]) for f in DOSE_FRACS[:4]]), 1)[0]
    kr_s = np.polyfit(lf[:4], np.log([np.median(rows[f]["random"]) for f in DOSE_FRACS[:4]]), 1)[0]
    res = {"slug": a.slug, "model": a.model, "n_prompts": B, "prompt_indices": idxs,
           "lens_layers": [int(x) for x in lens_layers], "n_probe": n_probe,
           "by_dose": {str(k): v for k, v in rows.items()},
           "A_by_dose": {str(k): rows[k]["A"] for k in DOSE_FRACS},
           "exponent_aligned_all": float(ka), "exponent_random_all": float(kr),
           "exponent_aligned_small": float(ka_s), "exponent_random_small": float(kr_s),
           "transformers": transformers.__version__, "torch": torch.__version__}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\nexponents (k in KL ~ eps^k):  aligned all={ka:.2f} small={ka_s:.2f} | "
          f"random all={kr:.2f} small={kr_s:.2f}   (2.0 = first-order)")
    print(f"A across ladder: {[round(rows[f]['A'],3) for f in DOSE_FRACS]}")
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
