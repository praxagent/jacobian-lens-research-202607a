"""v2: same-prompt cross-layer swap damage. Design frozen in PREREG_V2.md.

Capture the full residual stream at layer i, re-run the SAME prompt substituting that state
at layer j, and measure KL(patched || clean) on the next-token distribution. Nothing semantic
changes, so this isolates format compatibility: can layers j..L consume a layer-i
representation of the same input?

D(i,i) = 0 by construction (sanity gate) and D has no ceiling (fixes v1's saturation).

  python swap_v2.py --model google/gemma-3-270m --slug gemma-3-270m --out out/v2_pilot.json
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def hook_capture(store, key):
    def fn(mod, inp, out):
        store[key] = (out[0] if isinstance(out, tuple) else out).detach().clone()
    return fn


def hook_patch(vec):
    def fn(mod, inp, out):
        return (vec.clone(),) + out[1:] if isinstance(out, tuple) else vec.clone()
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--slug", required=True)
    ap.add_argument("--prompts", default=str(HERE / "prompts.json"))
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32 if a.device == "cpu" else torch.bfloat16
    model = transformers.AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=dtype).to(a.device).eval()
    layers = model.model.layers if hasattr(model, "model") else model.transformer.h
    L = len(layers)

    # v2 uses prompts as independent texts; no source/target roles
    raw = json.load(open(a.prompts))
    texts = [p["src"] for p in raw] + [p["tgt"] for p in raw]
    print(f"{len(texts)} prompts, L={L}", flush=True)

    def run(patch=None):
        enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to(a.device)
        h = layers[patch[0]].register_forward_hook(hook_patch(patch[1])) if patch else None
        with torch.no_grad():
            lg = model(**enc).logits[:, -1, :].float()
        if h: h.remove()
        return torch.log_softmax(lg, -1)

    store = {}
    handles = [layers[i].register_forward_hook(hook_capture(store, i)) for i in range(L)]
    clean = run()
    acts = {i: store[i].clone() for i in range(L)}
    for h in handles: h.remove()

    D = np.zeros((L, L))
    # RECEIPT FIX (found by analysis C): store PER-PROMPT KL, not just the mean, so a
    # prompt-level bootstrap can be computed from the receipt without re-renting the GPU.
    Dp = np.zeros((L, L, len(texts)), dtype=np.float32)
    t0 = time.time()
    for i in range(L):
        for j in range(L):
            lp = run(patch=(j, acts[i]))
            # KL(patched || clean), per prompt, in nats
            kl = (lp.exp() * (lp - clean)).sum(-1)
            Dp[i, j] = kl.cpu().numpy()
            D[i, j] = float(kl.mean())
        print(f"  source layer {i+1}/{L} ({time.time()-t0:.0f}s)", flush=True)

    diag = [float(D[i, i]) for i in range(L)]
    off = D[~np.eye(L, dtype=bool)]
    far = np.array([D[i, j] for i in range(L) for j in range(L) if abs(i - j) >= 2])
    res = {"slug": a.slug, "model": a.model, "n_layers": L, "n_prompts": len(texts),
           "device": a.device, "dtype": str(dtype), "D": D.tolist(),
           "D_per_prompt": Dp.tolist(), "prompts": texts,
           "diag_max_abs": float(np.max(np.abs(diag))),
           "median_D_far": float(np.median(far)),
           "mean_D_by_dist": {int(d): float(np.mean([D[i, j] for i in range(L) for j in range(L)
                                                     if abs(i - j) == d])) for d in range(1, L)},
           "transformers": transformers.__version__, "torch": torch.__version__}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\nSANITY  max|D(i,i)| = {res['diag_max_abs']:.2e}  (gate: ~0)")
    print(f"CALIBRATION  median D over |i-j|>=2 = {res['median_D_far']:.3f} nats "
          f"(gate: 0.05 to 5.0)")
    print("SWAP_V2_DONE", flush=True)


if __name__ == "__main__":
    main()
