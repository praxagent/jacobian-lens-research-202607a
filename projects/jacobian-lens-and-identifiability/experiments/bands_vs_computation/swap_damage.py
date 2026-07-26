"""Test C runner: same-prompt swap damage D(i,j), measured separately on prose and code.

Adapted from ../block_patching/swap_v2.py, made self-contained and pointed at the frozen
prompt artifact so no `datasets` import happens at run time. Design frozen in PREREG.md.

D(i, j) = mean over prompts of KL(patched || clean) for the next-token distribution, where
"patched" replaces layer j's output with layer i's captured clean output on the same prompt.
Saves PER-PROMPT KL so a bootstrap is possible from the receipt alone.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def hook_capture(store, key):
    def fn(mod, inp, out):
        store[key] = (out[0] if isinstance(out, tuple) else out).detach()
    return fn


def hook_patch(vec):
    def fn(mod, inp, out):
        if isinstance(out, tuple):
            return (vec,) + tuple(out[1:])
        return vec
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--corpus", required=True, choices=["prose", "code"])
    ap.add_argument("--prompts", default=str(HERE / "prompts_frozen.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    # float32 everywhere by default, on GPU too: a bf16 precision floor manufactured a finding
    # once already in this campaign (see ../geometry_causality/results.md), and D values here
    # are small KLs where that floor would bite.
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    a = ap.parse_args()
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, a.dtype)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=dtype).to(a.device).eval()
    layers = model.model.layers if hasattr(model, "model") else model.transformer.h
    L = len(layers)

    art = json.load(open(a.prompts))
    texts = art[a.corpus]
    print(f"{a.slug} / {a.corpus}: {len(texts)} prompts, L={L}", flush=True)

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
    Dp = np.zeros((L, L, len(texts)), dtype=np.float32)
    t0 = time.time()
    for i in range(L):
        for j in range(L):
            lp = run(patch=(j, acts[i]))
            kl = (lp.exp() * (lp - clean)).sum(-1)
            Dp[i, j] = kl.cpu().numpy()
            D[i, j] = float(kl.mean())
        print(f"  source layer {i+1}/{L} ({time.time()-t0:.0f}s)", flush=True)

    diag = [float(D[i, i]) for i in range(L)]
    far = np.array([D[i, j] for i in range(L) for j in range(L) if abs(i - j) >= 2])
    res = {"slug": a.slug, "model": a.model, "corpus": a.corpus, "n_layers": L,
           "n_prompts": len(texts), "device": a.device, "dtype": str(dtype),
           "D": D.tolist(), "D_per_prompt": Dp.tolist(), "prompts": texts,
           "prompt_artifact_target_tokens": art["target_tokens"],
           "diag_max_abs": float(np.max(np.abs(diag))),
           "median_D_far": float(np.median(far)),
           "transformers": transformers.__version__, "torch": torch.__version__}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"SWAP_DONE {a.slug}/{a.corpus}: diag_max_abs={res['diag_max_abs']:.2e} "
          f"median_D_far={res['median_D_far']:.4f} (gate band [0.05, 5.0])", flush=True)


if __name__ == "__main__":
    main()
