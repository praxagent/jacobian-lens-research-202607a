"""Cross-layer activation patching: are J-space depth blocks CAUSAL boundaries?

Design is frozen in PREREG.md. Summary: capture the source prompt's residual stream at
layer i, insert it at layer j of the target prompt's forward pass (final token position),
and measure how far the output moves toward the source answer:

    E(i,j) = (LD_patched - LD_target) / (LD_source - LD_target),  LD = logit(a_src) - logit(a_tgt)

The question is NOT whether E falls with layer distance (it must, trivially), but whether
there is a discontinuity at a fitted block boundary AFTER absorbing distance. Analysis
(analyze.py) regresses E on per-distance dummies plus a `crosses` indicator and compares the
coefficient against a random-3-segmentation null.

Runs on CPU for a small pilot model; the same script runs on GPU with --device cuda.

  python patch_blocks.py --model google/gemma-3-270m --slug gemma-3-270m --out out/pilot.json
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def load_prompts(tok, path, device):
    """Keep only pairs whose answers are single tokens in THIS tokenizer (recorded, not
    silently dropped: the count is written to the receipt)."""
    raw = json.load(open(path))
    kept, dropped = [], []
    for p in raw:
        ts, tt = tok.encode(p["a_src"], add_special_tokens=False), tok.encode(p["a_tgt"], add_special_tokens=False)
        ls, lt = len(tok.encode(p["src"])), len(tok.encode(p["tgt"]))
        if len(ts) == 1 and len(tt) == 1 and ls == lt:
            kept.append({**p, "id_src": ts[0], "id_tgt": tt[0]})
        else:
            dropped.append({"pair": p["src"], "n_src": len(ts), "n_tgt": len(tt),
                            "len_src": ls, "len_tgt": lt,
                            "why": "multi-token answer" if (len(ts)>1 or len(tt)>1) else "unequal prompt length"})
    return kept, dropped


def resid_hook_capture(store, key):
    """Capture the FULL residual stream (all positions). Amendment 1: last-position-only
    patching is washed out by later attention to the target's own subject tokens."""
    def fn(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        store[key] = h.detach().clone()
    return fn


def resid_hook_patch(vec):
    """Replace the full residual stream, so E(i,i) == 1 by construction."""
    def fn(mod, inp, out):
        if isinstance(out, tuple):
            return (vec.clone(),) + out[1:]
        return vec.clone()
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--prompts", default=str(HERE / "prompts.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-denom", type=float, default=1.0)   # frozen exclusion rule
    a = ap.parse_args()
    import transformers

    dev = a.device
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:                      # gpt2 and friends ship no pad token
        tok.pad_token = tok.eos_token
    dtype = torch.float32 if dev == "cpu" else torch.bfloat16
    model = transformers.AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(dev).eval()
    layers = model.model.layers if hasattr(model, "model") else model.transformer.h
    L = len(layers)
    pairs, dropped = load_prompts(tok, a.prompts, dev)
    print(f"{len(pairs)} usable pairs ({len(dropped)} dropped for multi-token answers), L={L}", flush=True)

    def run(texts, patch=None):
        """Batched forward; patch=(layer_idx, tensor[B,d]) inserts at that layer, last pos."""
        enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to(dev)
        hs = []
        if patch is not None:
            h = layers[patch[0]].register_forward_hook(resid_hook_patch(patch[1]))
            hs.append(h)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        for h in hs: h.remove()
        return logits

    src_txt = [p["src"] for p in pairs]; tgt_txt = [p["tgt"] for p in pairs]
    isrc = torch.tensor([p["id_src"] for p in pairs]); itgt = torch.tensor([p["id_tgt"] for p in pairs])
    LD = lambda lg: (lg[torch.arange(len(pairs)), isrc] - lg[torch.arange(len(pairs)), itgt]).cpu().numpy()

    # clean runs + per-layer source activations
    store = {}
    handles = [layers[i].register_forward_hook(resid_hook_capture(store, i)) for i in range(L)]
    ld_src = LD(run(src_txt)); src_acts = {i: store[i].clone() for i in range(L)}
    for h in handles: h.remove()
    ld_tgt = LD(run(tgt_txt))

    denom = ld_src - ld_tgt
    keep = np.abs(denom) >= a.min_denom
    print(f"clean: LD_src mean {ld_src.mean():+.2f}, LD_tgt mean {ld_tgt.mean():+.2f}; "
          f"{int(keep.sum())}/{len(pairs)} pairs pass |denom|>={a.min_denom}", flush=True)

    E = np.full((L, L), np.nan)
    t0 = time.time()
    for i in range(L):
        for j in range(L):
            lg = run(tgt_txt, patch=(j, src_acts[i]))
            e = (LD(lg) - ld_tgt) / denom
            E[i, j] = float(np.mean(e[keep]))
        print(f"  source layer {i+1}/{L} done ({time.time()-t0:.0f}s)", flush=True)

    self_patch = [float(E[i, i]) for i in range(L)]
    res = {"slug": a.slug, "model": a.model, "n_layers": L, "device": dev, "dtype": str(dtype),
           "n_pairs": len(pairs), "n_kept": int(keep.sum()), "dropped_pairs": dropped,
           "min_denom": a.min_denom,
           "clean_ld_src": ld_src.tolist(), "clean_ld_tgt": ld_tgt.tolist(),
           "E": E.tolist(), "self_patch": self_patch,
           "self_patch_median": float(np.median(self_patch)),
           "transformers": transformers.__version__, "torch": torch.__version__}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\nSANITY self-patch median E(i,i) = {np.median(self_patch):.3f} "
          f"(pre-registered gate: >= 0.9)", flush=True)
    print("PATCH_DONE", flush=True)


if __name__ == "__main__":
    main()
