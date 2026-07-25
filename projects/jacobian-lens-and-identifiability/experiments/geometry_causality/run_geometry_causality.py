"""Does J-space geometry have local causal purchase? Design frozen in PREREG.md.

Three equal-norm perturbation arms at the same layer and input, differing only in direction:
  ALIGNED  top right-singular vector of D_l = U_probe @ J_l  (input-INDEPENDENT lens claim)
  RANDOM   equal-norm random unit vectors, 8 draws           (floor)
  LOCAL    unit grad_{h_l} log p(t*), per prompt             (ceiling, input-DEPENDENT)

Because the arms share layer, input, and norm, update magnitude / depth / architecture /
generic sensitivity cancel in the contrast. That is the fix for the circularity that killed
the two block_patching designs.

Outcome: KL(perturbed || clean) at the final position, per prompt (stored, for bootstrap).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
# Self-contained: the probe helpers are inlined below rather than imported from the atlas
# package, so this runner ships to a pod as one file plus shared_tokens.json.
SHARED_TOKENS = Path(__file__).resolve().parent / "shared_tokens.json"
if not SHARED_TOKENS.exists():
    SHARED_TOKENS = HERE.parents[1] / "experiments/jacobian_lens/shared_tokens.json"

# Amendment 1: exact-doubling grid, extended downward. The original {0.05..1.0} grid
# sat above the first-order regime and the linear gate correctly rejected every dose.
# Amendment 2: grid extended further down as an exact-doubling ladder, because calibrating on
# the LOCAL arm (the strongest by construction) selects a much smaller dose than calibrating on
# ALIGNED did.
DOSE_FRACS = [0.0015625, 0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2]
N_RANDOM = 8
LINEAR_WINDOW = (3.2, 4.8)      # KL(2e)/KL(e) must land here: approximately quadratic


def get_layers(model):
    return model.model.layers if hasattr(model, "model") else model.transformer.h


KEY_SUFFIXES = ("embed_tokens.weight", "wte.weight", "embed_in.weight", "tok_embeddings.weight")


def resolve_ids_inline(hf_id, strings):
    """Canonical rule, inlined: space-prefixed form preferred, bare form fallback, drop any
    string needing more than one token (matches jacobian_lens/shared_vocab.py)."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    out = {}
    for s in strings:
        for form in (" " + s, s):
            e = tk.encode(form, add_special_tokens=False)
            if len(e) == 1:
                out[s] = e[0]; break
    return out


def probe_rows_inline(hf_id, ids):
    """Read only the probe rows of the embedding matrix, in row-chunks."""
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors import safe_open
    import torch as _t
    files = HfApi().list_repo_files(hf_id)
    idx = next((f for f in files if f.endswith(".safetensors.index.json")), None)
    if idx:
        wmap = json.load(open(hf_hub_download(hf_id, idx)))["weight_map"]
        key = next(k for k in wmap if k.endswith(KEY_SUFFIXES)); fname = wmap[key]
    else:
        fname = sorted(f for f in files if f.endswith(".safetensors"))[0]; key = None
    path = hf_hub_download(hf_id, fname)
    with safe_open(path, framework="pt") as f:
        if key is None or key not in f.keys():
            key = next(k for k in f.keys() if k.endswith(KEY_SUFFIXES))
        sl = f.get_slice(key); vocab, dd = sl.get_shape()
        order = np.argsort(ids); sid = np.asarray(ids)[order]
        out = np.empty((len(ids), dd), dtype=np.float32); pos = 0
        for st in range(0, vocab, 8192):
            sp = min(st + 8192, vocab)
            want = sid[(sid >= st) & (sid < sp)]
            if want.size == 0: continue
            blk = sl[st:sp].to(_t.float32).numpy()
            out[order[pos:pos + want.size]] = blk[want - st]; pos += want.size
    return out, int(dd)


def probe_M(slug, hf_id, d):
    """M = U_s^T U_s on the shared 4096-token probe. Aligned direction is the top
    eigenvector of J^T M J, which equals the top right-singular vector of U_s @ J."""
    strings = json.load(open(SHARED_TOKENS))["strings"]
    ids = resolve_ids_inline(hf_id, strings)
    idlist = [ids[s] for s in strings if s in ids]
    Us, dd = probe_rows_inline(hf_id, idlist)
    assert dd == d, f"probe d {dd} != model d {d}"
    Uc = Us - Us.mean(0, keepdims=True)
    return (Uc.T @ Uc).astype(np.float32), len(idlist)


def load_lens_J(slug):
    from huggingface_hub import hf_hub_download, list_repo_files
    fs = [f for f in list_repo_files("neuronpedia/jacobian-lens")
          if f.startswith(slug + "/") and f.endswith(".pt")]
    d = torch.load(hf_hub_download("neuronpedia/jacobian-lens", sorted(fs)[0]),
                   map_location="cpu", weights_only=False)
    return d["J"], sorted(d["J"].keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--slug", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--dose-layers", type=int, default=3, help="layers used for the dose scan")
    ap.add_argument("--calib-arm", default="aligned", choices=["aligned", "local"],
                    help="which arm the linear-regime gate calibrates on. LOCAL is the strongest "
                         "arm by construction, so calibrating on ALIGNED leaves LOCAL saturated "
                         "and makes C uninterpretable (see results.md).")
    ap.add_argument("--chunk", type=int, default=32,
                    help="prompt chunk per forward; full-batch logits are (B,T,vocab) and OOM "
                         "on large-vocab models (gemma 262k x 200 x 64 is ~13GB)")
    a = ap.parse_args()
    import transformers
    from datasets import load_dataset

    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dtype = torch.float32 if a.device == "cpu" else torch.bfloat16
    model = transformers.AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(a.device).eval()
    layers = get_layers(model); L = len(layers)
    d = model.config.hidden_size

    # prompts: held-out slice, disjoint from lens-fitting subset (recorded)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    texts, idxs = [], []
    for i in range(200_000, len(ds["text"])):        # far past the lens-fitting head slice
        t = ds["text"][i]
        if len(t) >= 300:
            texts.append(t); idxs.append(i)
        if len(texts) >= a.n_prompts: break
    enc = tok(texts, return_tensors="pt", truncation=True, max_length=a.seq_len,
              padding="max_length", padding_side="left").to(a.device)
    B = enc.input_ids.shape[0]
    print(f"{B} prompts, L={L}, d={d}", flush=True)

    Jd, lens_layers = load_lens_J(a.slug)
    M, n_probe = probe_M(a.slug, a.model, d)
    print(f"lens covers {len(lens_layers)} layers; probe {n_probe} tokens", flush=True)

    def fwd_last(chunk):
        """Final-position logits, chunked over prompts to bound the (B,T,vocab) tensor."""
        outs = []
        for i in range(0, B, chunk):
            sub = {k: v[i:i+chunk] for k, v in enc.items()}
            with torch.no_grad():
                outs.append(model(**sub).logits[:, -1, :].float())
        return torch.cat(outs, 0)

    clean_logits = fwd_last(a.chunk)
    clean_lp = torch.log_softmax(clean_logits, -1)
    t_star = clean_lp.argmax(-1)
    resid_norm = {}

    def cap_hook(store):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            store["h"] = h
            return out
        return fn

    def patch_hook(vec):
        def fn(mod, inp, out):
            if isinstance(out, tuple):
                return (out[0] + vec,) + out[1:]
            return out + vec
        return fn

    def kl_for(l, v):
        """v: (d,) or (B,1,d). Returns per-prompt KL(perturbed||clean) in nats. Chunked."""
        outs = []
        for i in range(0, B, a.chunk):
            sub = {k: vv2[i:i+a.chunk] for k, vv2 in enc.items()}
            vv = v[i:i+a.chunk] if v.dim() == 3 else v.view(1, 1, -1)
            h = layers[l].register_forward_hook(patch_hook(vv.to(dtype)))
            with torch.no_grad():
                outs.append(model(**sub).logits[:, -1, :].float())
            h.remove()
        lg = torch.cat(outs, 0)
        lp = torch.log_softmax(lg, -1)
        return (lp.exp() * (lp - clean_lp)).sum(-1).cpu().numpy()

    # residual norms + aligned/local directions per lens layer
    store = {}
    hs = [layers[l].register_forward_hook(cap_hook(store)) for l in range(L)]
    aligned, local = {}, {}
    for l in lens_layers:
        for h in hs: h.remove()
        hh = layers[l].register_forward_hook(cap_hook(store))
        with torch.no_grad():
            model(**{k: v[:a.chunk] for k, v in enc.items()})
        hh.remove()
        resid_norm[l] = float(store["h"].float().norm(dim=-1).median())
        J = Jd[l].float().numpy()
        A = J.T @ M @ J                                   # = (U_s J)^T (U_s J)
        w = np.linalg.eigh(A)[1][:, -1]                   # top right-singular vector
        aligned[l] = torch.tensor(w / np.linalg.norm(w), dtype=torch.float32, device=a.device)
    for h in hs:
        try: h.remove()
        except Exception: pass

    # LOCAL: per-prompt gradient direction
    for l in lens_layers:
        st = {}
        def gcap(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h.retain_grad(); st["h"] = h
            return out
        gparts = []
        for i in range(0, B, a.chunk):
            sub = {k: v[i:i+a.chunk] for k, v in enc.items()}
            hh = layers[l].register_forward_hook(gcap)
            lg = model(**sub).logits[:, -1, :]
            n = lg.shape[0]
            sel = torch.log_softmax(lg.float(), -1)[torch.arange(n), t_star[i:i+n]].sum()
            gparts.append(torch.autograd.grad(sel, st["h"])[0].float().sum(1).detach())
            hh.remove(); model.zero_grad(set_to_none=True)
        gs = torch.cat(gparts, 0)                         # one direction per prompt
        local[l] = (gs / gs.norm(dim=-1, keepdim=True).clamp(min=1e-12)).unsqueeze(1).detach()
    print("directions computed", flush=True)

    # GATE 1 execution: eps=0 identity
    zero_kl = kl_for(lens_layers[0], torch.zeros(d, device=a.device))
    gate_exec = float(np.max(np.abs(zero_kl))) < 1e-6

    # GATE 3 dose selection on a subset of layers
    scan_layers = [lens_layers[i] for i in np.linspace(0, len(lens_layers)-1, a.dose_layers).astype(int)]
    dose_scan = {}
    for frac in DOSE_FRACS:
        vals = []
        for l in scan_layers:
            eps = frac * resid_norm[l]
            probe_v = (local[l] * eps) if a.calib_arm == "local" else (aligned[l] * eps)
            vals.append(float(np.mean(kl_for(l, probe_v))))
        dose_scan[frac] = vals
    ratios = {}
    for i, frac in enumerate(DOSE_FRACS[:-1]):
        nxt = DOSE_FRACS[i+1]
        if abs(nxt / frac - 2.0) < 0.3:                   # adjacent doses that are ~2x
            r = [dose_scan[nxt][k] / max(dose_scan[frac][k], 1e-12) for k in range(len(scan_layers))]
            ratios[frac] = float(np.median(r))
    chosen = None
    for frac in DOSE_FRACS:
        if frac in ratios and LINEAR_WINDOW[0] <= ratios[frac] <= LINEAR_WINDOW[1]:
            chosen = frac
    gate_linear = chosen is not None
    if chosen is None:
        chosen = DOSE_FRACS[0]
    print(f"dose scan ratios {ratios}; chosen frac={chosen} linear_gate={gate_linear}", flush=True)

    # main measurement at the chosen dose
    rows = {}
    rng = np.random.default_rng(0)
    t0 = time.time()
    for l in lens_layers:
        eps = chosen * resid_norm[l]
        ka = kl_for(l, aligned[l] * eps)
        kr = []
        for r in range(N_RANDOM):
            v = torch.tensor(rng.standard_normal(d), dtype=torch.float32, device=a.device)
            v = v / v.norm()
            kr.append(kl_for(l, v * eps))
        kl_loc = kl_for(l, local[l] * eps)
        rows[int(l)] = {"eps": eps,
                        "norm_aligned": float((aligned[l]*eps).norm()),
                        "norm_random": float((v*eps).norm()),
                        "norm_local": float((local[l][0,0]*eps).norm()),
                        "kl_aligned": ka.tolist(),
                        "kl_random": np.stack(kr).tolist(),
                        "kl_local": kl_loc.tolist()}
        print(f"  layer {l}: aligned {ka.mean():.4g} random {np.stack(kr).mean():.4g} "
              f"local {kl_loc.mean():.4g}  ({time.time()-t0:.0f}s)", flush=True)

    # gates 2 and 4
    gate_norm = all(abs(r["norm_aligned"] - r["norm_random"]) / max(r["norm_aligned"], 1e-12) < 1e-6
                    and abs(r["norm_aligned"] - r["norm_local"]) / max(r["norm_aligned"], 1e-12) < 1e-6
                    for r in rows.values())
    gate_sens = float(np.mean([np.mean(r["kl_local"]) > np.mean(r["kl_random"]) for r in rows.values()])) > 0.5

    A_l = {l: float(np.log(np.mean(r["kl_aligned"]) / max(np.mean(r["kl_random"]), 1e-30)))
           for l, r in rows.items()}
    C_l = {l: float(np.mean(r["kl_aligned"]) / max(np.mean(r["kl_local"]), 1e-30))
           for l, r in rows.items()}
    res = {"slug": a.slug, "model": a.model, "n_layers": L, "d": d, "n_prompts": B,
           "seq_len": a.seq_len, "prompt_dataset_indices": idxs, "lens_layers": [int(x) for x in lens_layers],
           "n_probe_tokens": n_probe, "device": a.device, "dtype": str(dtype),
           "calibration_arm": a.calib_arm,
           "dose_scan": {str(k): v for k, v in dose_scan.items()},
           "dose_ratios": {str(k): v for k, v in ratios.items()}, "chosen_dose_frac": chosen,
           "gates": {"execution": bool(gate_exec), "equal_norm": bool(gate_norm),
                     "linear_regime": bool(gate_linear), "sensitivity_local_beats_random": bool(gate_sens)},
           "A_l": A_l, "C_l": C_l, "median_A": float(np.median(list(A_l.values()))),
           "median_C": float(np.median(list(C_l.values()))),
           "per_layer": rows,
           "transformers": transformers.__version__, "torch": torch.__version__}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\nGATES {res['gates']}")
    print(f"median A_l = {res['median_A']:+.4f}  (>0 means lens beats equal-norm random)")
    print(f"median C_l = {res['median_C']:.4f}  (fraction of achievable first-order effect)")
    print("GEOMCAUSAL_DONE", flush=True)


if __name__ == "__main__":
    main()
