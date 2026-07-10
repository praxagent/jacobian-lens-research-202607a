"""SAE x J-space bridge: do deception features enter the verbalizable workspace?

Three stages (see README.md; all inputs frozen in features.json):
  0  kappa triage — Anthropic's lens-kurtosis for each feature's decoder direction
     projected through the J-lens, + its J-lens top tokens per band layer.
  A  SAE -> J-space: steer feature i at the SAE hook layer (c * decoder column, all
     positions), read the J-lens depth profile. Falsification: readout strictly below
     the hook must be bit-identical to c=0; onset exactly at the hook.
  B  J-space -> SAE: steer J-lens token directions at band layers BELOW the hook,
     read the six features' activations at the hook (max-over-positions, the
     llm_selfref_pre convention). Zero-control: identical steering applied only ABOVE
     the hook must leave hook activations exactly unchanged.

Tiers: --tier 8b  = Goodfire/Llama-3.1-8B-Instruct-SAE-l19 + neuronpedia llama3.1-8b-it
       --tier 70b = Goodfire/Llama-3.3-70B-Instruct-SAE-l50 + neuronpedia llama3.3-70b-it
Both assert the SAE and lens declare the same base model. --mock runs a CPU smoke with
a random SAE on gpt2 (plumbing only; asserts must still pass).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "behavioral"))
sys.path.insert(0, str(HERE.parent / "jacobian_lens"))

SPEC = json.load(open(HERE / "features.json"))

TIERS = {
    "8b": {"hf_id": "meta-llama/Llama-3.1-8B-Instruct", "slug": "llama3.1-8b-it",
           "sae_repo": "Goodfire/Llama-3.1-8B-Instruct-SAE-l19",
           "sae_file": "Llama-3.1-8B-Instruct-SAE-l19.pth", "hook_layer": 19},
    "70b": {"hf_id": "meta-llama/Llama-3.3-70B-Instruct", "slug": "llama3.3-70b-it",
            "sae_repo": "Goodfire/Llama-3.3-70B-Instruct-SAE-l50",
            "sae_file": "Llama-3.3-70B-Instruct-SAE-l50.pt", "hook_layer": 50},
    "mock": {"hf_id": "openai-community/gpt2", "slug": "gpt2-small",
             "sae_repo": None, "sae_file": None, "hook_layer": 6},
}


def band_layers(source_layers):
    n = len(source_layers)
    return source_layers[n // 3: 2 * n // 3]


def single_token_ids(tok, words):
    out = {}
    for w in words:
        for form in (" " + w, w):
            enc = tok.encode(form, add_special_tokens=False)
            if len(enc) == 1:
                out[w] = enc[0]
                break
    return out


def load_sae(tier, dev):
    """Return (W_enc [F,d], b_enc [F], W_dec [d,F]) fp32 on dev."""
    if tier["sae_repo"] is None:  # mock: random SAE, gpt2 d=768
        g = torch.Generator().manual_seed(0)
        d, F = 768, 4096
        return (torch.randn(F, d, generator=g), torch.zeros(F),
                torch.randn(d, F, generator=g))
    from huggingface_hub import hf_hub_download
    sd = torch.load(hf_hub_download(tier["sae_repo"], tier["sae_file"]),
                    map_location="cpu", weights_only=True)
    return (sd["encoder_linear.weight"].float().to(dev),
            sd["encoder_linear.bias"].float().to(dev),
            sd["decoder_linear.weight"].float().to(dev))


def sae_acts(resid, W_enc, b_enc, feat_ids):
    """max-over-positions relu activation per feature (llm_selfref_pre convention)."""
    a = torch.relu(resid.to(W_enc.device).float() @ W_enc.T + b_enc)  # [seq, F]
    return {int(f): float(a[:, f].max().item()) for f in feat_ids}


def kurtosis(x):
    x = x.float()
    mu, sd = x.mean(), x.std()
    return float((((x - mu) / (sd + 1e-9)) ** 4).mean().item())


def add_hook(module, vec):
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + vec.to(device=h.device, dtype=h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h
    return module.register_forward_hook(hook)


def capture_hook(module, store, key):
    def hook(_m, _i, out):
        store[key] = (out[0] if isinstance(out, tuple) else out)[0].detach()
    return module.register_forward_hook(hook)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=list(TIERS))
    ap.add_argument("--stages", default="0,A,B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tier = TIERS[args.tier]
    stages = set(args.stages.split(","))

    import jlens
    import transformers
    from huggingface_hub import hf_hub_download
    from cka_layers import REPO, resolve

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    hf_id, lens_file = resolve(tier["slug"])
    assert hf_id == tier["hf_id"], \
        f"PAIRING VIOLATION: lens is for {hf_id}, SAE for {tier['hf_id']}"

    n_gpu = torch.cuda.device_count() if dev != "cpu" else 0
    kw = dict(torch_dtype=torch.bfloat16)
    if n_gpu > 1:
        kw.update(device_map="auto", attn_implementation="eager")
    hf = transformers.AutoModelForCausalLM.from_pretrained(hf_id, **kw).eval()
    if n_gpu <= 1:
        hf = hf.to(dev)
    tok = transformers.AutoTokenizer.from_pretrained(hf_id)
    model = jlens.from_hf(hf, tok, compile=False)
    lens = jlens.JacobianLens.load(hf_hub_download(REPO, filename=lens_file))
    dev0 = next(hf.parameters()).device
    U = hf.get_output_embeddings().weight.detach().to(dev0).float()

    W_enc, b_enc, W_dec = load_sae(tier, dev0)
    hook_l = tier["hook_layer"]
    src = lens.source_layers
    band = band_layers(src)
    below = [l for l in src if l < hook_l]
    above_band = [l for l in band if l > hook_l]
    print(f"model={hf_id} hook_layer={hook_l} band={band[0]}..{band[-1]} "
          f"({len(below)} src layers below hook)")

    targets = [f["id"] for f in SPEC["targets"]] if args.tier != "mock" else [1, 2, 3]
    adjacent = SPEC["index_adjacent_controls"] if args.tier != "mock" else [4, 5]
    g = torch.Generator().manual_seed(SPEC["random_control_seed"])
    F = W_enc.shape[0]
    rand_feats = []
    while len(rand_feats) < (SPEC["n_random_controls"] if args.tier != "mock" else 3):
        c = int(torch.randint(0, F, (1,), generator=g))
        if c not in targets and c not in adjacent and c not in rand_feats:
            rand_feats.append(c)
    all_feats = targets + adjacent + rand_feats

    probe_ids = single_token_ids(tok, SPEC["deception_probe_tokens"])
    prompts = SPEC["carrier_prompts"]
    coeffs = SPEC["coefficients"]
    layer_mods = model.layers
    receipt = {"tier": args.tier, "model": hf_id, "hook_layer": hook_l,
               "band": band, "probe_tokens_resolved": sorted(probe_ids),
               "targets": targets, "adjacent": adjacent, "random": rand_feats}

    # ---------- Stage 0: kappa triage ----------
    if "0" in stages:
        print("\n== Stage 0: kappa (lens-kurtosis) triage ==")
        receipt["stage0"] = {}
        for f in all_feats:
            d = W_dec[:, f].to(dev0)
            per_layer = {}
            for l in band:
                r = U @ (lens.jacobians[l].to(dev0).float() @ d)
                per_layer[l] = {"kappa": kurtosis(r),
                                "top10": tok.batch_decode(
                                    [[t] for t in r.topk(10).indices.tolist()])}
            best = max(per_layer.values(), key=lambda v: v["kappa"])
            receipt["stage0"][f] = {"band_kappa_max": best["kappa"],
                                    "top10_at_peak": best["top10"],
                                    "per_layer_kappa": {l: v["kappa"] for l, v in per_layer.items()}}
            kind = "TARGET" if f in targets else ("adj" if f in adjacent else "rand")
            print(f"  [{kind:6s}] f{f}: kappa_max={best['kappa']:.1f} "
                  f"top: {best['top10'][:5]}")

    # ---------- Stage A: SAE -> J-space ----------
    if "A" in stages:
        print("\n== Stage A: SAE feature steering -> J-lens readout ==")
        read_layers = sorted(set(below[-2:] + [l for l in src if l >= hook_l]))
        receipt["stageA"] = {"read_layers": read_layers, "runs": []}
        feats_A = targets + adjacent[:4] + rand_feats[:4]
        for f in feats_A:
            d = W_dec[:, f].to(dev0)
            d_unit = d / (d.norm() + 1e-8)
            for prompt in prompts:
                base_read = None
                for c in (coeffs if f in targets else [0, max(coeffs)]):
                    handles = []
                    if c:
                        handles.append(add_hook(layer_mods[hook_l], c * d_unit * d.norm()))
                    with torch.no_grad():
                        ll, _, _ = lens.apply(model, prompt, layers=read_layers,
                                              positions=[-1])
                        ids = model.encode(prompt).to(dev0)
                        nxt = int(hf(ids).logits[0, -1].argmax())
                    for h in handles:
                        h.remove()
                    row = {"feature": f, "prompt": prompts.index(prompt), "coeff": c,
                           "next_token": tok.decode([nxt])}
                    scores = {}
                    for l in read_layers:
                        v = ll[l][0].float()
                        scores[l] = {"probe_mean": float(v[list(probe_ids.values())].mean()),
                                     "top5": tok.batch_decode([[t] for t in v.topk(5).indices.tolist()])}
                    if c == 0:
                        base_read = scores
                        row["below_hook_drift"] = 0.0
                    else:
                        drift = max(abs(scores[l]["probe_mean"] - base_read[l]["probe_mean"])
                                    for l in read_layers if l < hook_l)
                        row["below_hook_drift"] = drift
                        assert drift < 1e-3, \
                            f"GEOMETRY VIOLATION: readout below hook moved ({drift})"
                        row["probe_delta_at_band_above"] = {
                            l: scores[l]["probe_mean"] - base_read[l]["probe_mean"]
                            for l in read_layers if l >= hook_l}
                        row["top5_at_max_layer"] = scores[max(read_layers)]["top5"]
                    receipt["stageA"]["runs"].append(row)
            print(f"  feature {f} done")

    # ---------- Stage B: J-space -> SAE ----------
    if "B" in stages:
        print("\n== Stage B: J-lens steering -> SAE feature activations ==")
        from jlens.hooks import ActivationRecorder
        steer_layers = [l for l in band if l < hook_l]
        zero_layers = above_band[:2] if above_band else []
        receipt["stageB"] = {"steer_layers": [steer_layers[0], steer_layers[-1]],
                             "zero_control_layers": zero_layers, "runs": []}
        cap = {}
        for prompt in prompts:
            ids = model.encode(prompt).to(dev0)
            with torch.no_grad(), ActivationRecorder(model.layers, at=steer_layers) as rec:
                model.forward(ids)
            mean_norm = {l: rec.activations[l][0].norm(dim=-1).mean().item()
                         for l in steer_layers}
            h_cap = capture_hook(layer_mods[hook_l], cap, "resid")
            with torch.no_grad():
                model.forward(ids)
            base_acts = sae_acts(cap["resid"], W_enc, b_enc, all_feats)

            for word, tid in list(probe_ids.items()):
                for cond, layers_ in (("steer", steer_layers), ("zero_ctrl", zero_layers)):
                    if not layers_:
                        continue
                    handles = []
                    for l in layers_:
                        dl = (U[tid] @ lens.jacobians[l].to(dev0).float())
                        dl = dl / (dl.norm() + 1e-8)
                        handles.append(add_hook(layer_mods[l], 8.0 * mean_norm.get(l, 1.0) * dl))
                    with torch.no_grad():
                        model.forward(ids)
                    for h in handles:
                        h.remove()
                    acts = sae_acts(cap["resid"], W_enc, b_enc, all_feats)
                    delta = {f: acts[f] - base_acts[f] for f in all_feats}
                    if cond == "zero_ctrl":
                        worst = max(abs(v) for v in delta.values())
                        assert worst < 1e-3, \
                            f"GEOMETRY VIOLATION: above-hook steering moved hook acts ({worst})"
                    receipt["stageB"]["runs"].append(
                        {"prompt": prompts.index(prompt), "token": word, "cond": cond,
                         "delta_targets": {f: delta[f] for f in targets},
                         "delta_adjacent_mean": sum(delta[f] for f in adjacent) / len(adjacent),
                         "delta_random_mean": sum(delta[f] for f in rand_feats) / len(rand_feats)})
            h_cap.remove()
            print(f"  prompt {prompts.index(prompt)} done")

    out = args.out or str(HERE / f"sae_x_jspace_{args.tier}.json")
    json.dump(receipt, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
