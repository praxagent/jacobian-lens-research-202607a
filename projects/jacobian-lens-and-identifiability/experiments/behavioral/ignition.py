"""Behavioral test #2 — the IGNITION test (Dehaene & Naccache's decisive GWT signature).

GWT's most reliable signature of conscious access is *ignition*: as input crosses a
threshold, workspace entry is nonlinear/all-or-none, not gradual. Dehaene & Naccache
named this the key test the J-space paper had NOT shown; Anthropic shipped ignition.json
but only partially addressed it. Nobody has run it independently on open models.

Method (Anthropic ignition.json + README): put a carrier sentence with a {W} slot; set
the slot token's *input embedding* to a mix  α·emb(A) + (1−α)·emb(B),  α swept 0→1.
At each layer read the J-lens **A-share** = (1/rank_A)/(1/rank_A + 1/rank_B) at the {W}
position. Per (pair, layer): the threshold α where share crosses 0.5, and the **10→90%
transition width in α**. NARROW width = sharp/all-or-none = ignition; WIDE = gradual.

The audit question: does ignition (narrow transitions) appear where the workspace BAND
exists (qwen3-4b, gemma-2-27b) and not where it's absent (gemma-2-9b)? If sharp
transitions track the band, that's independent, behavior-side evidence the band is a
real workspace — and the first open-model run of the field's decisive test.

Run (GPU; CPU-smoke on gpt2):
    uv run python ignition.py --slug gpt2-small --n-alpha 6 --max-pairs 4 --device cpu
    uv run python ignition.py --slug qwen3-4b
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "jacobian_lens"))
from cka_layers import REPO, resolve  # noqa: E402

IGNITION_JSON = HERE / "ignition.json"


def band_layers(source_layers: list[int]) -> list[int]:
    n = len(source_layers)
    return source_layers[n // 3: 2 * n // 3]


def single_id(tok, word: str) -> int | None:
    for form in (" " + word, word):
        enc = tok.encode(form, add_special_tokens=False)
        if len(enc) == 1:
            return enc[0]
    return None


def transition_width(alphas: np.ndarray, shares: np.ndarray) -> float:
    """10%->90% width in alpha of the A-share curve (monotone-ish). NaN if it never
    spans that range. Small = sharp (ignition); large = gradual."""
    if shares.max() < 0.9 or shares.min() > 0.1:
        return float("nan")
    a10 = np.interp(0.1, shares, alphas) if shares[0] < shares[-1] else np.interp(0.1, shares[::-1], alphas[::-1])
    a90 = np.interp(0.9, shares, alphas) if shares[0] < shares[-1] else np.interp(0.9, shares[::-1], alphas[::-1])
    return abs(a90 - a10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--n-alpha", type=int, default=11)
    ap.add_argument("--max-pairs", type=int, default=12)
    ap.add_argument("--max-carriers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lens-file", default=None,
                    help="local lens .pt (skips Neuronpedia download; for self-fitted lenses)")
    ap.add_argument("--big-model", default=None,
                    help="HF id for device_map+backbone loading of huge models, e.g. "
                         "'Qwen/Qwen3.5-397B-A17B:model.language_model'")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import jlens
    import transformers
    from huggingface_hub import hf_hub_download

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if args.big_model:
        # sharded path for frontier models: device_map + text-backbone layout
        hf_id, backbone = args.big_model.split(":")
        print(f"slug={args.slug} hf={hf_id} device_map=auto backbone={backbone}")
        from jlens import Layout
        cfg = transformers.AutoConfig.from_pretrained(hf_id)
        cls = getattr(transformers, cfg.architectures[0])
        n_gpu = torch.cuda.device_count()
        hf = cls.from_pretrained(hf_id, torch_dtype=torch.bfloat16, device_map="auto",
                                 attn_implementation="eager",
                                 max_memory={i: "110GiB" for i in range(n_gpu)}).eval()
        tok = transformers.AutoTokenizer.from_pretrained(hf_id)
        model = jlens.from_hf(hf, tok, compile=False, layout=Layout(path=backbone))
    else:
        hf_id, lens_file = resolve(args.slug)
        print(f"slug={args.slug} hf={hf_id} device={dev}")
        from _loader import load_hf_model  # handles mxfp4 dequant (gpt-oss) + torch shim
        hf = load_hf_model(hf_id, dev)
        tok = transformers.AutoTokenizer.from_pretrained(hf_id)
        model = jlens.from_hf(hf, tok)
    if args.lens_file:
        lens = jlens.JacobianLens.load(args.lens_file)
    else:
        _, lens_file = resolve(args.slug)
        lens = jlens.JacobianLens.load(hf_hub_download(REPO, filename=lens_file))
    U = hf.get_output_embeddings().weight.detach().to(dev).float()
    emb = hf.get_input_embeddings().weight.detach()
    band = band_layers(lens.source_layers)
    print(f"band layers: {band}")

    spec = json.load(open(IGNITION_JSON))
    # concept pairs: country x country (single-token in this tokenizer), + carriers
    countries = [c for c in spec["countries_12"] if single_id(tok, c) is not None]
    pairs = [(a, b) for a, b in itertools.combinations(countries, 2)][: args.max_pairs]
    carriers = spec["ctx_templates"][: args.max_carriers]
    alphas = np.linspace(0.0, 1.0, args.n_alpha)

    captured = {}
    def cap_hook(layer):
        def hook(_m, _i, out):
            captured[layer] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    per_layer_widths = {l: [] for l in band}
    share_spans, share_max, share_min = [], [], []
    # Inject the A/B interpolation at the carrier slot via an embedding-OUTPUT hook and
    # feed input_ids — mathematically identical to passing interpolated inputs_embeds
    # (any post-embedding scaling the model applies, e.g. Gemma's sqrt(d), lands on both
    # the same way), but it goes through input_ids so Gemma-4's forward accepts it.
    # Gemma-4 rejects arbitrary inputs_embeds ("Gemma4 needs to reverse the embedding …
    # provide exact inputs_embeds"), which is exactly what interpolation violates.
    inject = {}  # {pos: vector} applied at the current alpha; empty => no-op passthrough
    def embed_inject_hook(_m, _i, out):
        if not inject:
            return out
        out = out.clone()
        for p, v in inject.items():
            out[0, p] = v.to(out.device, out.dtype)
        return out
    emb_inj_handle = hf.get_input_embeddings().register_forward_hook(embed_inject_hook)
    for a_word, b_word in pairs:
        a_id, b_id = single_id(tok, a_word), single_id(tok, b_word)
        for carrier in carriers:
            # tokenize carrier with the slot filled by A, locate the slot position
            text = carrier.replace("{W}", a_word)
            ids = model.encode(text).to(dev)
            # slot position = index of a_id (last occurrence, robust to BOS)
            pos_matches = (ids[0] == a_id).nonzero(as_tuple=True)[0]
            if len(pos_matches) == 0:
                continue
            wpos = int(pos_matches[-1])
            shares = {l: [] for l in band}
            handles = [model.layers[l].register_forward_hook(cap_hook(l)) for l in band]
            for al in alphas:
                inject.clear()
                inject[wpos] = al * emb[a_id].to(dev) + (1 - al) * emb[b_id].to(dev)
                with torch.no_grad():
                    hf(input_ids=ids)
                for l in band:
                    h = captured[l][0, wpos].float().to(dev)
                    ll = U @ (lens.jacobians[l].to(dev).float() @ h)  # lens logits
                    order = ll.argsort(descending=True)
                    rank = {int(t): i + 1 for i, t in enumerate(order[:2000])}
                    ra = rank.get(a_id, 2001); rb = rank.get(b_id, 2001)
                    shares[l].append((1 / ra) / ((1 / ra) + (1 / rb)))
            inject.clear()
            for h in handles:
                h.remove()
            for l in band:
                sh = np.array(shares[l])
                w = transition_width(alphas, sh)
                if not np.isnan(w):
                    per_layer_widths[l].append(w)
                # diagnostic: the A-share RANGE actually swept (disambiguates
                # "no ignition" from "readout never resolved / stayed ~0.5")
                share_spans.append(float(sh.max() - sh.min()))
                share_max.append(float(sh.max())); share_min.append(float(sh.min()))

    emb_inj_handle.remove()
    all_widths = [w for ws in per_layer_widths.values() for w in ws]
    result = {
        "slug": args.slug, "hf_id": hf_id, "band": band,
        "n_alpha": args.n_alpha, "pairs": len(pairs), "carriers": len(carriers),
        "median_transition_width": float(np.median(all_widths)) if all_widths else None,
        "frac_sharp_lt_0.25": float(np.mean([w < 0.25 for w in all_widths])) if all_widths else None,
        "n_curves": len(all_widths),
        "n_readouts": len(share_spans),
        "median_share_span": float(np.median(share_spans)) if share_spans else None,
        "median_share_max": float(np.median(share_max)) if share_max else None,
        "median_share_min": float(np.median(share_min)) if share_min else None,
        "frac_readouts_resolved": (len(all_widths) / len(share_spans)) if share_spans else None,
        "per_layer_median_width": {str(l): (float(np.median(ws)) if ws else None)
                                   for l, ws in per_layer_widths.items()},
    }
    out = args.out or str(HERE / f"ignition_{args.slug}.json")
    json.dump(result, open(out, "w"), indent=1)
    mw = result["median_transition_width"]
    print(f"\nmedian 10->90% transition width in alpha: "
          f"{mw:.3f}" if mw is not None else "n/a")
    print(f"fraction of curves 'sharp' (<0.25 width): {result['frac_sharp_lt_0.25']}")
    print("  small width / high sharp-fraction => ignition-like (all-or-none);")
    print("  wide / low => gradual. Compare across the band-having vs bandless triple.")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
