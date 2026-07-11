"""Q1 (free, local): where do digit features live, and what company do digits keep
at depth? — read digit-token directions through the published 397B J-lens across
layers. No model forward pass, no pod: just the local lens + the lm_head shard.

For each fitted layer l: project the digit tokens through J_l (r = U_digits @ J_l),
measure per-layer kurtosis (peakedness = "how legible / how much do digits live here")
and, for the mean digit direction, read the top co-occurring vocabulary tokens
("what company they keep") — shallow vs deep.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LENS = HERE.parents[1] / "artifacts" / "lenses-397b" / "qwen35_397b_dm.pt"
HF_ID = "Qwen/Qwen3.5-397B-A17B"
OUT = HERE / "digit_geometry_397b.json"


def kurtosis(x: torch.Tensor) -> float:
    x = x.float()
    mu, sd = x.mean(), x.std()
    return float((((x - mu) / (sd + 1e-9)) ** 4).mean())


def main() -> None:
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(HF_ID)

    def sid(w):
        for form in (w, " " + w):
            e = tok.encode(form, add_special_tokens=False)
            if len(e) == 1:
                return e[0], form
        return None, None

    groups = {
        "single_digit": [str(d) for d in range(10)],
        "multi_digit": ["12", "42", "100", "123", "2024", "1999", "007"],
        "number_word": ["one", "two", "three", "ten", "hundred", "thousand", "million"],
        "control_color": ["red", "blue", "green", "yellow", "black"],
        "control_animal": ["dog", "cat", "horse", "bird", "fish"],
    }
    ids, meta = {}, {}
    for g, ws in groups.items():
        ids[g] = []
        for w in ws:
            i, form = sid(w)
            if i is not None:
                ids[g].append(i); meta.setdefault(g, []).append((w, form, i))
    print({g: len(v) for g, v in ids.items()})

    # lm_head shard -> only the rows we need, then free it
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    idx = json.load(open(hf_hub_download(HF_ID, "model.safetensors.index.json")))
    shard = hf_hub_download(HF_ID, idx["weight_map"]["lm_head.weight"])
    all_ids = sorted({i for v in ids.values() for i in v})
    with safe_open(shard, framework="pt", device="cpu") as f:
        W = f.get_tensor("lm_head.weight")  # [vocab, d]
        U = {i: W[i].float().clone() for i in all_ids}
        vocab, d = W.shape
    del W
    print(f"U rows extracted: {len(U)}; vocab {vocab} d {d}")

    d0 = torch.load(LENS, map_location="cpu", weights_only=False)
    layers = d0["source_layers"]
    # digit "centroid" direction per group = mean unit unembedding row
    cent = {g: torch.stack([U[i] / U[i].norm() for i in ids[g]]).mean(0)
            for g in ids if ids[g]}

    # need full U to decode "what company" — load shard again, stream top-k per layer
    per_layer = {g: [] for g in ids}
    company = {}  # deep-layer top tokens for digit groups
    deep = layers[int(len(layers) * 0.8):]  # deepest 20% of fitted layers
    with safe_open(shard, framework="pt", device="cpu") as f:
        Wfull = f.get_tensor("lm_head.weight").float()  # 2GB transient
        for l in layers:
            J = d0["J"][l].float()  # [d, d] one layer
            for g, c in cent.items():
                r_all = Wfull @ (J @ c)           # [vocab] readout of the centroid
                per_layer[g].append((l, kurtosis(r_all)))
                if l in deep and g in ("single_digit", "multi_digit", "number_word"):
                    top = r_all.topk(15).indices.tolist()
                    company.setdefault(g, {})[l] = tok.batch_decode([[t] for t in top])
            del J
        del Wfull

    out = {"model": HF_ID, "lens": str(LENS), "n_prompts": d0.get("n_prompts"),
           "layers": layers, "tokens": {g: meta.get(g) for g in ids},
           "per_layer_kurtosis": {g: per_layer[g] for g in per_layer},
           "deep_company": company}
    json.dump(out, open(OUT, "w"), indent=1)

    print("\n=== WHERE DIGITS LIVE (mean centroid kurtosis by depth) ===")
    for g in ("single_digit", "multi_digit", "number_word",
              "control_color", "control_animal"):
        ks = [k for _, k in per_layer[g]]
        peak_l, peak_k = max(per_layer[g], key=lambda x: x[1])
        print(f"{g:14s} median κ {np.median(ks):6.1f} | peak κ {peak_k:6.1f} @ L{peak_l} "
              f"(early {ks[0]:.0f} / mid {ks[len(ks)//2]:.0f} / late {ks[-1]:.0f})")
    print("\n=== WHAT COMPANY DIGITS KEEP AT DEPTH (single_digit centroid) ===")
    for l, toks in list(company.get("single_digit", {}).items())[:4]:
        print(f"  L{l}: {toks[:12]}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
