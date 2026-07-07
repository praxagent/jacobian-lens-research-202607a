"""Sub-experiment A/B (CPU) — layer x layer CKA of J-lens token geometry.

Partial replication of the eliebak "J-lens CKA Explorer"
(https://eliebak.com/viz/jspace-open), using Anthropic's Apache-2.0 `jlens` +
Neuronpedia's pre-fitted lenses. Because the lens is already fitted, this is
cheap matrix algebra — runs on CPU, no GPU spend.

Question it bears on (the "oversold" audit): does a distinct mid-network
"workspace band" — a block of layers whose J-lens token geometry is
self-similar and set apart from early/late layers — actually show up? A clean
band supports the workspace story; a smeared/absent one (esp. at small scale)
is evidence the band is a large-model phenomenon, not a universal.

J-lens token direction for token t at layer l is the residual-space direction
that drives token t:  D_l = U @ J_l   (U = unembedding, J_l = lens transport).
We take a fixed random sample of vocab tokens as probes and compute CKA between
the per-layer (n_probe, d_model) geometries.

Run (CPU):
    uv run python cka_layers.py --model gpt2-small
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.cka import linear_cka  # noqa: E402

REPO = "neuronpedia/jacobian-lens"
# Neuronpedia slug -> (HF model id for the unembedding, lens filename)
MODELS = {
    "gpt2-small": ("openai-community/gpt2",
                   "gpt2-small/jlens/Salesforce-wikitext/gpt2-small_jacobian_lens.pt"),
    "pythia-70m-deduped": ("EleutherAI/pythia-70m-deduped",
                   "pythia-70m-deduped/jlens/Salesforce-wikitext/pythia-70m-deduped_jacobian_lens.pt"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-small", choices=list(MODELS))
    ap.add_argument("--n-probe", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jlens
    from transformers import AutoModelForCausalLM

    hf_id, lens_file = MODELS[args.model]
    print(f"model={args.model}  ({hf_id})")
    lens = jlens.JacobianLens.from_pretrained(REPO, filename=lens_file)
    print("lens:", lens)

    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32)
    U = model.get_output_embeddings().weight.detach().float()  # (vocab, d_model)
    vocab, d_model = U.shape
    print(f"unembed U = ({vocab}, {d_model})")

    rng = np.random.default_rng(args.seed)
    probe = rng.choice(vocab, size=min(args.n_probe, vocab), replace=False)
    Up = U[probe]  # (n_probe, d_model)

    layers = lens.source_layers
    geom = {}
    for l in layers:
        J = lens.jacobians[l].float()          # (d_model, d_model)
        geom[l] = (Up @ J).numpy()             # (n_probe, d_model): D_l = U @ J_l

    L = len(layers)
    M = np.eye(L)
    for i in range(L):
        for j in range(i + 1, L):
            M[i, j] = M[j, i] = linear_cka(geom[layers[i]], geom[layers[j]])

    print(f"\nlayer x layer CKA of J-lens token geometry ({L} source layers):")
    print("      " + " ".join(f"{l:4d}" for l in layers))
    for i, l in enumerate(layers):
        print(f"{l:4d}  " + " ".join(f"{M[i,j]:4.2f}" for j in range(L)))

    # Band summary: split into thirds; report mean within-third vs cross-third CKA.
    thirds = np.array_split(np.arange(L), 3)
    def block_mean(a, b):
        vals = [M[i, j] for i in a for j in b if i != j]
        return float(np.mean(vals)) if vals else 1.0
    early, mid, late = thirds
    print("\nblock-mean CKA (off-diagonal):")
    print(f"  within early {block_mean(early,early):.2f} | within MID {block_mean(mid,mid):.2f} "
          f"| within late {block_mean(late,late):.2f}")
    print(f"  early<->mid {block_mean(early,mid):.2f} | mid<->late {block_mean(mid,late):.2f} "
          f"| early<->late {block_mean(early,late):.2f}")
    mid_sep = block_mean(mid, mid) - 0.5 * (block_mean(early, mid) + block_mean(mid, late))
    print(f"\nmid-band separation (within-mid minus mid-to-neighbors) = {mid_sep:+.3f}")
    print("  >0 and sizeable => a distinct mid 'workspace' band; ~0 => no clean band.")
    np.save(Path(__file__).with_name(f"cka_{args.model}.npy"), M)


if __name__ == "__main__":
    main()
