"""Compare two Jacobian lenses of the SAME model via CKA of their token geometry.

Used for both tier-1 checks:
  - VALIDATION: --a our_gpt2_lens.pt --b <neuronpedia gpt2 lens>  (want high CKA)
  - STABILITY:  --a qwen4b_seed0.pt   --b qwen4b_seed1.pt         (want high CKA)

CKA is computed on the J-lens token directions D_l = U @ J_l per layer (needs the
model's unembedding), averaged over the source layers both lenses share. High mean
CKA => the two lenses agree (faithful / stable); low => fitting-dependent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.cka import linear_cka  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jacobian_lens"))
from unembed import load_unembedding  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="lens A .pt (local path or hf: repo/file)")
    ap.add_argument("--b", required=True, help="lens B .pt")
    ap.add_argument("--model", required=True, help="HF model id (for the unembedding U)")
    ap.add_argument("--n-probe", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jlens

    def load(spec: str):
        if spec.startswith("hf:"):
            from huggingface_hub import hf_hub_download
            repo, fn = spec[3:].split("::", 1)
            return jlens.JacobianLens.load(hf_hub_download(repo, filename=fn))
        return jlens.JacobianLens.load(spec)

    lens_a, lens_b = load(args.a), load(args.b)
    U = load_unembedding(args.model)
    vocab, _ = U.shape
    rng = np.random.default_rng(args.seed)
    probe = rng.choice(vocab, size=min(args.n_probe, vocab), replace=False)
    Up = U[probe]

    shared = sorted(set(lens_a.source_layers) & set(lens_b.source_layers))
    ckas = []
    for l in shared:
        Da = (Up @ lens_a.jacobians[l].float()).numpy()
        Db = (Up @ lens_b.jacobians[l].float()).numpy()
        ckas.append(linear_cka(Da, Db))
    ckas = np.array(ckas)
    print(f"shared layers: {len(shared)}  |  mean CKA(A,B) = {ckas.mean():.4f}  "
          f"(min {ckas.min():.3f}, max {ckas.max():.3f})")
    print("per-layer:", " ".join(f"{c:.2f}" for c in ckas))
    print("\nhigh (~>0.9) => lenses agree (faithful / stable); "
          "low => fitting-dependent (a real limitation to report).")


if __name__ == "__main__":
    main()
