"""Sub-experiment A/B (CPU) — layer x layer CKA of J-lens token geometry.

Partial replication of the eliebak "J-lens CKA Explorer"
(https://eliebak.com/viz/jspace-open), using Anthropic's Apache-2.0 `jlens` +
Neuronpedia's pre-fitted lenses. Fitting is already done, so this is cheap
matrix algebra — CPU, no GPU. Memory is dominated by the unembedding U + the
lens; we load ONLY U (not the whole model), so a high-RAM CPU box climbs the
whole size ladder with zero GPU spend.

Question (the "oversold" audit): does a distinct mid-network "workspace band"
show up, and at what scale does it emerge? `--null` is the confound control:
with random J_l, if CKA stays high the signal is just the shared unembedding U.

J-lens token direction for token t at layer l:  D_l = U @ J_l.

Everything is resolved from the Neuronpedia slug (HF id + lens file via the
lens's own config.yaml), so any of the 38 models works by name.

Run (CPU):
    uv run python cka_layers.py --slug gpt2-small
    uv run python cka_layers.py --slug qwen3-1.7b --null   # confound control
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.cka import linear_cka  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unembed import load_unembedding  # noqa: E402

REPO = "neuronpedia/jacobian-lens"
EMERGENCE_CSV = Path(__file__).with_name("emergence.csv")

# odd cases the regex can't parse; everything else comes from the slug (e.g. "4b")
_PARAM_OVERRIDE = {
    "gpt2-small": 0.124e9, "pythia-70m-deduped": 0.070e9,
    "gemma-4-e2b": 2.0e9, "gemma-4-e4b": 4.0e9,  # "effective" param sizes
}


def parse_params(slug: str) -> float:
    """Best-effort parameter count from a Neuronpedia slug. Log-x scale only
    needs order-of-magnitude, so approximate 'nb'/'nm' tokens are fine."""
    import re
    if slug in _PARAM_OVERRIDE:
        return _PARAM_OVERRIDE[slug]
    m = re.search(r"(\d+\.?\d*)\s*([bm])", slug.lower())
    if not m:
        return float("nan")
    val = float(m.group(1))
    return val * (1e9 if m.group(2) == "b" else 1e6)


def resolve(slug: str) -> tuple[str, str]:
    """Return (hf_model_id, lens_filename) for a Neuronpedia slug."""
    import yaml
    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi().list_repo_files(REPO) if f.startswith(slug + "/")]
    lens_file = next(f for f in files if f.endswith(".pt"))
    cfg_file = next(f for f in files if f.endswith("config.yaml"))
    cfg = yaml.safe_load(open(hf_hub_download(REPO, filename=cfg_file)))
    return cfg["hf_model_name"], lens_file


def band_stats(M: np.ndarray, layers: list[int]) -> dict:
    L = len(layers)
    thirds = np.array_split(np.arange(L), 3)

    def block_mean(a, b):
        vals = [M[i, j] for i in a for j in b if i != j]
        return float(np.mean(vals)) if vals else 1.0

    early, mid, late = thirds
    wm = block_mean(mid, mid)
    mid_sep = wm - 0.5 * (block_mean(early, mid) + block_mean(mid, late))
    return {
        "within_early": block_mean(early, early),
        "within_mid": wm,
        "within_late": block_mean(late, late),
        "early_mid": block_mean(early, mid),
        "mid_late": block_mean(mid, late),
        "early_late": block_mean(early, late),
        "mid_sep": mid_sep,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="gpt2-small")
    ap.add_argument("--n-probe", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--null", action="store_true",
                    help="confound control: replace J_l with scale-matched random matrices")
    ap.add_argument("--shared-probe", default=None,
                    help="path to shared_tokens.json (from shared_vocab.py): probe with "
                         "token STRINGS shared across all models' tokenizers — the "
                         "tokenizer-confound mitigation (Eleos). Results go to "
                         "emergence_shared.csv")
    args = ap.parse_args()

    # jlens no longer needed here: we torch.load the raw fp16 checkpoint directly
    from huggingface_hub import hf_hub_download

    hf_id, lens_file = resolve(args.slug)
    print(f"slug={args.slug}  hf={hf_id}  null={args.null}")
    # Load the RAW fp16 checkpoint ourselves. JacobianLens.load() casts the whole
    # dict to fp32 at load (peak ~32GB for the 70B: fp16 source + fp32 copies) —
    # we keep it fp16 and upcast one layer at a time instead.
    ckpt = torch.load(hf_hub_download(REPO, filename=lens_file),
                      map_location="cpu", weights_only=True)
    jacobians = {int(l): J for l, J in ckpt["J"].items()}  # fp16 as saved
    source_layers = sorted(jacobians)
    d_model_lens = next(iter(jacobians.values())).shape[0]
    print(f"lens: {len(source_layers)} source layers, d_model={d_model_lens}, "
          f"dtype={next(iter(jacobians.values())).dtype} (raw, no fp32 dict)")

    U = load_unembedding(hf_id)  # (vocab, d_model) float32 CPU
    vocab, d_model = U.shape
    print(f"unembed U = ({vocab}, {d_model})")

    if args.shared_probe:
        import json
        from shared_vocab import resolve_ids
        spec = json.load(open(args.shared_probe))
        ids_map = resolve_ids(hf_id, spec["strings"])
        probe = np.array(sorted(ids_map.values()), dtype=np.int64)
        print(f"shared-probe: {len(probe)}/{len(spec['strings'])} strings resolved "
              f"to single tokens in this tokenizer")
    else:
        rng = np.random.default_rng(args.seed)
        probe = rng.choice(vocab, size=min(args.n_probe, vocab), replace=False)
    Up = U[probe]  # (n_probe, d_model)
    del U

    layers = source_layers
    # Memory discipline (70B = 80 x 8192^2): lens stays fp16 as loaded; upcast ONE
    # layer to fp32 transiently for the matmul; store its geometry fp16; free the
    # layer if nulling. Peak ~16.5GB for the 70B on a 30GB box.
    Up_t = torch.tensor(Up) if not isinstance(Up, torch.Tensor) else Up
    Up_t = Up_t.float()
    geom = {}
    for l in layers:
        J = jacobians[l]
        if args.null:
            J = torch.randn(d_model, d_model) * J.float().std()  # scale-matched random
        geom[l] = (Up_t @ J.float()).to(torch.float16)  # fp32 compute, fp16 store

    L = len(layers)
    M = np.eye(L)
    for i in range(L):
        gi = geom[layers[i]].float().numpy()
        for j in range(i + 1, L):
            M[i, j] = M[j, i] = linear_cka(gi, geom[layers[j]].float().numpy())
        del gi

    print(f"\nlayer x layer CKA ({L} source layers){' [NULL]' if args.null else ''}:")
    print("      " + " ".join(f"{l:4d}" for l in layers))
    for i, l in enumerate(layers):
        print(f"{l:4d}  " + " ".join(f"{M[i,j]:4.2f}" for j in range(L)))

    s = band_stats(M, layers)
    print(f"\nwithin early/MID/late = {s['within_early']:.2f} / {s['within_mid']:.2f} "
          f"/ {s['within_late']:.2f}")
    print(f"mid-band separation = {s['mid_sep']:+.3f}  "
          f"(>0 sizeable => distinct workspace band; ~0 => none)")

    # append to the emergence ledger — real / null / shared-probe runs to separate
    # files, so the plot can overlay them.
    if args.null:
        out = EMERGENCE_CSV.with_name("emergence_null.csv")
    elif args.shared_probe:
        out = EMERGENCE_CSV.with_name("emergence_shared.csv")
    else:
        out = EMERGENCE_CSV
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["slug", "hf_id", "params", "d_model", "n_layers",
                        "mid_sep", "within_mid", "within_early", "within_late"])
        w.writerow([args.slug, hf_id, f"{parse_params(args.slug):.3e}", d_model, L,
                    f"{s['mid_sep']:.4f}", f"{s['within_mid']:.4f}",
                    f"{s['within_early']:.4f}", f"{s['within_late']:.4f}"])
    print(f"appended to {out.name}")


if __name__ == "__main__":
    main()
