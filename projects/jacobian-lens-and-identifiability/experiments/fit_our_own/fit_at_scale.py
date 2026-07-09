"""Frontier fit — fit a lens on a model too big for one GPU, then compute its band.

Answers the audit's top limitation: is the sub-70B family-dependence a transient? We
fit our OWN lens on a frontier open model (no Neuronpedia lens exists above ~70B),
shard it across GPUs with device_map="auto" (jlens.fit is device-aware; only compile=True
is incompatible with sharding, per jlens.hf), then compute our mid_sep band statistic on
the result — directly comparable to the sweep.

Target: a large Qwen (the family with strong bands at 2–27B: 0.15–0.20). If the band
persists/strengthens at frontier scale, the pattern is not a small-model transient; if it
collapses, that's the honest headline caveat, now measured.

Recipe (Nanda's commentary): n≈16–25 prompts, Jacobians to the penultimate layer, cost
O(n·d_model) backward passes — ~1h on 8×H200 for a ~400B MoE.

Run (multi-GPU):
    uv run python fit_at_scale.py --model Qwen/Qwen3.5-397B-A17B --n-prompts 16 --out lenses/qwen397b.pt
    # validation (cheap, force a small model across 2 GPUs):
    uv run python fit_at_scale.py --model Qwen/Qwen3-14B --n-prompts 8 --out lenses/val14b.pt --max-memory-gb 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "jacobian_lens"))
from cka_layers import band_stats  # noqa: E402
sys.path.insert(0, str(HERE))
from fit_lens import load_corpus  # noqa: E402  (wikitext loader, seed-shuffled)
sys.path.insert(0, str(HERE.parents[1]))
from common.cka import linear_cka  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id (frontier)")
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--dim-batch", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-probe", type=int, default=4096)
    ap.add_argument("--max-memory-gb", type=int, default=0,
                    help="if >0, cap per-GPU memory (forces sharding for validation)")
    ap.add_argument("--attn-impl", default="eager",
                    help="attention backend. 'eager' = pure-PyTorch, robust for jlens's "
                         "retained-graph + repeated-backward under device_map sharding "
                         "(SDPA/flash backward kernels can index-assert in that pattern).")
    ap.add_argument("--backbone-path", default=None,
                    help="for multimodal wrappers (e.g. Qwen3.5-397B-A17B = "
                         "Qwen3_5MoeForConditionalGeneration), the module path to the text "
                         "backbone, e.g. 'model.language_model'. AutoModelForCausalLM expects "
                         "model.layers.* but the checkpoint stores it under "
                         "model.language_model.layers.* with NO conversion mapping, so it "
                         "would SILENTLY load uninitialized text weights. This loads the "
                         "checkpoint's real arch class (keys match) + an explicit jlens Layout.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import jlens
    import transformers

    n_gpu = torch.cuda.device_count()
    print(f"model={args.model}  GPUs={n_gpu}  "
          f"({[torch.cuda.get_device_name(i) for i in range(n_gpu)]})")

    # PURE-GPU sharding only. accelerate CPU-offload hooks break jlens.fit
    # (retain_graph + repeated backward passes over offloaded layers →
    # device-side index asserts), and the real 8×H200 run never offloads
    # anyway (1128GB >> 794GB), so we never put layers on CPU.
    kw = dict(torch_dtype=torch.bfloat16, device_map="auto",
              attn_implementation=args.attn_impl)
    if args.max_memory_gb > 0:  # validation: cap per-GPU to FORCE a multi-GPU split
        kw["max_memory"] = {i: f"{args.max_memory_gb}GiB" for i in range(n_gpu)}
        # deliberately NO "cpu" key — force pure-GPU sharding, error if it won't fit
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    if args.backbone_path:
        # Multimodal wrapper: load the checkpoint's real arch class (keys match) and point
        # jlens at the text backbone. Verified on meta device for Qwen3.5-397B-A17B:
        # Layout(path='model.language_model') -> 60 layers, d_model 4096, lm_head resolves.
        from jlens import Layout
        cfg = transformers.AutoConfig.from_pretrained(args.model)
        cls = getattr(transformers, cfg.architectures[0])
        print(f"multimodal load: {cls.__name__}  backbone={args.backbone_path}")
        hf = cls.from_pretrained(args.model, **kw).eval()
        model = jlens.from_hf(hf, tok, compile=False, layout=Layout(path=args.backbone_path))
    else:
        hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()
        model = jlens.from_hf(hf, tok, compile=False)  # compile OFF: required w/ device_map

    prompts = load_corpus(args.n_prompts, args.seed, corpus="wikitext")
    print(f"corpus: {len(prompts)} prompts; fitting (this is the slow part)...")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    lens = jlens.fit(model, prompts, dim_batch=args.dim_batch,
                     max_seq_len=args.max_seq_len,
                     checkpoint_path=str(out.with_suffix(".ckpt")))
    lens.save(str(out))
    print(f"saved lens -> {out}  ({lens})")

    # --- band statistic on the fitted lens (same as the sweep) ---
    U = hf.get_output_embeddings().weight.detach().float().cpu()
    vocab, d_model = U.shape
    rng = np.random.default_rng(args.seed)
    probe = rng.choice(vocab, size=min(args.n_probe, vocab), replace=False)
    Up = U[probe]
    layers = lens.source_layers
    geom = {l: (Up @ lens.jacobians[l].float().cpu()).numpy() for l in layers}
    L = len(layers); M = np.eye(L)
    for i in range(L):
        for j in range(i + 1, L):
            M[i, j] = M[j, i] = linear_cka(geom[layers[i]], geom[layers[j]])
    s = band_stats(M, layers)
    print(f"\n=== FRONTIER BAND RESULT: {args.model} ===")
    print(f"params≈ (see model card) | layers={L} | d_model={d_model}")
    print(f"mid_sep = {s['mid_sep']:+.4f}  "
          f"(within early/MID/late = {s['within_early']:.3f}/{s['within_mid']:.3f}/{s['within_late']:.3f})")
    print("Compare to Qwen3/3.5 at 2–27B (0.15–0.20): does the band hold at frontier scale?")
    import json
    json.dump({"model": args.model, "d_model": d_model, "n_layers": L,
               "mid_sep": s["mid_sep"], **s}, open(out.with_suffix(".band.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
