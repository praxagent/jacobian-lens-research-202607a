"""Fit a Jacobian lens under TENSOR PARALLELISM (torchrun + tp_plan="auto").

Unlike fit_at_scale.py's device_map="auto" (naive layer-sharding: 1 GPU active at a time,
memory-only), tp_plan shards every weight matrix so all GPUs compute each layer together —
~world_size× faster. Requires torch>=2.5. Validated: jlens.fit's retain_graph + repeated
backward survives DTensor collectives (qwen3-4b, 2 GPUs).

Launch:
    torchrun --nproc_per_node=8 tp_fit.py --model Qwen/Qwen3.5-397B-A17B \
        --backbone-path model.language_model --n-prompts 24 --out lenses/qwen35_397b.pt
"""
import os, sys, argparse, traceback
from pathlib import Path

import numpy as np
import torch
import transformers
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "jacobian_lens"))
sys.path.insert(0, str(HERE))

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL = int(os.environ.get("LOCAL_RANK", "0"))


def log(*a):
    if RANK == 0:
        print("[TPFIT]", *a, flush=True)


def _full(t):
    return t.full_tensor() if hasattr(t, "full_tensor") else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backbone-path", default=None,
                    help="text-backbone module path for multimodal wrappers (e.g. model.language_model)")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--dim-batch", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-probe", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.cuda.set_device(LOCAL)
    if WORLD > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    import jlens
    from jlens import Layout
    from fit_lens import load_corpus

    log(f"world={WORLD}  model={args.model}  tp_plan=auto  backbone={args.backbone_path}")
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    if args.backbone_path:
        cfg = transformers.AutoConfig.from_pretrained(args.model)
        cls = getattr(transformers, cfg.architectures[0])
        log(f"multimodal TP load: {cls.__name__}")
        hf = cls.from_pretrained(args.model, torch_dtype=torch.bfloat16, tp_plan="auto").eval()
        model = jlens.from_hf(hf, tok, compile=False, layout=Layout(path=args.backbone_path))
    else:
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, tp_plan="auto").eval()
        model = jlens.from_hf(hf, tok, compile=False)

    log(f"loaded; n_layers={model.n_layers}; corpus n={args.n_prompts}; fitting (all GPUs) ...")
    prompts = load_corpus(args.n_prompts, args.seed, corpus="wikitext")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    lens = jlens.fit(model, prompts, dim_batch=args.dim_batch, max_seq_len=args.max_seq_len,
                     checkpoint_path=str(out.with_suffix(".ckpt")))
    log("jlens.fit COMPLETE under TP")

    if RANK == 0:
        lens.save(str(out))
        from cka_layers import band_stats
        from common.cka import linear_cka
        U = _full(hf.get_output_embeddings().weight.detach()).float().cpu()
        rng = np.random.default_rng(args.seed)
        probe = rng.choice(U.shape[0], size=min(args.n_probe, U.shape[0]), replace=False)
        Up = U[probe]
        layers = lens.source_layers
        geom = {l: (Up @ _full(lens.jacobians[l]).float().cpu()).numpy() for l in layers}
        L = len(layers); M = np.eye(L)
        for i in range(L):
            for j in range(i + 1, L):
                M[i, j] = M[j, i] = linear_cka(geom[layers[i]], geom[layers[j]])
        s = band_stats(M, layers)
        print(f"\n=== 397B TP BAND RESULT: {args.model} ===", flush=True)
        print(f"layers={L}  n_prompts={args.n_prompts}  mid_sep = {s['mid_sep']:+.4f}  "
              f"(within early/MID/late = {s['within_early']:.3f}/{s['within_mid']:.3f}/{s['within_late']:.3f})",
              flush=True)
        import json
        json.dump({"model": args.model, "n_layers": L, "n_prompts": args.n_prompts,
                   "mid_sep": s["mid_sep"], **s}, open(out.with_suffix(".band.json"), "w"), indent=1)
        print("TPFIT_DONE", flush=True)

    if WORLD > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if RANK == 0:
            print("[TPFIT] FAILED:", type(e).__name__, str(e)[:600], flush=True)
            traceback.print_exc()
            print("TPFIT_DONE", flush=True)
        raise
