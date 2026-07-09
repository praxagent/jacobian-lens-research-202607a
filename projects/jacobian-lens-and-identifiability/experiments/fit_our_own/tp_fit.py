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


# Hand-written TP plan for Qwen3.5-MoE checkpoints (2 KV heads => the shipped colwise
# k/v plan can't run at ws>2, but the 807GB model needs ws>=6; upstream: by-design,
# #40953). Attention: colwise_gather_output = weight sharded for MEMORY, output
# all-gathered so reshapes see config-shaped heads (KV-head-safe at any world size).
# Experts (96% of params): genuinely 8-way sharded = the COMPUTE win. CPU-validated in
# tp_smoke_cpu.py: forward parity 4.2e-7, retain_graph repeated-backward grad parity.
# Keys carry the CondGen wrapper prefix; _strip_prefix() adapts them for plain models.
TP_PLAN_QWEN35_MOE = {
    "model.language_model.layers.*.self_attn.q_proj": "colwise_gather_output",
    "model.language_model.layers.*.self_attn.k_proj": "colwise_gather_output",
    "model.language_model.layers.*.self_attn.v_proj": "colwise_gather_output",
    "model.language_model.layers.*.self_attn.o_proj": "colwise_gather_output",
    "model.language_model.layers.*.linear_attn.in_proj_qkv": "colwise_gather_output",
    "model.language_model.layers.*.linear_attn.in_proj_z": "colwise_gather_output",
    "model.language_model.layers.*.linear_attn.in_proj_b": "colwise_gather_output",
    "model.language_model.layers.*.linear_attn.in_proj_a": "colwise_gather_output",
    "model.language_model.layers.*.linear_attn.out_proj": "colwise_gather_output",
    "model.language_model.layers.*.mlp.experts.gate_up_proj": "packed_colwise",
    "model.language_model.layers.*.mlp.experts.down_proj": "rowwise",
    "model.language_model.layers.*.mlp.experts": "moe_tp_experts",
    "model.language_model.layers.*.mlp.shared_expert.gate_proj": "colwise",
    "model.language_model.layers.*.mlp.shared_expert.up_proj": "colwise",
    "model.language_model.layers.*.mlp.shared_expert.down_proj": "rowwise",
    "lm_head": "colwise_gather_output",
}


# Dense qwen3_5 (e.g. Qwen3.5-0.8B — the cheap NCCL validator): same KV-head-safe
# attention treatment, standard colwise/rowwise MLP. Exercises the exact
# colwise_gather_output mechanics the 397B plan relies on.
TP_PLAN_QWEN35_DENSE = {
    "model.layers.*.self_attn.q_proj": "colwise_gather_output",
    "model.layers.*.self_attn.k_proj": "colwise_gather_output",
    "model.layers.*.self_attn.v_proj": "colwise_gather_output",
    "model.layers.*.self_attn.o_proj": "colwise_gather_output",
    "model.layers.*.linear_attn.in_proj_qkv": "colwise_gather_output",
    "model.layers.*.linear_attn.in_proj_z": "colwise_gather_output",
    "model.layers.*.linear_attn.in_proj_b": "colwise_gather_output",
    "model.layers.*.linear_attn.in_proj_a": "colwise_gather_output",
    "model.layers.*.linear_attn.out_proj": "colwise_gather_output",
    "model.layers.*.mlp.gate_proj": "colwise",
    "model.layers.*.mlp.up_proj": "colwise",
    "model.layers.*.mlp.down_proj": "rowwise",
    "lm_head": "colwise_gather_output",
}


def _plan_for(model_type: str, backbone_path: str | None) -> "str | dict":
    """Pick the TP plan: custom dicts for qwen3_5(_moe), else transformers' auto."""
    if model_type == "qwen3_5":
        return TP_PLAN_QWEN35_DENSE
    if model_type != "qwen3_5_moe":
        return "auto"
    if backbone_path:  # CondGen wrapper — keys already carry model.language_model.
        return TP_PLAN_QWEN35_MOE
    # plain Qwen3_5MoeForCausalLM: same plan, module paths are model.layers.*
    return {k.replace("model.language_model.", "model."): v
            for k, v in TP_PLAN_QWEN35_MOE.items()}


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
    ap.add_argument("--source-layers", default=None,
                    help="comma-separated layer indices to fit (default: all). Cost note: "
                         "backward count doesn't depend on this; only raising min(layer) "
                         "shortens each backward (graph is rooted at min(source_layers)).")
    ap.add_argument("--tp-plan", default=None, choices=[None, "auto", "custom"],
                    help="override plan selection (default: custom for qwen3_5_moe, else auto)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.cuda.set_device(LOCAL)
    if WORLD > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    import jlens
    from jlens import Layout
    from fit_lens import load_corpus

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    cfg = transformers.AutoConfig.from_pretrained(args.model)
    plan = _plan_for(cfg.model_type, args.backbone_path)
    if args.tp_plan == "auto":
        plan = "auto"
    elif args.tp_plan == "custom" and isinstance(plan, str):
        raise SystemExit(f"no custom plan defined for model_type={cfg.model_type}")
    log(f"world={WORLD}  model={args.model}  model_type={cfg.model_type}  "
        f"tp_plan={'custom dict (%d keys)' % len(plan) if isinstance(plan, dict) else plan}  "
        f"backbone={args.backbone_path}")
    if args.backbone_path:
        cls = getattr(transformers, cfg.architectures[0])
        log(f"multimodal TP load: {cls.__name__}")
        hf = cls.from_pretrained(args.model, torch_dtype=torch.bfloat16, tp_plan=plan).eval()
        model = jlens.from_hf(hf, tok, compile=False, layout=Layout(path=args.backbone_path))
    else:
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, tp_plan=plan).eval()
        model = jlens.from_hf(hf, tok, compile=False)

    src_layers = ([int(x) for x in args.source_layers.split(",")]
                  if args.source_layers else None)
    log(f"loaded; n_layers={model.n_layers}; corpus n={args.n_prompts}; "
        f"dim_batch={args.dim_batch} -> {args.n_prompts}*ceil(d/{args.dim_batch}) traversals; "
        f"fitting (all GPUs) ...")
    prompts = load_corpus(args.n_prompts, args.seed, corpus="wikitext")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = out.with_suffix(".ckpt")
    # Resume footgun guard: jlens's checkpoint validates source_layers/target_layer but
    # NOT max_seq_len — resuming with a different value silently mixes estimators.
    meta_path = out.with_suffix(".fitmeta.json")
    import json as _json
    if ckpt_path.exists() and meta_path.exists():
        prev = _json.load(open(meta_path))
        if prev.get("max_seq_len") != args.max_seq_len:
            raise SystemExit(f"RESUME GUARD: checkpoint was fit with max_seq_len="
                             f"{prev.get('max_seq_len')}, got {args.max_seq_len}. "
                             f"Delete {ckpt_path} or match it.")
    if RANK == 0:
        _json.dump({"max_seq_len": args.max_seq_len, "seed": args.seed,
                    "model": args.model}, open(meta_path, "w"))
    fit_kw = dict(dim_batch=args.dim_batch, max_seq_len=args.max_seq_len,
                  checkpoint_path=str(ckpt_path))
    if src_layers is not None:
        fit_kw["source_layers"] = src_layers
    lens = jlens.fit(model, prompts, **fit_kw)
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
