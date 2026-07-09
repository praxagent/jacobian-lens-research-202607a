"""Validate jlens.fit under transformers native tensor parallelism.
Launch: torchrun --nproc_per_node=2 tp_test.py   (TP_MODEL env overrides the model)
The go/no-go: does jlens.fit's retain_graph + repeated-backward complete under tp_plan=auto
(DTensor collectives in the backward)? If yes, TP unlocks ~world_size× compute for the 397B.
"""
import os, sys, traceback
from pathlib import Path
import torch, transformers
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "jacobian_lens"))
sys.path.insert(0, str(HERE))

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(LOCAL)
if WORLD > 1 and not dist.is_initialized():
    dist.init_process_group(backend="nccl")

def log(*a):
    if RANK == 0:
        print("[TP]", *a, flush=True)

def done(marker):
    if RANK == 0:
        print(marker, flush=True)

mid = os.environ.get("TP_MODEL", "Qwen/Qwen3-4B")
try:
    import jlens
    from fit_lens import load_corpus
    log(f"world={WORLD}  model={mid}  loading with tp_plan=auto ...")
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        mid, torch_dtype=torch.bfloat16, tp_plan="auto").eval()
    tok = transformers.AutoTokenizer.from_pretrained(mid)
    model = jlens.from_hf(hf, tok, compile=False)
    log(f"loaded; n_layers={model.n_layers}; fitting n=8 (THE test — backward under TP) ...")
    prompts = load_corpus(8, 0, corpus="wikitext")
    lens = jlens.fit(model, prompts, dim_batch=4, max_seq_len=128)
    log(f"jlens.fit COMPLETED under TP — layers={len(lens.source_layers)}")
    if RANK == 0:
        import numpy as np
        from cka_layers import band_stats
        from common.cka import linear_cka
        U = hf.get_output_embeddings().weight.detach()
        if hasattr(U, "full_tensor"):
            U = U.full_tensor()
        U = U.float().cpu()
        rng = np.random.default_rng(0)
        probe = rng.choice(U.shape[0], size=min(4096, U.shape[0]), replace=False)
        Up = U[probe]
        layers = lens.source_layers
        geom = {}
        for l in layers:
            J = lens.jacobians[l]
            if hasattr(J, "full_tensor"):
                J = J.full_tensor()
            geom[l] = (Up @ J.float().cpu()).numpy()
        L = len(layers); M = np.eye(L)
        for i in range(L):
            for j in range(i + 1, L):
                M[i, j] = M[j, i] = linear_cka(geom[layers[i]], geom[layers[j]])
        s = band_stats(M, layers)
        log(f"mid_sep = {s['mid_sep']:+.4f}  (qwen3-4b ref ~0.06 -> validates numerics too)")
        done("TP_TEST_OK")
except Exception as e:
    if RANK == 0:
        print("[TP] FAILED:", type(e).__name__, str(e)[:500], flush=True)
        traceback.print_exc()
    done("TP_TEST_FAIL")
finally:
    done("TP_TEST_DONE")
    if WORLD > 1 and dist.is_initialized():
        dist.destroy_process_group()
