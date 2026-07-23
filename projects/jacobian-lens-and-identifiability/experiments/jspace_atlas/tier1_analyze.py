"""Tier-1 analysis (runs ON THE POD, where models+lenses live).

Per fitted lens: within-model shared-vocab CKA map + band stats. Then the arm A/B/C
comparisons. Also checks FP8 validity by comparing the actual loaded weights of
Qwen3-4B vs Qwen3-4B-FP8 (quant config was None; must confirm the checkpoint really
differs before trusting arm C). Small outputs to /workspace/tier1_analysis/.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "jacobian_lens"))
from common.cka import linear_cka          # noqa
from cka_layers import band_stats          # noqa
from shared_vocab import resolve_ids       # noqa

SHARED = json.load(open(HERE.parent / "jacobian_lens/shared_tokens.json"))["strings"]
LENS = "/workspace/lenses"
OUT = Path("/workspace/tier1_analysis"); OUT.mkdir(exist_ok=True)
N_PROBE = 4096


def U_shared(hf_id):
    import transformers
    m = transformers.AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype="auto").eval()
    U = m.get_output_embeddings().weight.detach().float()
    ids = resolve_ids(hf_id, SHARED)
    strings = [s for s in SHARED if s in ids]
    rows = U[[ids[s] for s in strings]]
    del m
    return rows, strings


def dshared(lens_pt, Up):
    d = torch.load(lens_pt, map_location="cpu", weights_only=False)
    J = d["J"]; layers = sorted(J)
    return [ (Up @ J[l].float()).to(torch.float16).numpy() for l in layers ], layers


def within_map(D):
    L = len(D); M = np.eye(L)
    for i in range(L):
        gi = D[i].astype(np.float32)
        for j in range(i+1, L):
            M[i, j] = M[j, i] = linear_cka(gi, D[j].astype(np.float32))
    return M


def cross_curve(Da, Db):
    # same model -> same layer count -> matched depth = same index
    return [linear_cka(Da[i].astype(np.float32), Db[i].astype(np.float32))
            for i in range(min(len(Da), len(Db)))]


res = {}

# ---- Arm A: gemma-2-9b our fit ----
print("== Arm A: gemma-2-9b ==", flush=True)
Ug, _ = U_shared("google/gemma-2-9b")
Dg, gl = dshared(f"{LENS}/gemma2_9b_wiki.pt", Ug)
Mg = within_map(Dg); bs = band_stats(Mg, gl)
tri = Mg[np.triu_indices_from(Mg, 1)]
res["armA_gemma2_9b_ours"] = {"mid_sep": bs["mid_sep"], "offdiag_cka_median": float(np.median(tri)),
                              "offdiag_cka_min": float(tri.min()), "n_layers": len(gl)}
np.savez_compressed(OUT/"gemma2_9b_ours.npz", cka=Mg, layers=np.array(gl))
print(res["armA_gemma2_9b_ours"], flush=True)
del Ug, Dg

# ---- Arm B + C: qwen3-4b wiki / code / fp8 (shared U) ----
print("== Arm B/C: qwen3-4b ==", flush=True)
Uq, _ = U_shared("Qwen/Qwen3-4B")
Dw, ql = dshared(f"{LENS}/qwen4b_wiki.pt", Uq)
Dc, _  = dshared(f"{LENS}/qwen4b_code.pt", Uq)
Df, _  = dshared(f"{LENS}/qwen4b_fp8_wiki.pt", Uq)
armB = cross_curve(Dw, Dc)
armC = cross_curve(Dw, Df)
def band3(curve):
    t = len(curve)//3
    return {"early": float(np.mean(curve[:t])), "mid": float(np.mean(curve[t:2*t])),
            "late": float(np.mean(curve[2*t:])), "median": float(np.median(curve))}
res["armB_code_vs_wiki"] = band3(armB)
res["armC_fp8_vs_wiki"] = band3(armC)
np.savez_compressed(OUT/"qwen_curves.npz", armB=np.array(armB), armC=np.array(armC), layers=np.array(ql))
print("armB code-vs-wiki:", res["armB_code_vs_wiki"], flush=True)
print("armC fp8-vs-wiki :", res["armC_fp8_vs_wiki"], flush=True)

# ---- FP8 validity: do the two checkpoints' weights actually differ? ----
print("== FP8 validity ==", flush=True)
import transformers
mb = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype="auto").eval()
mf = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-FP8", torch_dtype="auto").eval()
wb = mb.get_output_embeddings().weight.detach().float()
wf = mf.get_output_embeddings().weight.detach().float()
diff = (wb - wf).abs()
res["fp8_validity"] = {"bf16_dtype": str(mb.get_output_embeddings().weight.dtype),
                       "fp8_dtype": str(mf.get_output_embeddings().weight.dtype),
                       "lm_head_max_abs_diff": float(diff.max()),
                       "lm_head_mean_abs_diff": float(diff.mean()),
                       "identical": bool(diff.max() == 0)}
print("fp8_validity:", res["fp8_validity"], flush=True)

(OUT/"tier1_results.json").write_text(json.dumps(res, indent=1, default=float))
print("TIER1_ANALYSIS_DONE", flush=True)
