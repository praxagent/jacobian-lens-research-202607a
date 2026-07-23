#!/bin/bash
# Tier-1 Arm C REDO on an FP8-capable GPU (RTX 4090 / H100, compute cap >= 8.9).
# The A6000 (cc 8.6) silently dequantized FP8 to bf16, voiding the first attempt.
# Here: verify FP8 truly applies, then fit BOTH bf16 and FP8 on the SAME GPU (no
# cross-GPU numerics confound) and compare. Aborts early if FP8 does not apply.
set -u
cd /workspace
[ -d repo ] || git clone --depth 40 https://github.com/praxagent/jacobian-lens-research-202607a repo
cd repo && git fetch -q && git checkout -q ARMSHA
# torch 2.7+ needed for torch.float8_e8m0fnu that the finegrained-fp8 kernel imports;
# driver here supports CUDA 13, so cu126 wheels run fine.
pip -q install --upgrade "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" \
  --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -1
pip -q install "git+https://github.com/anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e" \
  "transformers>=5,<6" "kernels==0.15.2" datasets accelerate safetensors 2>&1 | tail -2
python -c "import torch,torchvision,torchaudio,transformers,jlens; import transformers.modeling_utils; from transformers import AutoModelForCausalLM; assert hasattr(torch,"float8_e8m0fnu"),"no e8m0"; print("deps OK torch",torch.__version__)" || { echo DEPS_FAILED; exit 1; }
cd projects/jacobian-lens-and-identifiability/experiments/fit_our_own
mkdir -p /workspace/lenses

echo "===== FP8 capability + applies-genuinely gate ====="
python -u - <<'PY'
import torch, transformers, sys
cc=torch.cuda.get_device_capability(); print("compute capability", cc)
if cc < (8,9):
    print("FP8_UNSUPPORTED_GPU", cc); sys.exit(2)
import io, contextlib
buf=io.StringIO()
with contextlib.redirect_stderr(buf):
    mf=transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-FP8",torch_dtype="auto").to("cuda").eval()
warn=buf.getvalue()
mb=transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B",torch_dtype="auto").to("cuda").eval()
wb=mb.get_output_embeddings().weight.detach().float()
wf=mf.get_output_embeddings().weight.detach().float()
diff=float((wb-wf).abs().max())
print("fp8 dtype", mf.get_output_embeddings().weight.dtype, "| lm_head max|diff| vs bf16 =", diff)
print("dequant_warning:", "dequantiz" in warn.lower())
# backward-finiteness through FP8
t=transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-FP8")
ids=t("def f(x): return x+1\nprint(f(3))",return_tensors="pt").input_ids.to("cuda")
o=mf(ids,output_hidden_states=True); h=o.hidden_states[10]; h.retain_grad(); o.logits.sum().backward()
gf=bool(torch.isfinite(h.grad).all()) if h.grad is not None else False
print("backward_grad_finite:", gf)
if diff==0.0 or "dequantiz" in warn.lower():
    print("FP8_NOT_APPLIED (weights identical to bf16 or dequantized) -> Arm C still void here"); sys.exit(3)
print("FP8_APPLIES_OK")
PY
[ $? -ne 0 ] && { echo FP8_GATE_FAILED; exit 1; }

FIT="python -u fit_lens.py --n-prompts 100 --dim-batch 8 --max-seq-len 128 --seed 0 --match-length --corpus wikitext --device cuda"
echo "===== fit bf16 baseline (same GPU) ====="
$FIT --model Qwen/Qwen3-4B --out /workspace/lenses/fp8redo_qwen4b_bf16.pt 2>&1 | grep -vE "Loading|it/s|Repo card|deprecated" || echo BF16_FIT_FAILED
echo "===== fit FP8 (genuine, same GPU) ====="
$FIT --model Qwen/Qwen3-4B-FP8 --out /workspace/lenses/fp8redo_qwen4b_fp8.pt 2>&1 | grep -vE "Loading|it/s|Repo card|deprecated" || echo FP8_FIT_FAILED

echo "===== cross-lens CKA (bf16 vs genuine FP8) ====="
python -u - <<'PY'
import torch, json, sys, numpy as np
sys.path.insert(0,"../..")           # experiments/
sys.path.insert(0,"../../..")        # project root for common
from pathlib import Path
sys.path.insert(0,str(Path("/workspace/repo/projects/jacobian-lens-and-identifiability")))
from common.cka import linear_cka
sys.path.insert(0,str(Path("/workspace/repo/projects/jacobian-lens-and-identifiability/experiments/jacobian_lens")))
from shared_vocab import resolve_ids
import transformers
SH=json.load(open("/workspace/repo/projects/jacobian-lens-and-identifiability/experiments/jacobian_lens/shared_tokens.json"))["strings"]
m=transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B",torch_dtype="auto").eval()
U=m.get_output_embeddings().weight.detach().float(); ids=resolve_ids("Qwen/Qwen3-4B",SH)
Up=U[[ids[s] for s in SH if s in ids]]
def ds(p):
    J=torch.load(p,map_location="cpu",weights_only=False)["J"]
    return [ (Up@J[l].float()).to(torch.float16).numpy() for l in sorted(J) ]
Db=ds("/workspace/lenses/fp8redo_qwen4b_bf16.pt"); Df=ds("/workspace/lenses/fp8redo_qwen4b_fp8.pt")
cur=[linear_cka(Db[i].astype(np.float32),Df[i].astype(np.float32)) for i in range(len(Db))]
t=len(cur)//3
res={"early":float(np.mean(cur[:t])),"mid":float(np.mean(cur[t:2*t])),"late":float(np.mean(cur[2*t:])),
     "median":float(np.median(cur)),"n_layers":len(cur)}
json.dump({"armC_fp8_vs_bf16_GENUINE":res,"curve":cur},open("/workspace/lenses/fp8redo_result.json","w"),indent=1)
print("armC GENUINE fp8-vs-bf16:",res)
PY
echo TIER1_FP8_DONE
