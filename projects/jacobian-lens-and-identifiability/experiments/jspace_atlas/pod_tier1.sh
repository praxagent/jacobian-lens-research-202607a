#!/bin/bash
# Tier-1 lens fits on a RunPod GPU (A6000). Pre-reg: TIER1_PREREG.md.
# Correctness gate (gpt2) first; then 4 small-model fits, each isolated (one failure
# does not abort the rest). Lenses scp'd home for CPU analysis. HF token inline (gated gemma).
set -u
cd /workspace
[ -d repo ] || git clone --depth 40 https://github.com/praxagent/jacobian-lens-research-202607a repo
cd repo && git fetch -q && git checkout -q ARMSHA
pip -q install "git+https://github.com/anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e" \
  "transformers>=4.50" datasets accelerate safetensors 2>&1 | tail -2
python -c "import jlens; print('jlens installed OK')" || { echo JLENS_INSTALL_FAILED; exit 1; }
cd projects/jacobian-lens-and-identifiability/experiments/fit_our_own
mkdir -p /workspace/lenses
FIT="python -u fit_lens.py --n-prompts 100 --dim-batch 8 --max-seq-len 128 --seed 0 --device cuda"

step () { echo "===== $1 ====="; }

step "GATE: gpt2 fit (correctness) + loadable check"
$FIT --model gpt2 --corpus wikitext --out /workspace/lenses/gpt2_wiki.pt 2>&1 | grep -vE "Loading|it/s|Token indices|Repo card"
python -u -c "import torch; d=torch.load('/workspace/lenses/gpt2_wiki.pt',map_location='cpu',weights_only=False); J=d['J'] if 'J' in d else d; print('GATE_OK gpt2 lens layers', len(J))" || { echo GATE_FAILED; exit 1; }

step "ARM A: gemma-2-9b (model-vs-fit + release)"
$FIT --model google/gemma-2-9b --corpus wikitext --out /workspace/lenses/gemma2_9b_wiki.pt 2>&1 | grep -vE "Loading|it/s|Token indices|Repo card" || echo "ARM_A_FAILED"

step "ARM B/C baseline: qwen3-4b wikitext (length-matched)"
$FIT --model Qwen/Qwen3-4B --corpus wikitext --match-length --out /workspace/lenses/qwen4b_wiki.pt 2>&1 | grep -vE "Loading|it/s|Token indices|Repo card" || echo "ARM_B_WIKI_FAILED"

step "ARM B: qwen3-4b code (length-matched)"
$FIT --model Qwen/Qwen3-4B --corpus code --match-length --out /workspace/lenses/qwen4b_code.pt 2>&1 | grep -vE "Loading|it/s|Token indices|Repo card" || echo "ARM_B_CODE_FAILED"

step "ARM C sanity: can jlens backprop through the FP8 checkpoint?"
python -u -c "
import torch, transformers, jlens
m=transformers.AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B-FP8',torch_dtype='auto').to('cuda').eval()
t=transformers.AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-FP8')
cfg=m.config.to_dict(); print('FP8_CONFIG quant=', cfg.get('quantization_config'))
ids=t('def f(x): return x+1', return_tensors='pt').input_ids.to('cuda')
h=m(ids, output_hidden_states=True).hidden_states[10]; h.retain_grad()
m(ids).logits.sum().backward()
print('FP8_BACKWARD_OK grad_finite=', bool(torch.isfinite(h.grad).all()) if h.grad is not None else False)
" 2>&1 | grep -E "FP8_" || echo "FP8_SANITY_FAILED"

step "ARM C: qwen3-4b-fp8 (only if sanity passed; length-matched)"
$FIT --model Qwen/Qwen3-4B-FP8 --corpus wikitext --match-length --out /workspace/lenses/qwen4b_fp8_wiki.pt 2>&1 | grep -vE "Loading|it/s|Token indices|Repo card" || echo "ARM_C_FAILED"

step "receipts"
python -u -c "
import json,glob,os,hashlib,transformers,torch
r={}
for f in sorted(glob.glob('/workspace/lenses/*.pt')):
    sz=os.path.getsize(f)
    r[os.path.basename(f)]={'bytes':sz,'sha256':hashlib.sha256(open(f,'rb').read()).hexdigest()[:16]}
r['versions']={'transformers':transformers.__version__,'torch':torch.__version__}
json.dump(r,open('/workspace/lenses/tier1_receipt.json','w'),indent=1); print(json.dumps(r,indent=1))
"
ls -la /workspace/lenses/
echo TIER1_FITS_DONE
