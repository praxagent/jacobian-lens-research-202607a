#!/bin/bash
# One pod session (lesson 18: one download, one setup) for two cheap GPU items, 2026-09-05:
#   1. qwen3.5-0.8b float32 perturbation C run (per-prompt receipts for the float32 interval;
#      the July attempt OOMed in lm_head on a 24GB card). Runner: geometry_causality.
#   2. Corpus dependence at 8B: llama3.1-8b fitted on WikiText seed 0, WikiText seed 1, code
#      (length-matched, 100 prompts, same recipe as the three small models), GATED on a timing
#      probe (lesson 7: measure one unit of work and multiply before committing).
# Usage on the pod (HF_TOKEN passed inline by the launching ssh command, never stored):
#   HF_TOKEN=... bash pod_corpus8b.sh <git-ref> [MAX_FIT_HOURS]
set -u
REF="${1:-main}"; MAX_FIT_HOURS="${2:-6}"
cd /workspace
[ -d repo ] || git clone --depth 60 https://github.com/praxagent/jacobian-lens-research-202607a repo
cd repo && git fetch -q && git checkout -q "$REF"
# torch trio first (transformers 5.x needs torch>=2.5; base image ships 2.4.1), then jlens.
pip -q install --upgrade "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
  --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
pip -q install "git+https://github.com/anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e" \
  "transformers>=5,<6" datasets accelerate safetensors 2>&1 | tail -2
python -c "import torch,torchvision,torchaudio,transformers,jlens
import transformers.modeling_utils
from transformers import AutoModelForCausalLM
print('deps OK torch',torch.__version__,'tf',transformers.__version__)" || { echo DEPS_INSTALL_FAILED; exit 1; }
mkdir -p /workspace/out /workspace/lenses
PROJ=/workspace/repo/projects/jacobian-lens-and-identifiability/experiments
step () { echo; echo "===== $1  ($(date -u +%H:%M:%S))  ====="; }

step "GATE: gpt2 fit (correctness gate for the fit path)"
cd $PROJ/fit_our_own
python -u fit_lens.py --model gpt2 --n-prompts 4 --dim-batch 8 --max-seq-len 128 --seed 0 \
  --corpus wikitext --match-length --device cuda --out /workspace/lenses/gate_gpt2.pt 2>&1 \
  | grep -vE "Loading|it/s|Token indices|Repo card" | tail -3
python -u -c "import torch; d=torch.load('/workspace/lenses/gate_gpt2.pt',map_location='cpu',weights_only=False); print('GATE_OK layers', len(d['J']))" || { echo GATE_FAILED; exit 1; }

step "ITEM 1: qwen3.5-0.8b perturbation C run, float32, calib-arm local, 200 frozen prompts"
cd $PROJ/geometry_causality
python -u run_geometry_causality.py --model Qwen/Qwen3.5-0.8B --slug qwen3.5-0.8b --device cuda \
  --dtype float32 --calib-arm local --n-prompts 200 --out /workspace/out/qwen3.5-0.8b_C32.json 2>&1 \
  | grep -vE "Loading|it/s|Token indices|Repo card" | tail -8
python -u -c "
import json; d=json.load(open('/workspace/out/qwen3.5-0.8b_C32.json'))
pl=d['per_layer']; l0=sorted(pl,key=int)[0]
print('ITEM1_RECEIPT layers',len(pl),'prompts',len(pl[l0]['kl_aligned']),'dtype',d.get('dtype'),'gates',d.get('gates'))" || echo ITEM1_FAILED

step "ITEM 2a: llama3.1-8b timing probe (2 prompts, wikitext, length-matched)"
cd $PROJ/fit_our_own
T0=$(date +%s)
python -u fit_lens.py --model meta-llama/Llama-3.1-8B --n-prompts 2 --dim-batch 8 --max-seq-len 128 \
  --seed 0 --corpus wikitext --match-length --device cuda --out /workspace/lenses/probe_llama8b.pt 2>&1 \
  | grep -vE "Loading|it/s|Token indices|Repo card" | tail -4
T1=$(date +%s); PROBE_S=$((T1-T0))
# the probe includes the model download (~16GB) once; fit time per prompt is what we extrapolate.
# fit_lens prints nothing per prompt we can rely on, so time a second 2-prompt fit with a warm cache.
T2=$(date +%s)
python -u fit_lens.py --model meta-llama/Llama-3.1-8B --n-prompts 2 --dim-batch 8 --max-seq-len 128 \
  --seed 1 --corpus wikitext --match-length --device cuda --out /workspace/lenses/probe2_llama8b.pt 2>&1 \
  | grep -vE "Loading|it/s|Token indices|Repo card" | tail -2
T3=$(date +%s); WARM_S=$((T3-T2))
python - "$PROBE_S" "$WARM_S" "$MAX_FIT_HOURS" <<'PY'
import sys, json
probe, warm, maxh = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
load = max(warm - 0, 1)                 # warm run = load + 2 prompts
per_prompt = max((warm) / 2.0, 1e-9)    # conservative: attribute the whole warm run to 2 prompts
est_fit_s = per_prompt * 100            # one 100-prompt fit
est_total_h = 3 * est_fit_s / 3600.0
d = {"probe_cold_s": probe, "probe_warm_s": warm, "per_prompt_s_upper": per_prompt,
     "est_one_fit_h": est_fit_s / 3600.0, "est_three_fits_h": est_total_h, "max_fit_hours": maxh,
     "go": est_total_h <= maxh}
json.dump(d, open('/workspace/out/llama8b_timing.json', 'w'), indent=1)
print("ITEM2_TIMING", json.dumps(d))
PY
GO=$(python -c "import json; print('yes' if json.load(open('/workspace/out/llama8b_timing.json'))['go'] else 'no')")

if [ "$GO" = "yes" ]; then
  step "ITEM 2b: llama3.1-8b corpus fits (wiki_a seed 0, wiki_b seed 1, code seed 0), 100 prompts"
  FIT="python -u fit_lens.py --model meta-llama/Llama-3.1-8B --n-prompts 100 --dim-batch 8 --max-seq-len 128 --match-length --device cuda"
  $FIT --seed 0 --corpus wikitext --out /workspace/lenses/llama8b_wiki_a.pt 2>&1 | grep -vE "Loading|it/s|Token indices|Repo card" | tail -2 || echo FIT_WIKI_A_FAILED
  echo "wiki_a done $(date -u +%H:%M:%S)"
  $FIT --seed 1 --corpus wikitext --out /workspace/lenses/llama8b_wiki_b.pt 2>&1 | grep -vE "Loading|it/s|Token indices|Repo card" | tail -2 || echo FIT_WIKI_B_FAILED
  echo "wiki_b done $(date -u +%H:%M:%S)"
  $FIT --seed 0 --corpus code --out /workspace/lenses/llama8b_code.pt 2>&1 | grep -vE "Loading|it/s|Token indices|Repo card" | tail -2 || echo FIT_CODE_FAILED
  echo "code done $(date -u +%H:%M:%S)"
else
  echo "ITEM2_SKIPPED: extrapolated fit time exceeds MAX_FIT_HOURS=$MAX_FIT_HOURS (see llama8b_timing.json)"
fi

step "receipts"
python -u -c "
import json,glob,os,hashlib,transformers,torch
r={}
for f in sorted(glob.glob('/workspace/lenses/*.pt'))+sorted(glob.glob('/workspace/out/*.json')):
    r[os.path.basename(f)]={'bytes':os.path.getsize(f),'sha256':hashlib.sha256(open(f,'rb').read()).hexdigest()[:16]}
r['versions']={'transformers':transformers.__version__,'torch':torch.__version__}
json.dump(r,open('/workspace/out/pod_receipt.json','w'),indent=1); print(json.dumps(r,indent=1))
"
ls -la /workspace/lenses /workspace/out
echo POD_CORPUS8B_DONE
