#!/bin/bash
# PREREG_C_8B.md (2026-09-08): Test C damage matrices for llama3.1-8b, prose and code, float32, one pod.
#   HF_TOKEN=... bash pod_testC_8b.sh <git-ref>
set -u
REF="${1:-main}"
mkdir -p /workspace/out
fail () { echo "$1 ($(date -u +%H:%M:%S))"; touch /workspace/out/FAILED; exit 1; }
step () { echo; echo "===== $1  ($(date -u +%H:%M:%S))  ====="; }
cd /workspace
[ -d repo ] || git clone --depth 60 https://github.com/praxagent/jacobian-lens-research-202607a repo || fail CLONE_FAILED
cd repo && git fetch -q && git checkout -q "$REF" || fail CHECKOUT_FAILED
echo "repo at $(git rev-parse --short HEAD)"
pip -q install --upgrade "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
pip -q install "transformers>=5,<6" accelerate safetensors 2>&1 | tail -1
python -c "import torch,transformers; print('deps OK torch',torch.__version__,'tf',transformers.__version__)" || fail DEPS_INSTALL_FAILED
cd /workspace/repo/projects/jacobian-lens-and-identifiability/experiments/bands_vs_computation

step "GATE: gpt2 prose damage, 4 prompts equivalent (mechanics on this pod)"
python -u swap_damage.py --model gpt2 --slug gpt2-small --corpus prose --device cuda --dtype float32 --out /workspace/out/gate_gpt2_prose.json 2>&1 | grep -E 'prompts, L=|SWAP_DONE|Error|Traceback' | tail -3
python -c "import json; d=json.load(open('/workspace/out/gate_gpt2_prose.json')); assert d['diag_max_abs']==0.0; print('GATE_OK median_D_far', round(d['median_D_far'],3))" || fail GATE_FAILED

for C in prose code; do
  step "llama3.1-8b damage on $C (float32)"
  T0=$(date +%s)
  python -u swap_damage.py --model meta-llama/Llama-3.1-8B --slug llama3.1-8b --corpus $C --device cuda --dtype float32 --out /workspace/out/llama3.1-8b_${C}.json 2>&1 | grep -E 'prompts, L=|source layer (8|16|24|32)/|SWAP_DONE|Error|Traceback|out of memory' | tail -8
  T1=$(date +%s)
  python - "$C" "$T0" "$T1" <<'PY' || fail "RECEIPT_${C}_FAILED"
import json, sys, hashlib
c, t0, t1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
p = f"/workspace/out/llama3.1-8b_{c}.json"; d = json.load(open(p))
import numpy as np
D = np.array(d["D"]); assert D.shape[0] == D.shape[1] >= 30 and np.isfinite(D).all() and d["diag_max_abs"] == 0.0
r = {"prereg": "PREREG_C_8B.md", "corpus": c, "file": p.split("/")[-1], "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
     "n_layers": int(D.shape[0]), "n_prompts": d["n_prompts"], "median_D_far": d["median_D_far"], "wall_s": t1 - t0, "all_finite": True}
json.dump(r, open(f"/workspace/out/receipt_{c}.json", "w"), indent=1); print("RECEIPT", json.dumps(r))
PY
done
ls -la /workspace/out
touch /workspace/out/DONE
echo "POD_TESTC_8B_DONE ($(date -u +%H:%M:%S))"
