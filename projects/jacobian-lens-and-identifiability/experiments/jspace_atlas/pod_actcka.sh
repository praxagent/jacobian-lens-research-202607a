#!/bin/bash
# Activation-CKA: gemma-2-9b/2b (flat lens) vs qwen3-4b (structured control). Inference
# only, no jlens/fp8 -> simple stack on the base torch (no upgrade needed).
set -u
cd /workspace
[ -d repo ] || git clone --depth 30 https://github.com/praxagent/jacobian-lens-research-202607a repo
cd repo && git fetch -q && git checkout -q ARMSHA
pip -q install "transformers>=4.44,<5" datasets accelerate safetensors 2>&1 | tail -1
python -c "import torch,transformers; from transformers import AutoModelForCausalLM; print('deps OK torch',torch.__version__,'tf',transformers.__version__)" || { echo DEPS_FAILED; exit 1; }
cd projects/jacobian-lens-and-identifiability/experiments/jspace_atlas
mkdir -p /workspace/actcka
A="python -u activation_cka.py --n-prompts 80 --max-len 128 --n-tok 4096"
$A --model google/gemma-2-9b --tag gemma2_9b_flat  --out /workspace/actcka/gemma2_9b 2>&1 | grep -vE "Loading|it/s|Repo card|deprecat|checkpoint shards" || echo GEMMA9B_FAILED
$A --model google/gemma-2-2b --tag gemma2_2b_flat  --out /workspace/actcka/gemma2_2b 2>&1 | grep -vE "Loading|it/s|Repo card|deprecat|checkpoint shards" || echo GEMMA2B_FAILED
$A --model Qwen/Qwen3-4B     --tag qwen3_4b_struct --out /workspace/actcka/qwen3_4b 2>&1 | grep -vE "Loading|it/s|Repo card|deprecat|checkpoint shards" || echo QWEN_FAILED
echo ACTCKA_DONE
