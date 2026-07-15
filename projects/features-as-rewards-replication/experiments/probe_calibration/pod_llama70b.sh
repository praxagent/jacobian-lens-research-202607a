#!/bin/bash
# Llama-3.3-70B arm (2x80GB GPUs, device_map=auto). SHA substituted at deploy.
set -e
cd /workspace
[ -d repo ] || git clone --depth 50 https://github.com/praxagent/jacobian-lens-research-202607a repo
cd repo && git fetch -q && git checkout -q ARMSHA
pip -q install datasets transformers safetensors 2>&1 | tail -1
python - <<'PY'
from huggingface_hub import hf_hub_download
import os
p = hf_hub_download("Goodfire/Llama-3.3-70B-Instruct-SAE-l50",
                    "Llama-3.3-70B-Instruct-SAE-l50.pt")
os.system(f"ln -sf {p} /workspace/sae_l50.pth")
PY
cd projects/features-as-rewards-replication/experiments/probe_calibration
python -u run.py --model meta-llama/Llama-3.3-70B-Instruct \
  --config Llama-3.3-70B-Instruct \
  --layers 40 50 60 --primary-layer 50 --device auto \
  --sae-path /workspace/sae_l50.pth --sae-format goodfire \
  --exploratory-capacity \
  --out /workspace/receipt_llama33_70b.json
sha256sum /workspace/receipt_llama33_70b.json > /workspace/receipt70b.sha
echo LLAMA70B_ARM_DONE
