#!/bin/bash
# Llama-3.1-8B arm on a RunPod GPU (frozen @ 6830d9c). Idempotent; nohup-safe.
set -e
cd /workspace
[ -d repo ] || git clone --depth 50 https://github.com/praxagent/jacobian-lens-research-202607a repo
cd repo && git fetch -q && git checkout -q 6830d9cdd28b4dc489cb33ea1c04af4b2f468830
pip -q install datasets transformers safetensors 2>&1 | tail -1
# SAE (Goodfire l19, pinned revision)
python - <<'PY'
from huggingface_hub import hf_hub_download
import os
p = hf_hub_download("Goodfire/Llama-3.1-8B-Instruct-SAE-l19",
                    "Llama-3.1-8B-Instruct-SAE-l19.pth",
                    revision="f6775a221e47b44233af4bac2c7b65189265519a")
os.system(f"ln -sf {p} /workspace/sae_l19.pth")
PY
cd projects/features-as-rewards-replication/experiments/probe_calibration
python -u run.py --model meta-llama/Llama-3.1-8B-Instruct \
  --config Meta-Llama-3.1-8B-Instruct \
  --layers 12 16 19 24 --primary-layer 19 --device cuda \
  --sae-path /workspace/sae_l19.pth --sae-format goodfire \
  --out /workspace/receipt_llama31_8b.json
sha256sum /workspace/receipt_llama31_8b.json > /workspace/receipt.sha
echo LLAMA8B_ARM_DONE
