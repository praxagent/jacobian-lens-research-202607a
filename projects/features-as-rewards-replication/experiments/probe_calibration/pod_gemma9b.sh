#!/bin/bash
# gemma-2-9b arm on a RunPod GPU. Idempotent; nohup-safe. SHA substituted at deploy.
set -e
cd /workspace
[ -d repo ] || git clone --depth 50 https://github.com/praxagent/jacobian-lens-research-202607a repo
cd repo && git fetch -q && git checkout -q ARMSHA
pip -q install datasets transformers safetensors 2>&1 | tail -1
python - <<'PY'
from huggingface_hub import hf_hub_download
import os
p = hf_hub_download("google/gemma-scope-9b-pt-res", "layer_20/width_16k/average_l0_68/params.npz",
                    revision="f9b689815814972562d28082f9f7d65d7e01fdc8")
os.system(f"ln -sf {p} /workspace/sae_gemma9b_l20.npz")
PY
cd projects/features-as-rewards-replication/experiments/probe_calibration
python -u run.py --model google/gemma-2-9b-it --config gemma-2-9b-it \
  --layers 12 20 28 --primary-layer 20 --device cuda \
  --sae-path /workspace/sae_gemma9b_l20.npz --sae-format gemmascope \
  --out /workspace/receipt_gemma2_9b.json
sha256sum /workspace/receipt_gemma2_9b.json > /workspace/receipt9b.sha
echo GEMMA9B_ARM_DONE
