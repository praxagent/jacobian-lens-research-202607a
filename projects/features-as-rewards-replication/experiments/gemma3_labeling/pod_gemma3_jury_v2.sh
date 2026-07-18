#!/bin/bash
# Amendment 7: powered gate-(b) re-test. Train/val = original folds; test = ALL new
# definitive spans (prompts 300-599). fp32 cache (amendment 6b).
set -e
cd /workspace
[ -d repo ] || git clone --depth 50 https://github.com/praxagent/jacobian-lens-research-202607a repo
cd repo && git fetch -q && git checkout -q ARMSHA
pip -q install datasets "transformers>=4.50,<5" safetensors accelerate 2>&1 | tail -1
python - <<'PY'
from huggingface_hub import hf_hub_download
import os
p = hf_hub_download("google/gemma-scope-2-12b-it",
                    "resid_post/layer_24_width_16k_l0_medium/params.safetensors",
                    revision="4c419f1ba0be8b7754d4151d4f26c23b92a9029e")
os.system(f"ln -sf {p} /workspace/sae_g3_l24.safetensors")
PY
cd projects/features-as-rewards-replication/experiments/probe_calibration
python -u run.py --model google/gemma-3-12b-it --layers 24 --primary-layer 24 \
  --device cuda --cache-dtype float32 \
  --jury-labels ../gemma3_labeling/receipts/jury_gemma3_labels.json \
  --completions-cache ../gemma3_agreement/receipts/completions_cache.jsonl \
  --jury-labels-test ../gemma3_labeling/receipts/jury_gemma3_labels_v2.json \
  --completions-cache-test ../gemma3_agreement/receipts/completions_cache_v2.jsonl \
  --sae-path /workspace/sae_g3_l24.safetensors --sae-format gemmascope \
  --out /workspace/receipt_gemma3_jury_a7.json
sha256sum /workspace/receipt_gemma3_jury_a7.json > /workspace/receipt_a7.sha
echo GEMMA3_A7_DONE
