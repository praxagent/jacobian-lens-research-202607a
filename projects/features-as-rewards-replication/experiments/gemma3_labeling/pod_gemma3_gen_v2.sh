#!/bin/bash
# Amendment 7: generate 300 NEW Gemma-3-12B completions (LongFact prompts 300-599).
# Same pinned generation as arm-4 (greedy, seed 0, max_new_tokens 768, bf16).
set -e
cd /workspace
[ -d repo ] || git clone --depth 50 https://github.com/praxagent/jacobian-lens-research-202607a repo
cd repo && git fetch -q && git checkout -q ARMSHA
pip -q install datasets "transformers>=4.50,<5" safetensors accelerate 2>&1 | tail -1
cd projects/features-as-rewards-replication/experiments/gemma3_agreement
python -u run_agreement.py --gen-only --n 300 --prompt-offset 300 --device cuda \
  --cache-file /workspace/completions_cache_v2.jsonl \
  --out /workspace/unused_a7.json
sha256sum /workspace/completions_cache_v2.jsonl > /workspace/cache_v2.sha
echo GEN_V2_DONE
