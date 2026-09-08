#!/bin/bash
# PREREG_8B_v2.md (2026-09-08): ONE 400-prompt llama3.1-8b fit per pod, three pods in parallel (one
# arm each), the same recipe as pod_corpus8b.sh's 100-prompt fits. No timing gate this time: the
# throughput is measured (about 117 s/prompt on an RTX A6000, so about 13 h per arm).
# The pod never holds an API key; a scheduled check from the work box fetches on DONE and terminates.
# Usage on the pod (HF_TOKEN passed inline by the launching ssh command, never stored):
#   HF_TOKEN=... bash pod_llama8b_n400.sh <git-ref> <wiki_a|wiki_b|code>
set -u
REF="${1:-main}"; ARM="${2:?arm required: wiki_a | wiki_b | code}"
case "$ARM" in
  wiki_a) CORPUS=wikitext; SEED=0 ;;
  wiki_b) CORPUS=wikitext; SEED=1 ;;
  code)   CORPUS=code;     SEED=0 ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac
mkdir -p /workspace/out /workspace/lenses
fail () { echo "$1 ($(date -u +%H:%M:%S))"; touch /workspace/out/FAILED; exit 1; }
step () { echo; echo "===== $1  ($(date -u +%H:%M:%S))  ====="; }

cd /workspace
[ -d repo ] || git clone --depth 60 https://github.com/praxagent/jacobian-lens-research-202607a repo || fail CLONE_FAILED
cd repo && git fetch -q && git checkout -q "$REF" || fail CHECKOUT_FAILED
echo "repo at $(git rev-parse --short HEAD)"
# torch trio first (transformers 5.x needs torch>=2.5; base image ships 2.4.1), then jlens.
pip -q install --upgrade "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
  --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
pip -q install "git+https://github.com/anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e" \
  "transformers>=5,<6" datasets accelerate safetensors 2>&1 | tail -2
python -c "import torch,transformers,jlens; print('deps OK torch',torch.__version__,'tf',transformers.__version__)" \
  || fail DEPS_INSTALL_FAILED
PROJ=/workspace/repo/projects/jacobian-lens-and-identifiability/experiments

step "GATE: gpt2 fit (correctness gate for the fit path on this pod)"
cd $PROJ/fit_our_own
python -u fit_lens.py --model gpt2 --n-prompts 4 --dim-batch 8 --max-seq-len 128 --seed 0 \
  --corpus wikitext --match-length --device cuda --out /workspace/lenses/gate_gpt2.pt 2>&1 \
  | grep -vE "Loading|it/s|Token indices|Repo card" | tail -3
python -u -c "import torch; d=torch.load('/workspace/lenses/gate_gpt2.pt',map_location='cpu',weights_only=False); print('GATE_OK layers', len(d['J']))" \
  || fail GATE_FAILED
rm -f /workspace/lenses/gate_gpt2.pt /workspace/lenses/gate_gpt2.ckpt

step "FIT: llama3.1-8b $ARM ($CORPUS, seed $SEED), 400 prompts"
T0=$(date +%s)
python -u fit_lens.py --model meta-llama/Llama-3.1-8B --n-prompts 400 --dim-batch 8 --max-seq-len 128 \
  --match-length --device cuda --seed $SEED --corpus $CORPUS \
  --out /workspace/lenses/llama8b400_${ARM}.pt 2>&1 | grep -vE "Loading|it/s|Token indices|Repo card" | tail -3
T1=$(date +%s)
[ -f /workspace/lenses/llama8b400_${ARM}.pt ] || fail FIT_FAILED

step "RECEIPT (lesson 12: validate the output before anyone terminates anything)"
python -u - "$ARM" "$T0" "$T1" <<'PY' || fail RECEIPT_FAILED
import sys, json, hashlib, os, torch, transformers
arm, t0, t1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
p = f"/workspace/lenses/llama8b400_{arm}.pt"
d = torch.load(p, map_location="cpu", weights_only=False)
J = d["J"]; layers = sorted(J.keys())
finite = all(bool(torch.isfinite(J[l]).all()) for l in layers)
h = hashlib.sha256()
with open(p, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
r = {"prereg": "PREREG_8B_v2.md", "arm": arm, "n_prompts": 400, "corpus": os.environ.get("CORPUS_NAME", ""),
     "file": os.path.basename(p), "bytes": os.path.getsize(p), "sha256": h.hexdigest(),
     "n_layers": len(layers), "d_model": d["d_model"], "all_finite": finite,
     "fit_wall_s": t1 - t0, "per_prompt_s": (t1 - t0) / 400.0,
     "versions": {"transformers": transformers.__version__, "torch": torch.__version__}}
json.dump(r, open(f"/workspace/out/receipt_{arm}.json", "w"), indent=1)
print("RECEIPT", json.dumps(r))
assert finite and len(layers) >= 30, "lens not usable"
PY
ls -la /workspace/lenses /workspace/out
touch /workspace/out/DONE
echo "POD_LLAMA8B_N400_${ARM}_DONE ($(date -u +%H:%M:%S))"
