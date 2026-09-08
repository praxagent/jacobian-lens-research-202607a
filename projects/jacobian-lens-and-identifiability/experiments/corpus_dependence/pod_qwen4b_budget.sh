#!/bin/bash
# PREREG_QWEN4B_BUDGET.md (2026-09-08): six qwen3-4b fits on one pod, WikiText-103, the same recipe
# as the corpus and fit-budget experiments: wiki_a (100, seed 0) reference, wiki_b (100, seed 1)
# seed null, n8/n16/n24/n48 (seed 0; nested prefixes of wiki_a). Qwen/Qwen3-4B is ungated.
# The pod never holds an API key; a scheduled check from the work box fetches on DONE and terminates.
#   bash pod_qwen4b_budget.sh <git-ref>
set -u
REF="${1:-main}"
mkdir -p /workspace/out /workspace/lenses
fail () { echo "$1 ($(date -u +%H:%M:%S))"; touch /workspace/out/FAILED; exit 1; }
step () { echo; echo "===== $1  ($(date -u +%H:%M:%S))  ====="; }

cd /workspace
[ -d repo ] || git clone --depth 60 https://github.com/praxagent/jacobian-lens-research-202607a repo || fail CLONE_FAILED
cd repo && git fetch -q && git checkout -q "$REF" || fail CHECKOUT_FAILED
echo "repo at $(git rev-parse --short HEAD)"
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

FIT="python -u fit_lens.py --model Qwen/Qwen3-4B --dim-batch 8 --max-seq-len 128 --match-length --device cuda --corpus wikitext"
: > /workspace/out/timing_q4b.txt
for job in "wiki_a 100 0" "wiki_b 100 1" "n8 8 0" "n16 16 0" "n24 24 0" "n48 48 0"; do
  set -- $job; NAME=$1; N=$2; SEED=$3
  step "FIT q4b_$NAME: $N prompts, seed $SEED"
  T0=$(date +%s)
  $FIT --n-prompts $N --seed $SEED --out /workspace/lenses/q4b_${NAME}.pt 2>&1 \
    | grep -vE "Loading|it/s|Token indices|Repo card" | tail -3
  T1=$(date +%s)
  [ -f /workspace/lenses/q4b_${NAME}.pt ] || fail "FIT_${NAME}_FAILED"
  echo "$NAME $N $SEED $((T1-T0))" | tee -a /workspace/out/timing_q4b.txt
done

step "RECEIPT (lesson 12: validate the outputs before anyone terminates anything)"
python -u - <<'PY' || fail RECEIPT_FAILED
import json, hashlib, os, glob, torch, transformers
timing = {l.split()[0]: {"n_prompts": int(l.split()[1]), "seed": int(l.split()[2]), "fit_wall_s": int(l.split()[3])}
          for l in open("/workspace/out/timing_q4b.txt").read().strip().splitlines()}
files = []
for p in sorted(glob.glob("/workspace/lenses/q4b_*.pt")):
    d = torch.load(p, map_location="cpu", weights_only=False)
    J = d["J"]; layers = sorted(J.keys())
    finite = all(bool(torch.isfinite(J[l]).all()) for l in layers)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    name = os.path.basename(p)[len("q4b_"):-3]
    files.append({"file": os.path.basename(p), "name": name, **timing.get(name, {}),
                  "bytes": os.path.getsize(p), "sha256": h.hexdigest(), "n_layers": len(layers),
                  "d_model": d["d_model"], "all_finite": finite})
    assert finite and len(layers) >= 30, f"{p} not usable"
r = {"prereg": "PREREG_QWEN4B_BUDGET.md", "model": "Qwen/Qwen3-4B", "files": files,
     "versions": {"transformers": transformers.__version__, "torch": torch.__version__}}
json.dump(r, open("/workspace/out/receipt_q4b.json", "w"), indent=1)
print("RECEIPT", json.dumps(r))
assert len(files) == 6, f"expected 6 lenses, found {len(files)}"
PY
ls -la /workspace/lenses /workspace/out
touch /workspace/out/DONE
echo "POD_QWEN4B_BUDGET_DONE ($(date -u +%H:%M:%S))"
