"""Correctness gate for extend_lens.py (CPU, gpt2, free — validate-cheap-first).

Reference: jlens.fit gpt2 on prompts[:6] in one uninterrupted run.
Staged:    fit prompts[:3] -> save fp16 (exactly like the published artifact) ->
           extend_lens.py to target 6, KILLED after its first completed prompt ->
           relaunched (resume from synced ckpt) -> finish.
Compare:   per-layer max relative Frobenius deviation staged-vs-reference. Expected:
           tiny (fp16 bootstrap rounding at weight 3/6), NOT zero; and the deviation
           bound documents the same effect for the 397B (weight 24/target).
"""
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import jlens  # noqa: E402
import transformers  # noqa: E402
from fit_lens import load_corpus  # noqa: E402

W = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/extend_gate")
W.mkdir(parents=True, exist_ok=True)

hf = transformers.AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2", torch_dtype=torch.float32).eval()
tok = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2")
model = jlens.from_hf(hf, tok, compile=False)
prompts = load_corpus(6, 0, corpus="wikitext")

print("== reference: uninterrupted n=6 fit")
ref = jlens.fit(model, prompts, dim_batch=8, max_seq_len=128)

print("== staged: n=3 fit -> fp16 save")
base = jlens.fit(model, prompts[:3], dim_batch=8, max_seq_len=128)
base.save(str(W / "base_n3.pt"))  # jlens saves fp16 — same as the published artifact

cmd = [sys.executable, str(HERE / "extend_lens.py"), "--model",
       "openai-community/gpt2", "--lens-file", str(W / "base_n3.pt"),
       "--target-n", "6", "--dim-batch", "8", "--device", "cpu",
       "--work", str(W / "work"), "--sync-dir", str(W / "sync"),
       "--snapshot-every", "1"]
print("== extension pass 1 (killed after first completed prompt)")
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print("   ", line.rstrip())
    if "mean_rel_change=" in line:  # first completed extension prompt
        p.kill()
        print("    KILLED mid-run (simulated spot pull / credit exhaustion)")
        break
p.wait()

print("== extension pass 2 (resume) ")
out = subprocess.run(cmd, capture_output=True, text=True)
print(out.stdout[-1200:])
assert "DONE at n=6" in out.stdout, "resume did not complete"

ext = jlens.JacobianLens.load(str(W / "sync" / "lens_n6.pt"))
rel = {l: float((ext.jacobians[l].float() - ref.jacobians[l].float()).norm()
                / ref.jacobians[l].float().norm()) for l in ref.jacobians}
worst = max(rel.values())
csv_txt = (W / "sync" / "convergence.csv").read_text()
print(f"max per-layer rel deviation staged-vs-reference: {worst:.2e}")
print("convergence.csv rows:", len(csv_txt.strip().splitlines()) - 1)
verdict = "PASS" if worst < 5e-3 else "FAIL"
print(f"EXTEND_GATE_{verdict}")
json.dump({"max_rel_dev": worst, "per_layer": rel,
           "csv": csv_txt}, open(W / "gate_result.json", "w"), indent=1)
