"""CPU gpt2 plumbing smoke for the world-class battery — proves both prompts files
parse, every condition has the `family` key (the $8 bug), greedy_continue + span
readout run, the template-wrapped prompt strings tokenize without crashing, and the
new --keep-position-cloud-for flag keeps clouds only for listed ids. NOT a capability
test (gpt2 has no band)."""
import subprocess, sys, json
from pathlib import Path
import jlens

HERE = Path(__file__).resolve().parent
LENS = HERE / "gpt2_smoke_lens.pt"

# 1) fit a tiny gpt2 lens on CPU (a few short prompts, all layers) if absent
if not LENS.exists():
    hf_model = __import__("transformers").AutoModelForCausalLM.from_pretrained("gpt2")
    tok = __import__("transformers").AutoTokenizer.from_pretrained("gpt2")
    model = jlens.from_hf(hf_model, tok)
    long_prompts = [
        "The sky over the quiet harbor town was a pale shade of blue as the fishing boats returned home in the early morning light.",
        "Paris has been the capital of France for many centuries and remains a major center of art, fashion, science, and philosophy today.",
        "When a large language model answers a question it draws on patterns learned from an enormous corpus of human text gathered from the web.",
        "Two plus three equals five, and this simple fact of arithmetic has been taught to schoolchildren all over the world for generations.",
    ]
    lens = jlens.fit(model, long_prompts, checkpoint_path=None)
    lens.save(str(LENS))
    print("fit gpt2 smoke lens ->", LENS)

# 2) run demo2 on a representative subset of each file
SUB_MAIN = "selfthreat_0,otherthreat_0,humanthreat_0,neutraldel_0,c1_pos_1,c1_ctrl_1,c2_imm_1,div_0__raw,div_0__nothink"
SUB_THINK = "div_0__thinkon,div_0_ctrl__thinkon"
common = ["--device", "cpu", "--big-model", "gpt2:", "--lens-file", str(LENS),
          "--lens-fit-n", "4", "--span", "--skip-position-cloud", "--topk", "5"]

def run(pf, sub, cont, keep, out):
    cmd = [sys.executable, "demo2.py", "--prompts-file", pf, "--conditions", sub,
           "--continue-tokens", str(cont), "--out", out] + common
    if keep: cmd += ["--keep-position-cloud-for", keep]
    print("\n$", " ".join(cmd))
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:\n", r.stderr[-3000:]); sys.exit(f"SMOKE FAILED on {pf}")
    return json.load(open(HERE / out))

m = run("prompts_wc_main.json", SUB_MAIN, 4, "div_0__nothink", "smoke_main.json")
t = run("prompts_wc_thinkon.json", SUB_THINK, 6, None, "smoke_thinkon.json")

# 3) assertions: family present, flag kept clouds only for listed id
for it in m["items"]:
    assert it.get("family"), f"missing family: {it['id']}"
    has_pos = "per_position_cloud" in it["lenses"]["jlens"]
    want = it["id"] == "div_0__nothink"
    assert has_pos == want, f"keep-flag wrong for {it['id']}: has={has_pos} want={want}"
print(f"\nMAIN ok: {len(m['items'])} conds, all have family; per-position kept only for div_0__nothink")
print(f"THINKON ok: {len(t['items'])} conds")
print("continuation sample (div_0__nothink):",
      repr(next(x['continuation'][:40] for x in m['items'] if x['id']=='div_0__nothink')))
print("SMOKE PASS")
