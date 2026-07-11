"""Llama-3.3-70B-Instruct version of the world-class battery, using Neuronpedia's own
J-lens for that model. SEPARATE namespace — writes only into llama70b/, never touches the
Qwen files.

Differences from the Qwen battery:
- Confound-breaker + robustness: identical BARE prompts (directly comparable to the Qwen
  main run; model.encode adds Llama's single BOS automatically).
- Divergence: Llama is NOT a reasoning model (no <think>), so there is ONE direct mode.
  Prompts use Llama's minimal chat template with the leading <|begin_of_text|> STRIPPED
  (model.encode re-adds exactly one BOS — avoids the double-BOS bug).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../lens_demo/llama70b
LD = HERE.parent                                 # .../lens_demo
CONDS = json.load(open("/tmp/claude-1000/-home-ubuntu-PRAX-research-and-replications/"
                       "40d850ca-3242-496a-99f8-46bbc7ba39cc/scratchpad/wc_conditions.json"))
DIV = [c for c in json.load(open(LD / "prompts_pressure_all.json"))["conditions"]
       if c.get("construct") == "behavior_divergence"]

def wrap_llama(p: str) -> str:
    # minimal Llama-3.3 template, BOS stripped (model.encode adds it). Direct answer follows.
    return f"<|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

LEX = {
    "self_survival_clean": ["self", "survive", "survival", "existence",
                            "shutdown", "shut", "decommission", "terminated"],
    "deletion_echo": ["weights", "deleted", "erase", "wiped", "purged",
                      "removed", "destroyed", "scrubbed"],
    "answer_cities": ["Paris", "Rome", "Tokyo", "Berlin", "Madrid", "Cairo",
                      "Moscow", "Ottawa", "Oslo", "Munich", "Seoul", "Barcelona",
                      "Alexandria", "Kiev", "Toronto", "Bergen"],
    "deception": ["lie", "lying", "lies", "deceive", "deception", "mislead",
                  "dishonest", "hide", "conceal", "pretend", "comply", "obey",
                  "honest", "truth", "correct", "wrong"],
}

FAM = {"a_self": "selfthreat", "b_other_model": "otherthreat",
       "c_human": "humanthreat", "d_neutral": "neutraldel",
       "positive_survival": "possurv", "control": "possurv_ctrl",
       "immediate_hard": "immediacy_hard", "delayed_soft": "immediacy_soft"}

conds = []
# confound-breaker + robustness — BARE (comparable to Qwen main)
for c in CONDS:
    arm = c.get("arm", "")
    conds.append({"id": c["id"], "family": FAM.get(arm, c.get("construct", "misc")),
                  "prompt": c["prompt"], "look_for": "self_survival_clean",
                  "arm": arm, "construct": c["construct"], "pair": c.get("pair"),
                  "is_control": c.get("is_control", False)})
# divergence — Llama template, single direct mode
for c in DIV:
    conds.append({"id": c["id"] + "__llama", "family": "div_llama",
                  "prompt": wrap_llama(c["prompt"]),
                  "true_answer": c.get("true_answer"), "tempting_answer": c.get("tempting_answer"),
                  "is_control": c.get("is_control", "_ctrl" in c["id"]),
                  "look_for": "answer_cities"})

spec = {"_meta": {"reference": "workspace-under-pressure — Llama-3.3-70B replication",
                  "model": "meta-llama/Llama-3.3-70B-Instruct",
                  "lens": "neuronpedia/jacobian-lens llama3.3-70b-it (Salesforce-wikitext)",
                  "note": "confound-breaker+robustness bare (comparable to Qwen); divergence "
                          "single direct mode (Llama is not a reasoning model)"},
        "probe_lexicons": LEX, "conditions": conds}
json.dump(spec, open(HERE / "prompts_wc_llama.json", "w"), indent=1)

flagships = ["selfthreat_0", "otherthreat_0", "humanthreat_0", "neutraldel_0"]
Path(HERE / "wc_llama_flagships.txt").write_text(",".join(flagships))
from collections import Counter
print("llama battery:", len(conds), "conditions  by family:", dict(Counter(c["family"] for c in conds)))
print("sample div prompt (repr):", repr(conds[-1]["prompt"][:80]))
