"""Amendment 7: local entity extraction for the NEW completion set (prompts 300-599).

Same DeepSeek extraction prompt as arm-4's run_agreement.extract_entities, run from
this box (API-only, no GPU). Output: spans json {cid: [entity, ...]} consumed by
jury_gemma3.py --spans.

  .venv/bin/python extract_entities_v2.py --execute
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "grounding"))
from llm import GuardedLLM  # noqa: E402
from jury import _env       # noqa: E402

CACHE = HERE.parents[1] / "gemma3_agreement/receipts/completions_cache_v2.jsonl"
OUT = HERE.parent / "receipts/entities_v2.json"
SYS = ("List the specific factual entities (names, dates, places, organizations, "
       "numbers, titles) in the text as VERBATIM substrings. Respond ONLY with a "
       "JSON array of strings, max 25, each copied exactly from the text.")


def main(a):
    rows = [json.loads(l) for l in CACHE.read_text().splitlines()]
    llm = GuardedLLM(_env("OPENROUTER_API_KEY"), "deepseek/deepseek-chat", 0.28, 0.88,
                     max_usd=2.0, max_calls=len(rows) + 10, dry_run=a.dry_run)
    out = {}
    for k, r in enumerate(rows):
        if a.dry_run:
            continue
        txt, _ = llm.chat([{"role": "system", "content": SYS},
                           {"role": "user", "content": r["c"][:4000]}], max_tokens=600)
        try:
            arr = json.loads(txt[txt.index("["): txt.rindex("]") + 1])
            ents = [s for s in arr if isinstance(s, str) and 2 < len(s) < 80
                    and s in r["c"]]
        except Exception:
            ents = []
        out[f"g3:{r['i']}"] = list(dict.fromkeys(ents))
        if k % 25 == 0:
            print(f"  {k}/{len(rows)} guard={llm.summary()}", flush=True)
    OUT.write_text(json.dumps({"what": "amendment-7 entity extraction (new prompts "
                                       "300-599), same prompt as arm-4",
                               "extractor": "deepseek/deepseek-chat",
                               "guard": llm.summary(), "n_completions": len(rows),
                               "n_entities": sum(len(v) for v in out.values()),
                               "spans": out}, indent=1))
    print(f"  EXTRACT_V2_DONE {sum(len(v) for v in out.values())} entities -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    a.dry_run = a.dry_run and not a.execute
    main(a)
