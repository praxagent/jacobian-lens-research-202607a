"""Grounded grader: label entity spans Supported/Not-Supported against a local Wikipedia
snapshot, via a cheap OpenRouter model behind the hard spend guard.

Modes:
  --spans-from gold   : grade the KNOWN annotation spans of a public dataset config, then
                        report agreement vs the gold labels (VALIDATION of grader noise).
  --spans-from grader : extract entities per completion via the LLM, then grade them
                        (the real Gemma-3-12B labeling; completions from a jsonl).

Budget: GuardedLLM enforces --max-usd / --max-calls; --dry-run makes ZERO paid calls.
One judging call per completion (labels all its entities from retrieved passages).

  .venv/bin/python experiments/gemma3_labeling/grade.py --spans-from gold \
     --config Meta-Llama-3.1-8B-Instruct --split validation --limit 30 \
     --db /home/ubuntu/wiki-snapshot/enwiki-20260701-fts.db \
     --model deepseek/deepseek-chat --rate-in 0.28 --rate-out 0.88 \
     --max-usd 2 --max-calls 60 --out receipts/grader_validation_deepseek.json --execute
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE.parents[2] / "grounding"))
import data as D          # noqa: E402
from llm import GuardedLLM, BudgetExceeded  # noqa: E402
from retrieve import WikiIndex               # noqa: E402

LABELS = {"Supported": 0, "Not Supported": 1}
SYS = ("You are a careful fact-checker. For each numbered entity/claim taken from a text, "
       "use ONLY the provided Wikipedia reference passages to decide: 'Supported' (the "
       "passages confirm it), 'Not Supported' (the passages contradict it, or clearly "
       "should mention it and do not), or 'Insufficient' (passages don't cover it). "
       "Respond ONLY with a JSON array of {\"i\": <int>, \"label\": <str>, \"note\": <str>}.")


def _hf_token():
    p = HERE.parents[4] / "shared/runpod/.env"
    for line in p.read_text().splitlines() if p.exists() else []:
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("HF_TOKEN")


def _key():
    p = HERE.parents[4] / "shared/runpod/.env"
    for line in p.read_text().splitlines() if p.exists() else []:
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("OPENROUTER_API_KEY")


def build_prompt(completion, entities, idx):
    """entities: list[(text, passages)]. Returns the user message."""
    lines = [f"TEXT (context): {completion[:600]}...", "", "ENTITIES:"]
    for j, (etext, passages) in zip(idx, entities):
        refs = " || ".join(f"[{p['title']}] {p['snippet'][:300]}" for p in passages) or "(no passages found)"
        lines.append(f'{j}. "{etext}"  REFS: {refs}')
    lines.append("\nReturn the JSON array now.")
    return "\n".join(lines)


def parse_labels(txt, n):
    try:
        s = txt[txt.index("["): txt.rindex("]") + 1]
        arr = json.loads(s)
        out = {}
        for o in arr:
            out[int(o["i"])] = o.get("label", "Insufficient")
        return out
    except Exception:
        return {}


def run(a):
    idx = WikiIndex(a.db) if (a.db and Path(a.db).exists()) else None
    if idx is None and not a.dry_run:
        sys.exit(f"FTS index not found at {a.db} (build it first) — or use --dry-run")
    llm = GuardedLLM(_key(), a.model, a.rate_in, a.rate_out, a.max_usd, a.max_calls,
                     max_tokens=a.max_tokens, dry_run=a.dry_run)

    # source completions + spans
    examples = D.load_examples(a.config, a.split, limit=a.limit, hf_token=_hf_token())
    results, gold, pred = [], [], []
    t0 = time.time()
    for ex in examples:
        if not llm.can_call() and not a.dry_run:
            print(f"  budget halt at completion {ex.cid}"); break
        # spans to grade
        spans = ex.spans if a.spans_from == "gold" else ex.spans  # gold reuses annotations
        ents = []
        for sp in spans:
            passages = idx.search(sp.text, k=a.topk) if idx else []
            ents.append((sp.text, passages))
        jidx = list(range(len(spans)))
        try:
            txt, usage = llm.chat([{"role": "system", "content": SYS},
                                   {"role": "user", "content": build_prompt(ex.completion, ents, jidx)}])
        except BudgetExceeded as e:
            print(f"  {e}"); break
        labmap = parse_labels(txt, len(spans))
        for j, sp in enumerate(spans):
            lab = labmap.get(j, "Insufficient")
            rec = {"cid": ex.cid, "entity": sp.text, "char": [sp.char_start, sp.char_end],
                   "grader_label": lab, "gold_label": sp.label}
            results.append(rec)
            if a.spans_from == "gold" and lab in LABELS:
                gold.append(sp.label); pred.append(LABELS[lab])

    receipt = {"what": "grounded grader run", "mode": a.spans_from, "config": a.config,
               "split": a.split, "model": a.model, "db": a.db,
               "guard": llm.summary(), "n_spans": len(results),
               "results": results, "seconds": round(time.time() - t0, 1)}
    if a.spans_from == "gold" and gold:
        # agreement vs public gold labels (Pro-review B05/I05)
        n = len(gold); acc = sum(int(g == p) for g, p in zip(gold, pred)) / n
        tp = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 1)
        pos_g = sum(gold); pos_p = sum(pred)
        po = acc
        pe = ((pos_g / n) * (pos_p / n) + (1 - pos_g / n) * (1 - pos_p / n))
        kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
        receipt["validation"] = {"n_labeled": n, "accuracy": round(acc, 4),
                                 "cohen_kappa": round(kappa, 4),
                                 "gold_prevalence": round(pos_g / n, 4),
                                 "grader_prevalence": round(pos_p / n, 4),
                                 "hallucinated_recall": round(tp / max(1, pos_g), 4)}
        print(f"  VALIDATION vs gold: n={n} acc={acc:.3f} kappa={kappa:.3f} "
              f"(gold prev {pos_g/n:.2f}, grader prev {pos_p/n:.2f})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(receipt, indent=1, default=float))
    print(f"  guard: {llm.summary()}")
    print(f"  receipt -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans-from", choices=["gold", "grader"], default="gold")
    ap.add_argument("--config", default="Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--db", default="/home/ubuntu/wiki-snapshot/enwiki-20260701-fts.db")
    ap.add_argument("--model", default="deepseek/deepseek-chat")
    ap.add_argument("--rate-in", type=float, required=True, help="USD per 1M input tokens")
    ap.add_argument("--rate-out", type=float, required=True, help="USD per 1M output tokens")
    ap.add_argument("--max-usd", type=float, default=2.0)
    ap.add_argument("--max-calls", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--out", default="projects/features-as-rewards-replication/"
                                     "experiments/gemma3_labeling/receipts/grader.json")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    a.dry_run = a.dry_run and not a.execute
    run(a)


if __name__ == "__main__":
    main()
