"""Cross-provider LLM-jury labeling on shared, archived Serper evidence.

Three judges from different labs (via one prepaid OpenRouter balance) independently
label each entity span {Supported | Not Supported | Insufficient} given the SAME
archived search evidence. Tiers: unanimous (3/3), majority (2/3). Per-row votes +
evidence links ship in the receipt. Validation mode (--spans-from gold) reports
per-judge, majority, and unanimous kappa vs the public gold labels — the same
pre-registered kappa>=0.5 gate that rejected the Wikipedia grader.

  --dry-run: zero searches, zero judge calls (prompt/cost estimate only).
  Validation (free Serper tier + ~$0.40 judges):
    .venv/bin/python jury.py --config Meta-Llama-3.1-8B-Instruct --split validation \
       --limit 40 --max-queries 800 --out receipts/jury_validation.json --execute
"""
from __future__ import annotations

import argparse, json, os, sys, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE.parents[2] / "grounding"))
import data as D                              # noqa: E402
from llm import GuardedLLM, BudgetExceeded    # noqa: E402
from serper import Serper, SerperExceeded     # noqa: E402

LABELS = {"Supported": 0, "Not Supported": 1}
# Cross-provider jury: different labs -> genuinely different failure modes.
# (rates = USD/M in,out on OpenRouter; verified against /models before execute)
JUDGES = [
    ("deepseek/deepseek-chat", 0.28, 0.88),
    ("z-ai/glm-4.6", 0.45, 1.80),
    ("google/gemini-2.5-flash", 0.30, 2.50),
]
SYS = ("You are a careful fact-checker. For each numbered entity/claim from a text, use "
       "the provided web search results to decide: 'Supported' (results confirm it, "
       "directly or by obvious paraphrase), 'Not Supported' (results contradict it -- "
       "wrong date, wrong person, wrong place, nonexistent thing -- or a claim this "
       "specific would clearly appear in these results and does not), or 'Insufficient' "
       "(results neither confirm nor contradict). "
       'Respond ONLY with a JSON array of {"i": <int>, "label": <str>, "note": <str>}.')


def _env(k):
    p = HERE.parents[4] / "shared/runpod/.env"
    for line in p.read_text().splitlines() if p.exists() else []:
        if line.startswith(k + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get(k)


def build_prompt(completion, spans, evidence):
    lines = [f"TEXT (context): {completion[:600]}...", "", "ENTITIES:"]
    for j, sp in enumerate(spans):
        refs = " || ".join(f"[{e['title']}] {e['snippet'][:280]}"
                           for e in evidence[j]) or "(no results)"
        lines.append(f'{j}. "{sp.text}"  SEARCH RESULTS: {refs}')
    lines.append("\nReturn the JSON array now.")
    return "\n".join(lines)


def parse_labels(txt):
    try:
        arr = json.loads(txt[txt.index("["): txt.rindex("]") + 1])
        return {int(o["i"]): o.get("label", "Insufficient") for o in arr}
    except Exception:
        return {}


def kappa(gold, pred):
    n = len(gold)
    if n == 0:
        return float("nan"), 0
    acc = sum(int(g == p) for g, p in zip(gold, pred)) / n
    pg, pp = sum(gold) / n, sum(pred) / n
    pe = pg * pp + (1 - pg) * (1 - pp)
    return ((acc - pe) / (1 - pe) if pe < 1 else float("nan")), n


def run(a):
    if not a.dry_run:
        models = None
        try:
            import requests
            models = {m["id"] for m in requests.get(
                "https://openrouter.ai/api/v1/models", timeout=20).json()["data"]}
        except Exception:
            pass
        if models:
            for mid, _, _ in JUDGES:
                assert mid in models, f"judge {mid} not on OpenRouter — fix JUDGES"
    ser = Serper(_env("SERPER_DEV_API_KEY"), Path(a.out).parent / "serper_cache.jsonl",
                 max_queries=a.max_queries) if not a.dry_run else None
    juries = {mid: GuardedLLM(_env("OPENROUTER_API_KEY"), mid, ri, ro,
                              max_usd=a.max_usd_per_judge, max_calls=a.limit + 10,
                              max_tokens=1400, dry_run=a.dry_run)
              for mid, ri, ro in JUDGES}
    examples = D.load_examples(a.config, a.split, limit=a.limit, hf_token=_env("HF_TOKEN"))
    rows = []
    t0 = time.time()
    for ex in examples:
        spans = ex.spans
        topic = ex.prompt[:80].replace("\n", " ")
        try:
            evidence = [ser.search(f"{sp.text} {topic}") if ser else [] for sp in spans]
        except SerperExceeded as e:
            print(f"  search cap: {e}"); break
        prompt = build_prompt(ex.completion, spans, evidence)
        votes = {}
        try:
            for mid in juries:
                txt, _ = juries[mid].chat([{"role": "system", "content": SYS},
                                           {"role": "user", "content": prompt}])
                votes[mid] = parse_labels(txt)
        except BudgetExceeded as e:
            print(f"  judge budget: {e}"); break
        for j, sp in enumerate(spans):
            v = {mid: votes.get(mid, {}).get(j, "Insufficient") for mid in juries}
            cnt = Counter(v.values())
            top, ntop = cnt.most_common(1)[0]
            tier = ("unanimous" if ntop == 3 else "majority" if ntop == 2 else "split")
            rows.append({"cid": ex.cid, "entity": sp.text,
                         "char": [sp.char_start, sp.char_end], "gold": sp.label,
                         "votes": v, "jury_label": top if top in LABELS else None,
                         "tier": tier,
                         "evidence_links": [e["link"] for e in evidence[j]]})
        print(f"  [{len(rows)} spans] {ex.cid} search+3 judges ok", flush=True)

    receipt = {"what": "cross-provider LLM-jury labeling on shared archived Serper evidence",
               "judges": [m for m, _, _ in JUDGES], "config": a.config, "split": a.split,
               "n_rows": len(rows), "rows": rows,
               "serper": ser.summary() if ser else "dry-run",
               "judge_guards": {m: juries[m].summary() for m in juries},
               "seconds": round(time.time() - t0, 1), "args": vars(a)}
    if a.spans_from == "gold" and rows and not a.dry_run:
        val = {}
        for mid in juries:  # per-judge kappa
            g, p = zip(*[(r["gold"], LABELS[r["votes"][mid]]) for r in rows
                         if r["votes"][mid] in LABELS]) if rows else ([], [])
            k, n = kappa(list(g), list(p)); val[f"judge:{mid}"] = {"kappa": round(k, 3), "n": n}
        for tier in ("majority", "unanimous"):
            sel = [r for r in rows if r["jury_label"] is not None and
                   (r["tier"] == "unanimous" if tier == "unanimous"
                    else r["tier"] in ("unanimous", "majority"))]
            g = [r["gold"] for r in sel]; p = [LABELS[r["jury_label"]] for r in sel]
            k, n = kappa(g, p)
            val[tier] = {"kappa": round(k, 3), "n": n,
                         "coverage": round(n / max(1, len(rows)), 3)}
        receipt["validation"] = val
        print("  VALIDATION:", json.dumps(val))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(receipt, indent=1, default=float))
    print(f"  serper: {ser.summary() if ser else 'dry-run'}")
    print(f"  receipt -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans-from", choices=["gold"], default="gold")
    ap.add_argument("--config", default="Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--max-queries", type=int, default=800)
    ap.add_argument("--max-usd-per-judge", type=float, default=2.0)
    ap.add_argument("--out", default=str(HERE.parent / "receipts/jury_validation.json"))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    a.dry_run = a.dry_run and not a.execute
    run(a)
