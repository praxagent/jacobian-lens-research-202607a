"""Full jury labeling of the Gemma-3-12B entity spans (the release dataset run).

Inputs: the arm-4 receipt's 6,503 extracted spans (cid + entity + char offsets) and the
regenerated completion texts (completions_cache.jsonl, deterministic greedy). For each
completion: shared Serper evidence per span (archived + cached) -> 3 cross-provider judges
vote independently -> unanimous/majority tiers. Output receipt is dataset-ready: per-row
votes, tier, jury label, evidence links; verbatim snippets stay in the local serper cache
(receipts-only), NOT for the public card (legal surface: notes+URLs public).

  .venv/bin/python jury_gemma3.py --max-queries 9000 --dry-run
  .venv/bin/python jury_gemma3.py --max-queries 9000 --execute
"""
from __future__ import annotations

import argparse, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "grounding"))
sys.path.insert(0, str(HERE.parent))
from llm import GuardedLLM, BudgetExceeded    # noqa: E402
from serper import Serper, SerperExceeded     # noqa: E402
from jury import JUDGES, SYS, parse_labels, _env  # noqa: E402

ARM4 = HERE.parents[1] / "gemma3_agreement/receipts/receipt_gemma3_agreement.json"
CACHE = HERE.parents[1] / "gemma3_agreement/receipts/completions_cache.jsonl"
LABELS = {"Supported": 0, "Not Supported": 1}


def run(a):
    cache = Path(a.completions) if a.completions else CACHE
    comps = {json.loads(l)["i"]: json.loads(l)["c"] for l in cache.read_text().splitlines()}
    by_comp = defaultdict(list)
    if a.spans:   # amendment 7: {cid: [entity,...]} from extract_entities_v2.py
        for cid, ents in json.loads(Path(a.spans).read_text())["spans"].items():
            by_comp[int(cid.split(":")[1])] = [e.strip() for e in ents]
    else:
        arm4 = json.loads(ARM4.read_text())
        for row in arm4["per_span"]:
            i = int(row["cid"].split(":")[1])
            by_comp[i].append(row["entity"].strip())
    # dedupe entities per completion, preserve order
    for i in by_comp:
        by_comp[i] = list(dict.fromkeys(by_comp[i]))
    n_ent = sum(len(v) for v in by_comp.values())
    print(f"  {len(by_comp)} completions, {n_ent} unique entities")

    ser_cache = Path(a.serper_cache) if a.serper_cache else \
        HERE.parent / "receipts/serper_cache_g3.jsonl"
    ser = Serper(_env("SERPER_DEV_API_KEY"), ser_cache,
                 max_queries=a.max_queries) if not a.dry_run else None
    juries = {mid: GuardedLLM(_env("OPENROUTER_API_KEY"), mid, ri, ro,
                              max_usd=a.max_usd_per_judge, max_calls=len(by_comp) + 20,
                              max_tokens=1400, dry_run=a.dry_run)
              for mid, ri, ro in JUDGES}
    rows = []
    t0 = time.time()
    for i in sorted(by_comp):
        comp = comps.get(i)
        if comp is None:
            continue
        ents = by_comp[i]
        try:
            evidence = [ser.search(f"{e} {comp[:70]}") if ser else [] for e in ents]
        except SerperExceeded as e:
            print(f"  search cap: {e}"); break
        lines = [f"TEXT (context): {comp[:600]}...", "", "ENTITIES:"]
        for j, e in enumerate(ents):
            refs = " || ".join(f"[{v['title']}] {v['snippet'][:280]}"
                               for v in evidence[j]) or "(no results)"
            lines.append(f'{j}. "{e}"  SEARCH RESULTS: {refs}')
        lines.append("\nReturn the JSON array now.")
        prompt = "\n".join(lines)
        votes = {}
        try:
            for mid in juries:
                txt, _ = juries[mid].chat([{"role": "system", "content": SYS},
                                           {"role": "user", "content": prompt}])
                votes[mid] = parse_labels(txt)
        except BudgetExceeded as e:
            print(f"  judge budget: {e}"); break
        for j, e in enumerate(ents):
            v = {mid: votes.get(mid, {}).get(j, "Insufficient") for mid in juries}
            top, ntop = Counter(v.values()).most_common(1)[0]
            rows.append({"cid": f"g3:{i}", "entity": e, "votes": v,
                         "jury_label": top if top in LABELS else None,
                         "tier": "unanimous" if ntop == 3 else "majority" if ntop == 2 else "split",
                         "evidence_links": [x["link"] for x in evidence[j]]})
        if i % 20 == 0:
            print(f"  [{len(rows)} rows] comp {i}/{len(by_comp)} "
                  f"searches={ser.spent if ser else 0}", flush=True)

    tiers = Counter(r["tier"] for r in rows)
    labeled = [r for r in rows if r["jury_label"] is not None and r["tier"] != "split"]
    receipt = {"what": "Gemma-3-12B jury labels (release run). Labels = cross-provider jury "
                       "agreement on archived search evidence, validated at kappa .598 "
                       "(majority) / .691 (unanimous) vs public gold — NOT ground truth.",
               "judges": [m for m, _, _ in JUDGES],
               "n_completions": len(by_comp), "n_rows": len(rows),
               "tiers": dict(tiers), "n_labeled": len(labeled),
               "label_dist": dict(Counter(r["jury_label"] for r in labeled)),
               "rows": rows, "serper": ser.summary() if ser else "dry-run",
               "judge_guards": {m: juries[m].summary() for m in juries},
               "seconds": round(time.time() - t0, 1), "args": vars(a)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(receipt, indent=1, default=float))
    print(f"  tiers: {dict(tiers)} | labeled: {len(labeled)}")
    print(f"  serper: {ser.summary() if ser else 'dry-run'}")
    print("  JURY_G3_DONE ->", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-queries", type=int, default=9000)
    ap.add_argument("--max-usd-per-judge", type=float, default=5.0)
    ap.add_argument("--spans", default=None,
                    help="entities json from extract_entities_v2.py (default: arm-4 receipt)")
    ap.add_argument("--completions", default=None, help="completions cache jsonl override")
    ap.add_argument("--serper-cache", default=None)
    ap.add_argument("--out", default=str(HERE.parent / "receipts/jury_gemma3_labels.json"))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    a.dry_run = a.dry_run and not a.execute
    run(a)
