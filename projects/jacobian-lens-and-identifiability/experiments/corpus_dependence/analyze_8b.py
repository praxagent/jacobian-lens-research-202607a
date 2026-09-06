"""Apply PREREG_8B.md's frozen decision table to the llama3.1-8b arm of results.json.

  anchor: our wiki_a map within 0.01 map distance of the public Neuronpedia shared map, else FAILED RUN
  P1:     corpus map distance > 10x the seed null
  P2:     corpus boundary shift <= max(2, seed boundary shift)
  P3:     descriptive: |band_sep(code) - band_sep(wiki_a)| > seed band shift

Writes results_8b.json and prints the statement the decision table assigns.
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = "llama3.1-8b"
R = json.loads((HERE / "results.json").read_text())
d = R[SLUG]
seed, corp = d["seed_null"], d["corpus"]
anchor = d.get("anchor_public_fit") or {}
anchor_ok = anchor.get("public_vs_wiki_a_map_distance") is not None and anchor["public_vs_wiki_a_map_distance"] <= 0.01
ratio = corp["map_distance"] / max(seed["map_distance"], 1e-12)
p1 = ratio > 10
p2 = corp["boundary_shift"] <= max(2, seed["boundary_shift"])
p3 = corp["band_shift"] > seed["band_shift"]
if not anchor_ok:
    statement = "ANCHOR FAILED: our fit is not comparable to the public lens; no corpus statement at 8B"
elif p1 and p2:
    statement = "the corrected corpus picture replicates at 8B: map moves, boundaries do not"
elif p1 and not p2:
    statement = "at 8B the corpus does move the boundaries; the small-model result was scale-limited"
else:
    statement = "the corpus effect on the map shrinks with scale; reported as such"
out = {"slug": SLUG, "prereg": "PREREG_8B.md (73fb7cb)", "unembedding_tensor": d.get("unembedding_tensor"),
       "n_layers": d["n_layers"], "boundaries": d["boundaries"], "band_sep": d["band_sep"],
       "seed_null": seed, "corpus": corp, "corpus_over_seed_ratio": ratio, "anchor": anchor, "anchor_ok": anchor_ok,
       "P1_map_moves_gt10x": p1, "P2_boundaries_stay_le_max2_seed": p2, "P3_band_changes_gt_seed": p3,
       "statement": statement}
(HERE / "results_8b.json").write_text(json.dumps(out, indent=1))
print(f"{SLUG}: probe rows from {out['unembedding_tensor']}; L={d['n_layers']}")
print(f"  boundaries wiki_a={tuple(d['boundaries']['wiki_a'])} wiki_b={tuple(d['boundaries']['wiki_b'])} code={tuple(d['boundaries']['code'])}")
print(f"  band_sep {d['band_sep']}")
print(f"  seed null: shift {seed['boundary_shift']}, map d {seed['map_distance']:.5f}, band {seed['band_shift']:.4f}")
print(f"  corpus:    shift {corp['boundary_shift']}, map d {corp['map_distance']:.4f} ({ratio:.0f}x), band {corp['band_shift']:.4f}")
print(f"  anchor: public vs wiki_a map d {anchor.get('public_vs_wiki_a_map_distance')} (public boundaries {anchor.get('public_boundaries')}) -> {'OK' if anchor_ok else 'FAIL'}")
print(f"  P1 {p1}  P2 {p2}  P3 {p3}")
print(f"  STATEMENT: {statement}")
