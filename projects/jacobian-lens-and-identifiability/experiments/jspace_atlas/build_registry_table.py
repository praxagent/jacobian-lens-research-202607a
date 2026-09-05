"""Model registry for the 36-lens zoo, generated from the analysis artifacts (2026-09-05).

Replaces a hand-carried copy of the Neuronpedia listing (37 rows, two lenses that appear in no
figure, no row for the 397B). Columns are the facts a reader needs to check any claim: lens source,
layers, width, vocabulary, own-vocabulary and shared-probe band statistics, shared-probe fitted
boundaries, and whether those boundaries are identified. Writes atlas_out/model_registry_zoo.md
and .json; the note's appendix is spliced from the .md.
"""
import csv, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; OUT = HERE / "atlas_out"
own = {r["slug"]: r for r in csv.DictReader(open(OUT / "summary.csv"))}
shared = {r["slug"]: r for r in csv.DictReader(open(OUT / "shared_summary.csv"))}
conc = json.load(open(HERE / "decompose_out/zoo_concentration.json"))
ident = json.load(open(OUT / "boundary_identifiability.json"))["models"]
reg = {}
for line in open(OUT / "model_registry.md"):
    if line.startswith("| `"):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        reg[c[0].strip("`")] = {"hf_id": c[1].strip("`"), "lineage": c[2], "vocab": c[5]}
rows = []
for slug in sorted(own):
    o = own[slug]; sh = shared.get(slug, {}); c = conc.get(slug, {}); r = reg.get(slug, {})
    idn = ident.get(slug, {}).get("shared", {})
    src = "own fit, 24 prompts" if slug == "qwen35-397b-own" else "Neuronpedia public fit"
    vocab = c.get("vocab") or r.get("vocab", "n/a")
    lineage = r.get("lineage") or sh.get("family", "")
    if slug == "qwen35-397b-own": lineage = "qwen3.5"
    seg = idn.get("fitted_seg"); segs = f"{seg[0]} / {seg[1]}" if seg else "n/a (too few layers)"
    idf = ("yes" if idn.get("identified") else "no") if seg else "n/a"
    rows.append({"slug": slug, "hf_id": o["hf_id"], "lineage": lineage, "lens_source": src,
                 "lens_layers": int(o["n_layers"]), "d_model": int(o["d_model"]), "vocab": vocab,
                 "own_mid_sep": float(o["mid_sep"]), "shared_mid_sep": float(sh["mid_sep"]) if sh else None,
                 "shared_fitted_sep": float(sh["fitted_sep"]) if sh else None,
                 "shared_boundaries": segs, "boundaries_identified": idf})
md = ["| model | HF id | lineage | lens | layers | d_model | vocab | mid_sep own | mid_sep shared | fitted boundaries (shared) | identified |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    sm = "n/a" if r["shared_mid_sep"] is None else f"{r['shared_mid_sep']:.3f}"
    md.append(f"| `{r['slug']}` | `{r['hf_id']}` | {r['lineage']} | {r['lens_source']} | {r['lens_layers']} | {r['d_model']} | {r['vocab']} | {r['own_mid_sep']:.3f} | {sm} | {r['shared_boundaries']} | {r['boundaries_identified']} |")
(OUT / "model_registry_zoo.md").write_text("\n".join(md) + "\n")
json.dump({"n": len(rows), "rows": rows, "sources": ["atlas_out/summary.csv", "atlas_out/shared_summary.csv",
           "decompose_out/zoo_concentration.json", "atlas_out/boundary_identifiability.json", "atlas_out/model_registry.md"]},
          open(OUT / "model_registry_zoo.json", "w"), indent=1)
print(f"wrote {len(rows)} rows; identified {sum(r['boundaries_identified']=='yes' for r in rows)}; lineages {sorted(set(r['lineage'] for r in rows))}")
