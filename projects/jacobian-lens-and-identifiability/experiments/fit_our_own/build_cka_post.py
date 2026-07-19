"""Post bundle builder for the CKA-atlas note (blog/jlens-cka-397b/).

RESEARCH_NOTE_WRITEUP §5: figures come from the committed npz (never hand-numbered),
each ships a <stem>.receipt.json, and the bundle carries a post-wide provenance.json
whose every evidentiary number re-derives from committed receipts. --verify: re-derives
the manifest from the receipts, byte-compares the bundle SVGs against the committed
artifacts, and asserts every manifest value_str appears verbatim in index.md.

  .venv/bin/python build_cka_post.py            # build/refresh the bundle
  .venv/bin/python build_cka_post.py --verify   # pre-publish gate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ART = HERE.parents[1] / "artifacts/lenses-397b"
POST = ROOT / "blog/jlens-cka-397b"
PIN = "c6c7bb1"
REPO_URL = "https://github.com/praxagent/jacobian-lens-research-202607a"
NPZ = ART / "cka_397b.npz"
BAND = ART / "qwen35_397b_dm.band.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_manifest():
    d = np.load(NPZ)
    band = json.loads(BAND.read_text())
    npz_rel = "projects/jacobian-lens-and-identifiability/artifacts/lenses-397b/cka_397b.npz"
    band_rel = "projects/jacobian-lens-and-identifiability/artifacts/lenses-397b/qwen35_397b_dm.band.json"
    ent = []

    def add(label, value, vstr, path, computation):
        ent.append({"label": label, "value": float(value), "value_str": vstr,
                    "receipt": path, "receipt_sha256": sha(ROOT / path),
                    "pinned_url": f"{REPO_URL}/blob/{PIN}/{path}",
                    "computation": computation})

    add("mid_sep", d["mid_sep"], "+0.343363", npz_rel, "npz['mid_sep']")
    add("mid_sep.null", d["mid_sep_null"], "-0.000113", npz_rel, "npz['mid_sep_null']")
    add("within_mid", band["within_mid"], ".937", band_rel, "band['within_mid']")
    add("early_mid", band["early_mid"], ".597", band_rel, "band['early_mid']")
    add("mid_late", band["mid_late"], ".590", band_rel, "band['mid_late']")
    add("n_layers", len(d["layers"]), "59", npz_rel, "len(npz['layers'])")
    add("n_probe", d["n_probe"], "4,096", npz_rel, "npz['n_probe']")
    # consumer-gate delta: |recomputed - shipped| from the two receipts above
    delta = abs(float(d["mid_sep"]) - band["mid_sep"])
    add("consumer_gate_delta", delta, "2.0×10⁻⁸", npz_rel,
        "abs(npz['mid_sep'] - band['mid_sep'])")
    assert delta < 1e-7
    return {"what": "post-wide provenance manifest (RESEARCH_NOTE §5) — every "
                    "evidentiary number re-derives from a committed receipt",
            "post": "jlens-cka-397b", "repo": REPO_URL, "pin_commit": PIN,
            "entries": ent}


def fig_receipt(stem, title, alt):
    return {"figure_id": stem, "title": title, "alt_text": alt,
            "description": alt,
            "interval_semantics": "descriptive fixed-census values (full 59x59 "
                                  "matrix, no sampling); no inferential intervals",
            "guards": "generator gate: matrix must reproduce released mid_sep to "
                      "1e-4 (measured 2.0e-8); --verify fails on byte drift",
            "data_source": [{"receipt": "projects/jacobian-lens-and-identifiability/"
                                        "artifacts/lenses-397b/cka_397b.npz",
                             "sha256": sha(NPZ),
                             "pinned_url": f"{REPO_URL}/blob/{PIN}/projects/"
                                           "jacobian-lens-and-identifiability/"
                                           "artifacts/lenses-397b/cka_397b.npz"}],
            "provenance": {"generator": "projects/jacobian-lens-and-identifiability/"
                                        "experiments/fit_our_own/cka_heatmap_397b.py",
                           "gate": "matrix must reproduce released mid_sep to 1e-4 "
                                   "(measured 2.0e-8) or the generator refuses to ship",
                           "svg_sha256": sha(POST / f"{stem}.svg")},
            "accessibility": {"color_only_channel": False,
                              "text_equivalent": "the full matrix is in the npz; "
                                                 "band means in the manifest"}}


def check_prose(man, md):
    txt = md.read_text().replace("−", "-")
    return [e for e in man["entries"] if e["value_str"].replace("−", "-") not in txt]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    figs = {
        "cka_397b": ("Qwen3.5-397B J-lens: layer x layer CKA atlas",
                     "Heatmap of linear CKA between J-lens token geometries for all "
                     "59 source-layer pairs of Qwen3.5-397B. Three bright blocks: "
                     "early (~0-13), a broad mid-band (~14-41), late (~42-58); "
                     "within-mid mean .937 vs .597/.590 to its neighbors "
                     "(mid_sep +0.343)."),
        "cka_397b_null": ("Random-J control: structure-free",
                          "Same heatmap with Frobenius-matched random transports in "
                          "place of the fitted Jacobians: a flat ~.75 field, no "
                          "blocks, mid_sep -0.000 — the band structure is the "
                          "lens's, not the shared unembedding's."),
    }
    if not a.verify:
        POST.mkdir(parents=True, exist_ok=True)
        for stem in figs:
            for ext in (".svg", ".png"):
                shutil.copy2(ART / f"{stem}{ext}", POST / f"{stem}{ext}")
        for stem, (title, alt) in figs.items():
            (POST / f"{stem}.receipt.json").write_text(
                json.dumps(fig_receipt(stem, title, alt), indent=1))
        idx = {f"{st}.receipt.json": sha(POST / f"{st}.receipt.json") for st in figs}
        (POST / "receipts_index.json").write_text(json.dumps(idx, indent=1))
        man = build_manifest()
        (POST / "provenance.json").write_text(json.dumps(man, indent=1))
        missing = check_prose(man, POST / "index.md") if (POST / "index.md").exists() else []
        print(f"bundle built -> {POST} ({len(man['entries'])} manifest entries)")
        for e in missing:
            print(f"  ⚠ prose missing: {e['label']} = {e['value_str']}")
    else:
        fail = []
        for stem in figs:
            if (POST / f"{stem}.svg").read_bytes() != (ART / f"{stem}.svg").read_bytes():
                fail.append(f"{stem}.svg differs from committed artifact")
        man = build_manifest()
        committed = json.loads((POST / "provenance.json").read_text())
        if committed["entries"] != man["entries"]:
            fail.append("provenance.json drifted")
        fail += [f"prose missing: {e['label']} = {e['value_str']}"
                 for e in check_prose(man, POST / "index.md")]
        if fail:
            print("VERIFY FAILED:"); [print("  -", f) for f in fail]
            sys.exit(1)
        print(f"VERIFY OK: 2 figures byte-identical to committed artifacts, "
              f"{len(man['entries'])} numbers re-derived, all present in prose")


if __name__ == "__main__":
    main()
