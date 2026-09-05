"""Self-test: reproduce the campaign's cached gpt2-small maps from the public Neuronpedia lens.

Two checks, both CPU, a couple of minutes:
  shared probe, geometry fp16 mirror: map equals atlas_out/shared_maps/gpt2-small.npz to 1e-5
  own probe,    geometry fp16 mirror: mid_sep equals summary.csv (0.015) to 1e-3, seg == (2, 4)
plus the fp32 (default) run on both probes, reporting how far fp32 geometry sits from the fp16
cache so the documented tolerance is measured rather than asserted.

Run from this directory:  <repo>/.venv/bin/python selftest.py
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from jlens_atlas.run import run_atlas  # noqa: E402

ATLAS = HERE.parents[1] / "experiments/jspace_atlas/atlas_out"
SCR = Path("/tmp/claude-1000/-home-ubuntu-PRAX-research-and-replications/40d850ca-3242-496a-99f8-46bbc7ba39cc/scratchpad/jlens_atlas_selftest")
SCR.mkdir(parents=True, exist_ok=True)
SLUG = "gpt2-small"
ok = True


def check(name, cond, detail):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    ok = ok and cond


print("== shared probe ==")
ref = np.load(ATLAS / f"shared_maps/{SLUG}.npz")
for gd in ("fp16", "fp32"):
    s = run_atlas(neuronpedia=SLUG, probe="shared", geometry_dtype=gd, out_dir=SCR / f"shared_{gd}",
                  argv=["selftest", "shared", gd])
    M = np.load(SCR / f"shared_{gd}/cka.npz")["cka"]
    dmax = float(np.abs(M - ref["cka"]).max()); dms = abs(s["mid_sep"] - float(ref["mid_sep"]))
    print(f"  geometry {gd}: max |cka - cached| = {dmax:.2e}, |mid_sep - cached| = {dms:.2e}, "
          f"fitted_sep {s['fitted_sep']:.4f} (cached {float(ref['fitted_sep']):.4f}), seg {s['fitted_seg']}")
    if gd == "fp16":
        check("shared map reproduces cached map", dmax <= 1e-5, f"max abs diff {dmax:.2e} <= 1e-5")
        check("shared mid_sep reproduces cached", dms <= 1e-4, f"diff {dms:.2e} <= 1e-4")
    else:
        check("fp32 geometry stays within 1e-3 of the fp16 cache", dmax <= 1e-3, f"max abs diff {dmax:.2e}")

print("== own probe ==")
row = next(r for r in csv.DictReader(open(ATLAS / "summary.csv")) if r["slug"] == SLUG)
own = np.load(ATLAS / f"{SLUG}.npz")
for gd in ("fp16", "fp32"):
    s = run_atlas(neuronpedia=SLUG, probe="own", geometry_dtype=gd, out_dir=SCR / f"own_{gd}",
                  argv=["selftest", "own", gd])
    M = np.load(SCR / f"own_{gd}/cka.npz")["cka"]
    dmax = float(np.abs(M - own["cka"]).max())
    print(f"  geometry {gd}: mid_sep {s['mid_sep']:.5f} (summary.csv {row['mid_sep']}, npz {float(own['mid_sep']):.5f}), "
          f"seg {s['fitted_seg']} (cached {tuple(int(x) for x in own['seg'])}), max |cka - cached| = {dmax:.2e}")
    if gd == "fp16":
        check("own mid_sep matches summary.csv to 1e-3", abs(s["mid_sep"] - float(row["mid_sep"])) <= 1e-3,
              f"{s['mid_sep']:.5f} vs {row['mid_sep']}")
        check("own seg == (2, 4)", tuple(s["fitted_seg"]) == (int(row["seg_b1"]), int(row["seg_b2"])), str(s["fitted_seg"]))
        check("own map reproduces cached map", dmax <= 1e-5, f"max abs diff {dmax:.2e} <= 1e-5")

print("== null sanity ==")
s = run_atlas(neuronpedia=SLUG, probe="shared", out_dir=SCR / "null_check", argv=["selftest", "null"])
check("random-transport null mid_sep is near zero", abs(s["null"]["mid_sep"]) < 0.02,
      f"null mid_sep {s['null']['mid_sep']:+.4f}")
print("SELFTEST", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
