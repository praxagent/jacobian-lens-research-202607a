"""#2-followup: WHERE does instruct tuning flatten the readout? Uniformly, or by pinning
distant (early<->late) layers together?

For each base/instruct pair with both shared-vocab CKA maps cached, diff the maps
(it - base, same architecture so same L) and aggregate the change by band region
(early/mid/late x early/mid/late). If instruct raises the off-block (early-late) CKA most,
it is pulling distant layers into a common readout; if it raises uniformly, it is a global
contraction. Tiny compute (25x25 maps), CPU, no large arrays.
"""
from __future__ import annotations
import json, glob, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SM = HERE / "atlas_out/shared_maps"
OUT = HERE / "decompose_out"


def blocks(M):
    """Mean CKA in each of the 9 early/mid/late x early/mid/late regions (off-diagonal)."""
    L = M.shape[0]; th = np.array_split(np.arange(L), 3); names = ["E", "M", "L"]
    out = {}
    for a, an in zip(th, names):
        for b, bn in zip(th, names):
            v = [M[i, j] for i in a for j in b if i != j]
            out[an + bn] = float(np.mean(v)) if v else 1.0
    return out


def main():
    pairs = []
    for f in sorted(glob.glob(str(SM / "*-it.npz"))):
        it_slug = os.path.basename(f)[:-4]; base_slug = it_slug[:-3]
        bf = SM / f"{base_slug}.npz"
        if bf.exists():
            pairs.append((base_slug, bf, Path(f)))

    rows = []
    for base_slug, bf, itf in pairs:
        Mb = np.load(bf, allow_pickle=True)["cka"]; Mi = np.load(itf, allow_pickle=True)["cka"]
        if Mb.shape != Mi.shape:
            print(f"skip {base_slug}: shape {Mb.shape} vs {Mi.shape}"); continue
        bb, bi = blocks(Mb), blocks(Mi)
        delta = {k: bi[k] - bb[k] for k in bb}
        rows.append({"model": base_slug, "delta_blocks": delta,
                     "EL_base": round(bb["EL"], 4), "EL_it": round(bi["EL"], 4)})
        print(f"{base_slug:14s} dEE={delta['EE']:+.3f} dML={delta['ML']:+.3f} "
              f"dEL={delta['EL']:+.3f}  (EL {bb['EL']:.3f}->{bi['EL']:.3f})")

    # aggregate mean delta per block across pairs
    keys = ["EE", "EM", "EL", "MM", "ML", "LL"]
    agg = {k: round(float(np.mean([r["delta_blocks"][k] for r in rows])), 4) for k in keys}
    print("\n=== mean CKA change (instruct - base) by region, across", len(rows), "pairs ===")
    for k in keys:
        print(f"  {k}: {agg[k]:+.4f}")
    # is the early<->late (EL) region raised more than the within-block diagonal average?
    diag = np.mean([agg["EE"], agg["MM"], agg["LL"]])
    print(f"\n  early<->late (EL) change {agg['EL']:+.4f}  vs  within-block avg {diag:+.4f}")
    verdict = ("pins distant layers together (EL raised most)" if agg["EL"] >= max(agg["EE"], agg["LL"]) and agg["EL"] > 0
               else "raises within-block alignment most" if diag > agg["EL"]
               else "mixed")
    print(f"  verdict: instruct {verdict}")
    OUT.mkdir(exist_ok=True)
    (OUT / "instruct_contraction.json").write_text(json.dumps(
        {"n_pairs": len(rows), "mean_delta_blocks": agg, "within_block_avg": round(float(diag), 4),
         "verdict": verdict, "pairs": rows}, indent=1))
    print("INSTRUCT_CONTRACTION_DONE")


if __name__ == "__main__":
    main()
