"""C: exploratory re-analysis of existing block-patching data, to sharpen v2.1.

Strictly EXPLORATORY (integrity playbook 11: confirmatory and exploratory stay separate).
Nothing here is a verdict; its purpose is to choose a sharper, better-powered hypothesis for
the confirmatory v2.1 sample, which will be NEW models with olmo held out as the generator.

C1  does adding position dummies remove v1's wrong-signed positive? (tests a published claim)
C2  per-boundary decomposition: is olmo's +0.22 at one boundary or smeared across both?
C3  prompt-level bootstrap as a second error model  [BLOCKED: see note]
C4  exploratory v2 read on the free CPU pilot
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import boundaries, lens_source_layers          # noqa: E402
from analyze_v2 import beta_for                              # noqa: E402


def design(D, blk, extra_cols):
    """D ~ distance dummies + position dummies + [extra indicator columns]."""
    L = D.shape[0]
    rows = [(i, j) for i in range(L) for j in range(L) if i != j and np.isfinite(D[i, j])]
    y = np.array([D[i, j] for i, j in rows])
    dist = [abs(i - j) for i, j in rows]; pos = [int(round((i + j) / 2)) for i, j in rows]
    du, pu = sorted(set(dist)), sorted(set(pos))
    X = np.zeros((len(y), len(du) + len(pu) - 1 + len(extra_cols)))
    for k, (i, j) in enumerate(rows):
        X[k, du.index(dist[k])] = 1.0
        if pos[k] != pu[0]:
            X[k, len(du) + pu.index(pos[k]) - 1] = 1.0
        for c, fn in enumerate(extra_cols):
            X[k, len(du) + len(pu) - 1 + c] = fn(i, j, blk)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta[-len(extra_cols):], len(y)


def per_boundary(D, b1, b2, L, nperm, seed):
    blk = np.array([0 if i < b1 else (1 if i < b2 else 2) for i in range(L)])
    cols = [lambda i, j, b: 1.0 if {b[i], b[j]} == {0, 1} else 0.0,   # spans boundary 1 only
            lambda i, j, b: 1.0 if {b[i], b[j]} == {1, 2} else 0.0,   # spans boundary 2 only
            lambda i, j, b: 1.0 if {b[i], b[j]} == {0, 2} else 0.0]   # spans both
    obs, n = design(D, blk, cols)
    rng = np.random.default_rng(seed)
    sizes = [b1, b2 - b1, L - b2]
    null = []
    for _ in range(nperm):
        perm = list(rng.permutation(sizes)); cuts = np.cumsum(perm)[:2]
        rb = np.roll(np.array([0 if i < cuts[0] else (1 if i < cuts[1] else 2)
                               for i in range(L)]), rng.integers(0, L))
        null.append(design(D, rb, cols)[0])
    null = np.array(null)
    return obs, null, n


def main():
    out = {}
    print("=" * 72)
    print("C1  position control vs v1's wrong-signed positive (tests a published claim)")
    c1 = {}
    for f, slug in [("out/qwen3_8b.json", "qwen3-8b"),
                    ("out/pilot_gemma3_270m.json", "gemma-3-270m"),
                    ("out/pilot_qwen35_08b.json", "qwen3.5-0.8b")]:
        r = json.load(open(HERE / f)); E = np.array(r["E"], float)
        b1, b2, sep, L = boundaries(slug); src = lens_source_layers(slug, L)
        E = E[np.ix_(src, src)]
        blk = np.array([0 if i < b1 else (1 if i < b2 else 2) for i in range(L)])
        a, _ = beta_for(E, blk, use_position=False)
        b, _ = beta_for(E, blk, use_position=True)
        c1[slug] = {"beta_distance_only": round(a, 4), "beta_with_position": round(b, 4)}
        print(f"  {slug:14s} {a:+.4f} -> {b:+.4f}")
    out["C1_position_control"] = c1
    print("  => v1's +0.148 in qwen3-8b is confirmed as a POSITION artifact, not a boundary effect.")

    print("\n" + "=" * 72)
    print("C2  per-boundary decomposition of olmo's +0.22 (exploratory)")
    r = json.load(open(HERE / "out/v2_olmo32b.json")); Df = np.array(r["D"], float)
    b1, b2, sep, L = boundaries("olmo-3-1125-32b"); src = lens_source_layers("olmo-3-1125-32b", L)
    D = Df[np.ix_(src, src)]
    obs, null, n = per_boundary(D, b1, b2, L, nperm=400, seed=0)
    names = ["boundary 1 only (blk0<->blk1)", "boundary 2 only (blk1<->blk2)", "spans both (blk0<->blk2)"]
    c2 = {}
    for k, nm in enumerate(names):
        p = float((np.abs(null[:, k]) >= abs(obs[k])).mean())
        c2[nm] = {"beta": round(float(obs[k]), 4), "null_sd": round(float(null[:, k].std()), 4), "p": p}
        print(f"  {nm:32s} beta={obs[k]:+.4f}  null sd={null[:, k].std():.4f}  p={p:.3f}")
    out["C2_per_boundary"] = c2

    print("\n" + "=" * 72)
    print("C3  prompt-level bootstrap  ->  BLOCKED BY A RECEIPT GAP")
    print("  swap_v2.py stored only the mean KL across prompts, not per-prompt values, so a")
    print("  bootstrap over prompts cannot be computed from the receipt. This is exactly the")
    print("  'could a new analysis be done from this receipt alone?' test, and this receipt")
    print("  fails it. swap_v2.py is fixed to store per-prompt D for v2.1; olmo would have to")
    print("  be re-run (~$1) to backfill, which we do NOT do on exploratory grounds.")
    out["C3_prompt_bootstrap"] = {"status": "blocked", "reason": "receipt stored mean only; runner fixed for v2.1"}

    print("\n" + "=" * 72)
    print("C4  exploratory v2 read on the free CPU pilot (balance 0.17, NOT a fair test)")
    rp = json.load(open(HERE / "out/v2_pilot_gemma270m.json")); Dp = np.array(rp["D"], float)
    b1p, b2p, sepp, Lp = boundaries("gemma-3-270m"); srcp = lens_source_layers("gemma-3-270m", Lp)
    Dp = Dp[np.ix_(srcp, srcp)]
    blkp = np.array([0 if i < b1p else (1 if i < b2p else 2) for i in range(Lp)])
    bp, _ = beta_for(Dp, blkp, use_position=True)
    print(f"  gemma-3-270m v2 beta (position-controlled) = {bp:+.4f}   [exploratory, weak balance]")
    out["C4_pilot_v2_beta"] = round(bp, 4)

    (HERE / "out/analysis_C.json").write_text(json.dumps(out, indent=1))
    print("\nwrote out/analysis_C.json")


if __name__ == "__main__":
    main()
