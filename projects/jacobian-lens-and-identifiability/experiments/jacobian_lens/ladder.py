"""Emergence sweep driver — run the CKA experiment up a model-size ladder.

Runs `cka_layers.py --slug X` in a SEPARATE subprocess per model, so memory is
fully released between models (a 32B lens + unembedding is many GB) and a single
failure (OOM, gated repo, 404) skips that model instead of killing the sweep.
Then prints the emergence curve: mid-band separation vs model size.

The question: does a distinct mid-network "workspace band" (mid_sep well above 0)
EMERGE as models get bigger? Flat ≈ the universal-workspace framing overreached;
a sharp rise at some scale = it's a real large-model phenomenon.

Run on the high-RAM box:
    uv run python ladder.py
    uv run python ladder.py --slugs qwen3-1.7b qwen3-4b qwen3-8b   # custom subset
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Dense size-ordered sweep. The resolver pulls the exact HF id from each lens's
# config.yaml; gated repos (Gemma, Llama — need a HF token) and missing repos
# just skip. Gemma-3 (270m→27b) is the widest single-FAMILY ladder, ideal for
# the curve shape — worth a HF token to unlock.
DEFAULT_LADDER = [
    "pythia-70m-deduped",   # ~0.07B  open
    "gpt2-small",           # ~0.12B  open
    "gemma-3-270m",         # ~0.27B  gated
    "qwen3.5-0.8b",         # ~0.8B   open
    "gemma-3-1b",           # ~1B     gated
    "qwen3-1.7b",           # ~1.7B   open
    "gemma-2-2b",           # ~2B     gated
    "qwen3.5-2b-pt",        # ~2B     open
    "gemma-3-4b",           # ~4B     gated
    "qwen3-4b",             # ~4B     open
    "qwen3.5-4b",           # ~4B     open
    "qwen2.5-7b-it",        # ~7B     open
    "olmo-3-1025-7b",       # ~7B     open (cross-family)
    "llama3.1-8b",          # ~8B     gated
    "qwen3-8b",             # ~8B     open
    "gemma-2-9b",           # ~9B     gated
    "qwen3.5-9b-pt",        # ~9B     open
    "gemma-3-12b",          # ~12B    gated
    "qwen3-14b",            # ~14B    open
    "gpt-oss-20b",          # ~20B    open (cross-family)
    "gemma-2-27b",          # ~27B    gated
    "gemma-3-27b",          # ~27B    gated
    "qwen3.5-27b",          # ~27B    open
    "qwen3-32b",            # ~32B    open
    "olmo-3-1125-32b",      # ~32B    open (cross-family)
]


def run_one(slug: str, null: bool = False) -> bool:
    cmd = [sys.executable, str(HERE / "cka_layers.py"), "--slug", slug]
    if null:
        cmd.append("--null")
    print(f"\n{'='*70}\n### {slug}{' [NULL]' if null else ''}\n{'='*70}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"  !! {slug} failed (rc={r.returncode}) — skipping (gated/OOM/404?)")
    return r.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs", nargs="*", default=DEFAULT_LADDER)
    ap.add_argument("--null-every", type=int, default=3,
                    help="also run a null control every Nth model (0=never)")
    args = ap.parse_args()

    ok = []
    for i, slug in enumerate(args.slugs):
        if run_one(slug):
            ok.append(slug)
            if args.null_every and i % args.null_every == 0:
                run_one(slug, null=True)

    # Print the emergence curve from the accumulated ledger.
    csv_path = HERE / "emergence.csv"
    if not csv_path.exists():
        print("\nno emergence.csv produced.")
        return
    import csv as _csv
    rows = list(_csv.DictReader(open(csv_path)))
    rows.sort(key=lambda r: float(r.get("params", 0) or 0))  # true size proxy
    print(f"\n\n{'='*72}\nEMERGENCE CURVE (mid-band separation vs parameters)\n{'='*72}")
    print(f"{'slug':22s} {'params':>9s} {'layers':>6s} {'mid_sep':>8s}")
    for r in rows:
        bar = "#" * max(0, int(float(r["mid_sep"]) * 200))
        p = float(r.get("params", 0) or 0)
        psz = f"{p/1e9:.2f}B" if p >= 1e9 else f"{p/1e6:.0f}M"
        print(f"{r['slug']:22s} {psz:>9s} {r['n_layers']:>6s} "
              f"{float(r['mid_sep']):>+8.3f}  {bar}")
    print("\nRead: mid_sep climbing with size => workspace band EMERGES (claim holds "
          "as a large-model phenomenon); staying ~0 => framing overreached.")


if __name__ == "__main__":
    main()
