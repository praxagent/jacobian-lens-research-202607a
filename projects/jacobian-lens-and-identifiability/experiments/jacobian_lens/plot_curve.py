"""Plot the workspace-band emergence curve: mid-band separation vs model size.

The figure for the blog post / Twitter. X = parameters (log). Y = mid-band
separation (how distinct the mid-network "workspace" band is). Points are
colored + connected WITHIN model family, because the band's magnitude has a
family offset — so a single mixed scatter looks noisy while the within-family
curves are clean. The random-J null (flat ~0) is the confound floor: the real
curves lifting off it is the evidence the band is real structure, and WHERE they
lift off is the emergence onset.

Run (after the ladder):
    uv run python plot_curve.py    # writes emergence_curve.png
    uv run python plot_curve.py --csv emergence_shared.csv --out emergence_curve_shared.png \
        --title-suffix " (tokenizer-commensurable probes)"   # the shared-probe variant
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def family(slug: str) -> str:
    for fam in ("gemma-4", "gemma-3", "gemma-2", "qwen3.5", "qwen3.6", "qwen3",
                "qwen2.5", "llama3.1", "llama3.3", "olmo-3", "gpt-oss",
                "pythia", "gpt2"):
        if slug.startswith(fam):
            return fam
    return re.split(r"[-.]", slug)[0]


def _load(name: str):
    p = HERE / name
    if not p.exists():
        return []
    out = []
    for r in csv.DictReader(open(p)):
        try:
            out.append((float(r["params"]), float(r["mid_sep"]), r["slug"]))
        except (ValueError, KeyError):
            pass
    return out


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="emergence.csv")
    ap.add_argument("--out", default="emergence_curve.png")
    ap.add_argument("--title-suffix", default="")
    args = ap.parse_args()

    real = _load(args.csv)
    null = _load("emergence_null.csv")
    if not real:
        print(f"no {args.csv} — run the ladder first.")
        return

    # group by family; base and instruct (-it) variants are separate curves in the
    # same color (solid vs dashed) — instruct-tuning shifts the band, so mixing them
    # in one line draws misleading zig-zags
    fams: dict[str, list] = {}
    for x, y, slug in real:
        fams.setdefault(family(slug), []).append((x, y, slug))
    for f in fams:
        fams[f].sort()

    fig, ax = plt.subplots(figsize=(11, 6.5))
    cmap = plt.get_cmap("tab10")
    has_it = False
    for i, (fam, pts) in enumerate(sorted(fams.items(), key=lambda kv: kv[1][0][0])):
        color = cmap(i % 10)
        base = [p for p in pts if not p[2].endswith("-it")]
        inst = [p for p in pts if p[2].endswith("-it")]
        if base:
            xs, ys, _ = zip(*base)
            style = "o-" if len(base) >= 2 else "o"
            ax.plot(xs, ys, style, color=color, lw=2, ms=7 if len(base) < 2 else 6,
                    label=fam, zorder=3)
        if inst:
            has_it = True
            xs, ys, _ = zip(*inst)
            style = "o--" if len(inst) >= 2 else "o"
            ax.plot(xs, ys, style, color=color, lw=1.4, ms=6, mfc="white",
                    label=fam if not base else None, zorder=3)
    if has_it:
        ax.plot([], [], "o--", color="#6b7280", lw=1.4, ms=6, mfc="white",
                label="instruct (-it) variants")

    if null:
        nx, ny, _ = zip(*sorted(null))
        ax.plot(nx, ny, "s--", color="#9ca3af", lw=1.4, ms=4,
                label="random-J null (floor)", zorder=2)
        ax.fill_between([min(nx), max(nx)], min(ny) - 0.005, max(ny) + 0.005,
                        color="#e5e7eb", alpha=0.5, zorder=0)

    ax.axhline(0, color="#d1d5db", lw=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("parameters (log scale)", fontsize=11)
    ax.set_ylabel("mid-band separation\n(distinctness of the mid-network 'workspace' band)",
                  fontsize=11)
    ax.set_title("A J-lens 'global workspace' band emerges with scale — and its onset "
                 "is charted here" + args.title_suffix,
                 fontsize=12, pad=14)
    ax.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.margins(y=0.12)
    fig.text(0.5, 0.005,
             "Method: linear-CKA block separation of J-lens token geometry across layers "
             "(pre-fitted Neuronpedia lenses on Anthropic's jlens). "
             "Null = random scale-matched transports. praxagent.ai",
             ha="center", fontsize=7.5, color="#6b7280")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = HERE / args.out
    fig.savefig(out, dpi=160)
    print(f"wrote {out}  ({len(real)} models, {len(fams)} families)")


if __name__ == "__main__":
    main()
