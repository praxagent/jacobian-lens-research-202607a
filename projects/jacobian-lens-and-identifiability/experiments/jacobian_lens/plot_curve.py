"""Plot the workspace-band emergence curve: mid-band separation vs model size.

The figure for the blog post. X = parameter count (log). Y = mid-band separation
(how distinct the mid-network "workspace" band is). Real curve rising while the
random-J null stays flat near 0 is the evidence that the band is a real, scaling
phenomenon — and that its ONSET (where it lifts off ~0) is the quantity nobody
seems to have charted.

Run (after the ladder):
    uv run python plot_curve.py    # writes emergence_curve.png
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    p = HERE / name
    if not p.exists():
        return []
    rows = []
    for r in csv.DictReader(open(p)):
        try:
            rows.append((float(r["params"]), float(r["mid_sep"]), r["slug"]))
        except (ValueError, KeyError):
            pass
    return sorted(rows)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = _load("emergence.csv")
    null = _load("emergence_null.csv")
    if not real:
        print("no emergence.csv — run the ladder first.")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    rx, ry, rlab = zip(*real)
    ax.plot(rx, ry, "o-", color="#2563eb", lw=2, ms=7, label="J-lens (real lens)")
    for x, y, lab in real:
        ax.annotate(lab, (x, y), fontsize=7, xytext=(0, 6),
                    textcoords="offset points", ha="center", rotation=30)
    if null:
        nx, ny, _ = zip(*null)
        ax.plot(nx, ny, "s--", color="#9ca3af", lw=1.5, ms=5,
                label="random-J null (confound floor)")

    ax.axhline(0, color="#d1d5db", lw=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("parameters (log scale)")
    ax.set_ylabel("mid-band separation\n(distinctness of the 'workspace' band)")
    ax.set_title("Emergence of the J-lens 'workspace band' with model scale")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    out = HERE / "emergence_curve.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
