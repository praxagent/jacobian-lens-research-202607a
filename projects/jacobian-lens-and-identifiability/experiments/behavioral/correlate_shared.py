"""Tokenizer-confound robustness check for the behavioral correlation.

Re-runs correlate.py's exact pipeline with ONE change: the geometric band statistic
(mid_sep) comes from emergence_shared.csv — the re-sweep restricted to probe token
strings shared across every tokenizer in the study — instead of the own-vocab
emergence.csv. If the geometry→behavior correlation is real (and not an artifact of
each model being probed in its own vocabulary), the rank correlations should survive
the switch.

Writes behavioral_correlation_shared.csv, then re-runs the original own-vocab
correlate.main() so behavioral_correlation.csv is left exactly as before.

Run:  uv run python correlate_shared.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import correlate

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "jacobian_lens" / "emergence_shared.csv"


def load_midsep_shared() -> dict[str, float]:
    return {r["slug"]: float(r["mid_sep"]) for r in csv.DictReader(open(SHARED))}


def main() -> None:
    print(f"=== SHARED-VOCAB mid_sep (from {SHARED.name}) ===")
    correlate.load_midsep = load_midsep_shared
    correlate.main()
    out = HERE / "behavioral_correlation.csv"
    out.replace(HERE / "behavioral_correlation_shared.csv")
    print("renamed -> behavioral_correlation_shared.csv")

    print("\n=== OWN-VOCAB mid_sep (original, regenerating the canonical csv) ===")
    correlate.load_midsep = correlate.__dict__["load_midsep"]  # restored below anyway
    import importlib

    importlib.reload(correlate)
    correlate.main()


if __name__ == "__main__":
    main()
