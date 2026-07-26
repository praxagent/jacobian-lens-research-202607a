"""Freeze the ten factual prompts, per PREREG.md, before any model sees them.

Each prompt must have an unambiguous single-token continuation in every tokenizer we use, so the
ignition target is the same object across models. Written to an artifact; never edited after.
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

PROMPTS = [
    ("The capital city of France is called", " Paris"),
    ("The capital city of Japan is called", " Tokyo"),
    ("The capital city of Italy is called", " Rome"),
    ("The largest planet in our solar system is called", " Jupiter"),
    ("The chemical symbol for water is H2", "O"),
    ("The first President of the United States was George", " Washington"),
    ("The language mainly spoken in Brazil is", " Portuguese"),
    ("The ocean between Europe and North America is the", " Atlantic"),
    ("The currency used in Japan is the", " yen"),
    ("The scientist who developed the theory of relativity was Albert", " Einstein"),
]

if __name__ == "__main__":
    art = {"n": len(PROMPTS),
           "note": "single-token answers verified per model at run time; a prompt whose answer "
                   "is not a single token in a given tokenizer is dropped FOR THAT MODEL and "
                   "the drop is recorded in that model's receipt",
           "prompts": [{"prompt": p, "answer": a} for p, a in PROMPTS]}
    (HERE / "prompts_frozen.json").write_text(json.dumps(art, indent=1))
    print(f"froze {len(PROMPTS)} prompts -> {HERE / 'prompts_frozen.json'}")
