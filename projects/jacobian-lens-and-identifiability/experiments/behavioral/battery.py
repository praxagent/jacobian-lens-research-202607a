"""Behavioral battery — do causal workspace effects track the geometric band?

Runs two of Anthropic's released causal experiments (anthropics/jacobian-lens
data/experiments/, Apache-2.0; conventions per their README) on open models:

  verbal-report            "Think of a {category}." -> greedy answer; SWAP the
                           answer's J-lens direction for a candidate's across the
                           band layers at every prompt position; success = the
                           candidate becomes the rank-1 next token.
  flexible-generalization  fill "The capital of {arg} is..." templates; swap
                           arg_a -> arg_b directions; success = next token now
                           matches arg_b's answer for that template.

The question for the killer triple (qwen3-4b / gemma-2-9b / gemma-2-27b): does
CAUSAL steerability track the geometric band (strong/absent/strong), or is the
band a geometric epiphenomenon? Either answer changes the blog post; we report
whichever we get.

Swap semantics (README: "clamping a lens coordinate replaces one token's
direction with another's at every band layer at the specified positions"):
at each band layer l, with unit residual-space directions a = J_l^T U[t_out],
b = J_l^T U[t_in]:   h <- h - (h.a)a + (h.a)b   (coefficient transfer), with an
optional additive kick of strength*mean_resid_norm along b (README's
verbal-introspection scaling). Applied at all prompt positions via forward
hooks; grading is a single forward pass (rank at the final position).

Run (GPU; CPU works for gpt2 smoke):
    python battery.py --slug gpt2-small --exp verbal-report --limit 4 --device cpu
    python battery.py --slug qwen3-4b --exp both --out results_qwen3-4b.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "jacobian_lens"))
from cka_layers import REPO, resolve  # noqa: E402

EXP_DIR_CANDIDATES = [