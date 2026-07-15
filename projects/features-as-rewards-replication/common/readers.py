"""Four readers of a hallucination-belief signal over an entity span, plus AUROC.

Each reader maps the residual-stream activations of an entity span to ONE scalar
"how hallucinated" score. Higher = more likely hallucinated. We then compute AUROC
against gold labels. The four readers differ only in HOW they turn activations into
that scalar:

  attention_probe : SUPERVISED. Trained on gold labels (the paper's instrument).
  logit_lens      : UNSUPERVISED. Surprisal of the emitted entity token, read through
                    the model's own unembed at a chosen layer (identity transport).
                    This is the "model's calibrated belief" readout and the mandated
                    cheapest-prior-method baseline (integrity playbook).
  jlens           : UNSUPERVISED. Same surprisal, but the hidden state is first sent
                    through a fitted Jacobian transport J before the unembed.
  sae_latent      : SPARSE/UNSUPERVISED. Activation of a chosen SAE latent (e.g. an
                    "unsupported-claim / uncertainty" feature) pooled over the span.

No sklearn dependency: AUROC is the rank-based Mann-Whitney estimator.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# metric
# --------------------------------------------------------------------------- #
def auroc(scores, labels) -> float:
    """AUROC = P(score[pos] > score[neg]); ties count half. labels in {0,1}."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = {i: (csum[i] - counts[i] + csum[i] + 1) / 2.0 for i in range(len(counts))}
    ranks = np.array([avg[i] for i in inv])
    r_pos = ranks[labels == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def directed_auroc(scores, labels) -> float:
    """AUROC with the sign chosen so it is >= 0.5 (unsupervised readers have no a-priori
    sign; the direction is a single bit fixed on the train fold in real runs). Returns
    the max of a and 1-a so a null reader lands near 0.5, a signal reader above it."""
    a = auroc(scores, labels)
    return a if np.isnan(a) else max(a, 1.0 - a)


# --------------------------------------------------------------------------- #
# unsupervised readers (label-free, deterministic)
# --------------------------------------------------------------------------- #
def _lens_surprisal(hidden, unembed, final_norm, span_ids, transport=None, pool="mean"):
    """Surprisal (-log p) of each emitted entity token, read from `hidden` (a
    [span_len, d] residual-stream slice at one layer) through an optional linear
    transport then the model's unembed. Pool over the span.

    hidden:      [S, d] residual stream at the entity token positions, one layer
    unembed:     [V, d] (lm_head weight)
    final_norm:  callable applying the model's final norm to [S, d] (or None)
    span_ids:    [S] the actual token ids emitted at those positions
    transport:   [d, d] fitted J (jlens) or None (logit lens = identity)
    """
    h = hidden
    if transport is not None:
        h = h @ transport.T
    if final_norm is not None:
        h = final_norm(h)
    logits = h @ unembed.T                       # [S, V]
    logp = F.log_softmax(logits.float(), dim=-1)  # [S, V]
    tok_logp = logp[torch.arange(h.shape[0]), span_ids]  # [S]
    surprisal = -tok_logp                        # higher = model less confident
    return surprisal.mean().item() if pool == "mean" else surprisal.max().item()


def logit_lens_score(hidden_layers, unembed, final_norm, span_slice, span_ids,
                     layer, pool="mean", head_layer=None):
    """NOTE (caught by gate G1): HF `hidden_states[-1]` is ALREADY post-final-norm, so
    when reading the head layer pass head_layer=layer (or final_norm=None) — re-applying
    a learned LayerNorm/RMSNorm distorts the native NLL."""
    h = hidden_layers[layer][span_slice]         # [S, d]
    fn = None if (head_layer is not None and layer == head_layer) else final_norm
    return _lens_surprisal(h, unembed, fn, span_ids, transport=None, pool=pool)


def jlens_score(hidden_layers, unembed, final_norm, span_slice, span_ids,
                layer, transport, pool="mean"):
    h = hidden_layers[layer][span_slice]
    return _lens_surprisal(h, unembed, final_norm, span_ids, transport=transport, pool=pool)


def sae_latent_score(hidden_layers, sae_encode, latent_idx, span_slice, layer,
                     pool="mean"):
    """Activation of one SAE latent over the span. sae_encode: [S,d] -> [S,n_latents]."""
    h = hidden_layers[layer][span_slice]         # [S, d]
    acts = sae_encode(h)                         # [S, L]
    col = acts[:, latent_idx]
    return col.mean().item() if pool == "mean" else col.max().item()


# --------------------------------------------------------------------------- #
# supervised reader: the paper's attention probe (classify a span)
# --------------------------------------------------------------------------- #
class AttentionProbe(nn.Module):
    """Low-expressivity noncausal attention probe: a learned query attends over the
    span's token activations to pool them, then a linear head predicts hallucinated.
    Mirrors the paper's App-B attention-probe family (single learned query, few heads)."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.query = nn.Parameter(torch.randn(n_heads, d_model // n_heads) * 0.02)
        self.head = nn.Linear(d_model, 1)

    def forward(self, span, mask=None):
        # span: [B, S, d]  mask: [B, S] (1=valid)
        B, S, d = span.shape
        H, dh = self.n_heads, d // self.n_heads
        keys = span.view(B, S, H, dh)
        scores = torch.einsum("bshd,hd->bsh", keys, self.query) / (dh ** 0.5)  # [B,S,H]
        if mask is not None:
            scores = scores.masked_fill(~mask.bool().unsqueeze(-1), float("-inf"))
        attn = torch.softmax(scores, dim=1)                    # over span positions
        pooled = torch.einsum("bsh,bshd->bhd", attn, keys).reshape(B, d)  # [B,d]
        return self.head(pooled).squeeze(-1)                   # [B] logit


def train_probe(spans, masks, labels, d_model, n_heads=4, epochs=200, lr=1e-2,
                seed=0, val=None):
    """Train the attention probe with BCE. spans:[N,S,d] masks:[N,S] labels:[N].
    Returns (probe, train_scores). Deterministic given seed."""
    torch.manual_seed(seed)
    probe = AttentionProbe(d_model, n_heads).to(spans.device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    y = torch.as_tensor(labels, dtype=torch.float32).to(spans.device)
    for _ in range(epochs):
        opt.zero_grad()
        logit = probe(spans, masks)
        loss = F.binary_cross_entropy_with_logits(logit, y)
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        scores = torch.sigmoid(probe(spans, masks)).cpu().numpy()
    return probe, scores
