"""Load residual-stream SAEs for the label-selected sparse reader.

Pinned SAEs (FEASIBILITY/manifests):
  Llama-3.1-8B : Goodfire/Llama-3.1-8B-Instruct-SAE-l19   (resid layer 19, L0~91, .pth)
  Llama-3.3-70B: Goodfire/Llama-3.3-70B-Instruct-SAE-l50  (resid layer 50, .pth)
  gemma-2-9b   : google/gemma-scope-9b-pt-res             (JumpReLU, params.safetensors)
  Gemma-3-12B  : google/gemma-scope-2-12b-it              (JumpReLU, params.safetensors)

Two formats, one interface: SAEReader.encode(h[S,d]) -> latent activations [S, L].
Goodfire .pth is a torch state dict (keys inferred by shape). Gemma Scope stores
{W_enc, b_enc, W_dec, b_dec, threshold} with JumpReLU: act = relu(pre) * (pre > threshold).

NOTE (namespace-transfer trap, learned in the llm_selfref series): the SAE must be trained
on the SAME model whose activations we read (instruct vs base matters). All pins above are
model-matched; EleutherAI/sae-llama-3.1-8b-* was rejected (base-model, MLP-out hooks).
"""
from __future__ import annotations

import torch


class SAEReader:
    def __init__(self, W_enc, b_enc, threshold=None, kind="relu", meta=None):
        # W_enc: [L, d] (latents x model-dim); b_enc: [L]
        self.W_enc = W_enc
        self.b_enc = b_enc
        self.threshold = threshold
        self.kind = kind
        self.meta = meta or {}

    @property
    def n_latents(self):
        return self.W_enc.shape[0]

    @torch.no_grad()
    def encode(self, h):
        """h: [S, d] float -> [S, L] latent activations (on h's device)."""
        W = self.W_enc.to(h.device, h.dtype)
        b = self.b_enc.to(h.device, h.dtype) if self.b_enc is not None else 0
        pre = h @ W.T + b
        if self.kind == "jumprelu":
            thr = self.threshold.to(h.device, h.dtype)
            return torch.relu(pre) * (pre > thr)
        return torch.relu(pre)


def load_goodfire(pth_path, d_model):
    """Goodfire .pth state dict; infer encoder by shape [L, d] with L > d."""
    sd = torch.load(pth_path, map_location="cpu", weights_only=True)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    W_enc = b_enc = None
    for k, v in sd.items():
        if v.ndim == 2 and v.shape[1] == d_model and v.shape[0] > d_model and (
                "enc" in k.lower() or W_enc is None):
            if "dec" in k.lower():
                continue
            W_enc = v.float()
            bkey = k.replace("weight", "bias")
            if bkey in sd and sd[bkey].ndim == 1 and sd[bkey].shape[0] == v.shape[0]:
                b_enc = sd[bkey].float()
    if W_enc is None:  # maybe stored transposed [d, L]
        for k, v in sd.items():
            if v.ndim == 2 and v.shape[0] == d_model and v.shape[1] > d_model and "dec" not in k.lower():
                W_enc = v.float().T
                bkey = k.replace("weight", "bias")
                if bkey in sd and sd[bkey].shape[0] == v.shape[1]:
                    b_enc = sd[bkey].float()
                break
    assert W_enc is not None, f"could not identify encoder in {list(sd.keys())[:8]}"
    return SAEReader(W_enc, b_enc, kind="relu",
                     meta={"format": "goodfire", "keys": list(sd.keys())[:8]})


def load_gemmascope(safetensors_path):
    """Gemma Scope params.safetensors: W_enc [d, L], b_enc [L], threshold [L] (JumpReLU)."""
    from safetensors.torch import load_file
    sd = load_file(safetensors_path)
    W_enc = sd["W_enc"].float()
    if W_enc.shape[0] < W_enc.shape[1]:      # [d, L] -> [L, d]
        W_enc = W_enc.T
    return SAEReader(W_enc, sd.get("b_enc", None),
                     threshold=sd.get("threshold", None), kind="jumprelu",
                     meta={"format": "gemmascope", "keys": list(sd.keys())})


def select_latent(sae, spans_hidden, labels, auroc_fn, batch=2048):
    """Label-selected sparse reader (Pro-review B05: this is SUPERVISED selection).
    On the VALIDATION fold only: score = mean latent activation over the span; pick the
    latent (and sign) with max AUROC. Returns (latent_idx, sign, val_auroc).
    The procedure-matched null (B08) must run this SAME selection on permuted labels."""
    import numpy as np
    y = torch.as_tensor(labels)
    pooled = torch.stack([sae.encode(h).mean(0) for h in spans_hidden])  # [N, L]
    best = (None, 1, -1.0)
    for lo in range(0, sae.n_latents, batch):
        block = pooled[:, lo:lo + batch].numpy()
        for j in range(block.shape[1]):
            a = auroc_fn(block[:, j], y.numpy())
            if a != a:
                continue
            s, aa = (1, a) if a >= 0.5 else (-1, 1 - a)
            if aa > best[2]:
                best = (lo + j, s, aa)
    return best
