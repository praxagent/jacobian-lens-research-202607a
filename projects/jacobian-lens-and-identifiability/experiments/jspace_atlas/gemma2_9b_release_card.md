---
license: gemma
base_model: google/gemma-2-9b
tags:
- interpretability
- jacobian-lens
- jspace
- representation-geometry
library_name: jlens
---

# Jacobian lens for google/gemma-2-9b

An independently-fitted **Jacobian lens (J-space)** for
[`google/gemma-2-9b`](https://huggingface.co/google/gemma-2-9b), released by
[praxagent](https://praxagent.ai). The public
[Neuronpedia lens collection](https://huggingface.co/neuronpedia/jacobian-lens) covers
many Gemma models but not gemma-2-9b; this fills that gap, and was fit specifically to
answer a question about the model's geometry (below).

A Jacobian lens stores, per layer \(\ell\), the matrix \(J_\ell = \mathbb{E}[\partial
h_\text{final} / \partial h_\ell]\): the model's own linear map from the residual stream
at layer \(\ell\) to its final residual stream, averaged over natural text. It reads out
what an internal activation is disposed to make the model say. See Anthropic's
*Verbalizable Representations Form a Global Workspace in Language Models*
(transformer-circuits.pub, 2026) for the method.

## What this lens establishes

gemma-2-9b has an unusually **layer-invariant** readout geometry: measured on a
shared-vocabulary probe, the centered-kernel-alignment (CKA) between any two layers'
token geometries has a median of **0.993** (minimum 0.975), i.e. every layer's
downstream effect points in nearly the same vocabulary directions. Its band-separation
statistic is 0.006, near zero.

This lens was fit to test whether that flatness is a real property of the model or an
artifact of a particular public fit. It is the model: an *independent* fit (this one,
different recipe) reproduces the near-layer-invariance, and the flatness survives a
shared-vocabulary probe where the sibling gemma-2-27b (fit the same way) shows normal
depth structure. The included `gemma2_9b_cka.npz` is the layer x layer CKA map. Full
context: the praxagent "J-Space Atlas" campaign write-up.

## Contents

| File | What |
|---|---|
| `gemma2_9b_jlens.pt` | the lens: `{"J": {layer: [d,d] tensor}, "source_layers", "d_model", "n_prompts"}`, 41 source layers, d_model 3584, fp16 |
| `gemma2_9b_cka.npz` | layer x layer CKA map of the lens's token geometry (`cka`, `layers`) |

## Fit details (reproducible)

- Library: Anthropic `jlens` (`jlens.fit`), commit `581d398`.
- Corpus: 100 WikiText-103 passages, seed 0, truncated to 128 tokens.
- `dim_batch=8`, running-mean estimator, bf16 model on GPU.
- Consumer-verified: recomputing the CKA map from these lens bytes plus gemma-2-9b's own
  unembedding reproduces the released map to a maximum element difference of 1.7e-5.

## Loading

```python
import torch
d = torch.load("gemma2_9b_jlens.pt", map_location="cpu", weights_only=False)
J = d["J"]  # {layer_index: [d_model, d_model] tensor}
```

## License and attribution

This lens is a **derivative** of Google's Gemma model `google/gemma-2-9b` and is
distributed under, and subject to, the **[Gemma Terms of
Use](https://ai.google.dev/gemma/terms)** and the [Gemma Prohibited Use
Policy](https://ai.google.dev/gemma/prohibited_use_policy), which flow through to this
artifact. "Gemma" is a trademark of Google LLC. The fitting code and the derived analysis
(the CKA map and statistics) are praxagent's own work, released under Apache-2.0. This
release follows the established practice of publicly distributing Gemma-derived Jacobian
lenses (e.g. the Neuronpedia collection). It is a non-peer-reviewed research artifact;
verify numbers against the lens bytes before relying on them.
