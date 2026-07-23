# Activation-CKA: is Gemma's flat lens a model property or a lens property?

**Targets the open "mechanism" question** left by the Atlas note (why is the gemma-2
small-model *lens* near-perfectly layer-invariant?). The model-vs-fit question was already
settled (Tier-1 Arm A: an independent fit reproduces the flatness, so it is the model). This
asks a different question: does the flatness live in the **residual stream itself** (a genuine
"one-geometry" network, theory 1) or only in the **first-order output linearization** the
Jacobian lens computes (theory 2)?

## Method

`activation_cka.py` runs a forward pass over 80 WikiText-103 passages (>=300 chars, truncated
to 128 tokens), captures the RAW hidden states at every layer (embedding layer dropped),
subsamples 4096 token positions (seed 0), and computes the layer x layer linear CKA of the
activations themselves. No lens, no fitting, inference only. Compared against the
shared-vocabulary **lens** CKA from the atlas (`atlas_out/shared_maps/*.npz`) using the same
fixed-index-thirds band-separation statistic. Two flat-lens models (gemma-2-9b, gemma-2-2b) vs
a structured-lens control (qwen3-4b, whose lens and activations should both be structured).

## Run ledger (real runs only)

- **2026-07-23, RTX 3090 (pod rk75b1v5m48hx7, $0.22/hr), ~$0.15, terminated after receipt
  validation.** Repo @ `8ae9829`, torch 2.4.1+cu124, transformers 4.57.6. Receipts +
  layer x layer CKA maps in `actcka_out/{gemma2_9b,gemma2_2b,qwen3_4b}.{json,npz}`.
  - A first pod (`0uj4nehogal7i1`) died mid-run on a broken CUDA context (nvidia-smi saw the
    GPU, `torch.cuda.is_available()` was False even in a fresh process); terminated and
    relaunched. Gemma repos are gated: `HF_TOKEN` exported into the pod shell via `bash -s`
    stdin (never a file, not in `ps` args).

| model | activation band-sep | activation min CKA | activation median | lens band-sep | lens min CKA |
|---|---|---|---|---|---|
| gemma-2-9b | **+0.110** | **0.078** | 0.931 | +0.005 | 0.975 |
| gemma-2-2b | **+0.081** | **0.068** | 0.965 | +0.007 | 0.971 |
| qwen3-4b (control) | +0.264 | 0.028 | 0.927 | +0.038 | 0.843 |

## Result (theory 2 dominant, small theory-1 flavor)

The two flat-lens Gemmas have **structured raw activations**: near-orthogonal layer pairs
(min CKA ~0.07) and clear positive band separation (+0.08 to +0.11). If they were literal
one-geometry networks, the activations would be flat too. They are not. So the near-perfect
lens flatness (band-sep ~0.005, min 0.975 -- no two layers' readouts diverge more than 2.5%)
is dominated by the **first-order output linearization**: every layer's marginal push on the
final logits points in nearly the same vocabulary directions, even though the activations
being differentiated evolve normally with depth.

The control validates the method: qwen3-4b's activations are the most structured of the three
(band-sep +0.264), matching its structured lens -- so activation-CKA reliably surfaces depth
structure when it exists, and the Gemma reading is not an artifact of the measurement.

Honest nuance: (a) a lens is *always* a flattened view -- even qwen's lens (+0.038) is flatter
than its activations (+0.264); the Gemma effect is that its lens flatness is extreme (min 0.975
vs qwen 0.843). (b) Gemma's *activations* are modestly flatter than qwen's (band-sep 2-3x
smaller), so there is a small genuine activation-level homogeneity riding underneath -- but it
is an order of magnitude weaker than the lens flatness it produces. The tidy "Gemma is a
one-geometry network" reading is wrong; the accurate reading is "a modest real activation
homogeneity, massively amplified by a near-layer-invariant first-order readout." **Why** the
gemma-2 small-model recipe produces that readout remains open.

## Reproduce

```bash
# on a small GPU (3090 fine); gemma repos need HF_TOKEN in the env
python activation_cka.py --model google/gemma-2-9b --tag gemma2_9b_flat  --out out/gemma2_9b
python activation_cka.py --model google/gemma-2-2b --tag gemma2_2b_flat  --out out/gemma2_2b
python activation_cka.py --model Qwen/Qwen3-4B     --tag qwen3_4b_struct --out out/qwen3_4b
# figure (activation vs lens, 3x2): figures/build_actcka_figs.py  (--verify to check svg)
```
