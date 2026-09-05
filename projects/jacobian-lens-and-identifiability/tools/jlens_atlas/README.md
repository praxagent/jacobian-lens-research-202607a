# jlens_atlas

The layer x layer CKA map recipe behind praxagent's Jacobian-lens atlas, packaged so that anyone
with a Jacobian lens can run it on their own lens and compare against the 36-lens zoo. CPU only.
It never runs the model and never downloads the full model: it reads only the rows it needs from
the (un)embedding tensor.

Licence: Apache-2.0. Vendored, with attribution on every function, from the campaign repository
`jacobian-lens-research-202607a` (the code that produced the published numbers), so this file can
be checked line by line against the experiments.

## What it computes

For a lens with Jacobians `J_l` (one per source layer, shape `d_final x d_layer`) and an
unembedding matrix `U`:

1. **Readout geometry** `D_l = U_probe J_l`: one row per probe token, the direction at layer `l`
   that most increases that token's logit.
2. **Layer x layer linear CKA** of the geometries (column-centered; invariant to rotations and to
   isotropic scaling; not to per-dimension scaling).
3. **Fixed-thirds band statistic** `mid_sep`: mean within-middle-third CKA minus the mean of the
   early-mid and mid-late block means, diagonal excluded. This is the number the 397B release
   reported (+0.343) and the zoo is ranked by.
4. **Fitted three-segmentation** `(b1, b2)` and `fitted_sep`: the two boundaries that maximise the
   sum over blocks of mean within-block CKA; `fitted_sep` is within-mean minus between-mean there.
5. **Boundary identifiability**: the spread of each boundary over every segmentation whose
   objective is within 5% of the objective's range of the optimum, as a fraction of depth. A pair
   is *identified* when both spreads are at most 0.25. Read the caveat below.
6. **Participation ratio** per layer (a soft count of the readout's effective dimension).
7. **Random-transport null**: the same pipeline with each `J_l` replaced by a random matrix of
   matched Frobenius norm (`--null std` gives the std-matched variant used by the atlas's
   participation-ratio null). Its `mid_sep` should sit near zero; if it does not, the structure
   you see is coming from the unembedding, not the lens.

Outputs in `--out`: `cka.npz` (map, null map, layers, statistics, probe metadata),
`summary.json`, `receipt.json` (lens sha256 or HF revision, model id, probe kind/seed/size,
package and library versions, timestamp, command line, the exact definitions used) and, when
matplotlib is installed, `heatmap.png` / `heatmap.svg`.

## Commands

```bash
# a lens from the public Neuronpedia collection (model id is read from the lens's config.yaml)
python -m jlens_atlas --neuronpedia gpt2-small --out out/gpt2-small

# your own jlens checkpoint
python -m jlens_atlas --lens fits/my_lens.pt --model Qwen/Qwen3-4B --out out/qwen3-4b

# a lens hosted in any HF repo
python -m jlens_atlas --hf praxagent-org/jacobian-lens-gemma-2-9b --file jlens/wikitext/gemma2_9b.pt \
    --model google/gemma-2-9b --out out/gemma-2-9b

# the model's own vocabulary instead of the shared probe (see the warning below)
python -m jlens_atlas --neuronpedia gemma-3-12b --probe own --out out/gemma-3-12b-own
```

Requirements: `numpy`, `torch`, `huggingface_hub`, `safetensors`, `transformers` (tokenizer only,
for the shared probe), `pyyaml` (Neuronpedia config); `matplotlib` optional. Gated models need
`HF_TOKEN` in the environment. Install with `pip install -e .` from this directory or just run
`python -m jlens_atlas` from here.

Lens format: a `torch.load`-able dict with `"J": {layer_index: tensor}` as written by Anthropic's
`jlens` library; `d_model`, `source_layers` and `n_prompts` are read if present.

## The shared probe, and why the default is not the model's own vocabulary

`jlens_atlas/data/shared_tokens.json` holds 4,096 token **strings** that resolve to a single
token in every tokenizer of the 37-model zoo (built by
`experiments/jacobian_lens/shared_vocab.py`: intersect each tokenizer's single-token, printable,
letter-containing strings of length 2 to 24; shuffle with seed 0; keep 4,096). At run time each
string is mapped to this model's token id, space-prefixed form first, bare form as fallback;
strings that need more than one token in your tokenizer are dropped and counted in the receipt.

**Own-vocabulary CKA is a valid question within one model and the wrong instrument across
models.** Sampling 4,096 rows from each model's own vocabulary changes which tokens each model is
probed on, and that alone reshapes the map: in the zoo, Gemma lenses that look featureless on
their own vocabulary recover ordinary depth structure on the shared probe (gemma-3-12b goes from
`mid_sep` 0.001 to 0.114). The shared probe removes the token-set confound. It does not by
itself equalise model-specific baselines, because each model still has its own unembedding, so
compare rankings, not raw values, across models.

## The identifiability caveat

The fitted boundaries are the argmax of an objective that is nearly flat for some lenses. When
the near-optimal set spans a third of the depth, "the boundary" is a near-arbitrary pick and
should not be used as a variable in a downstream test. In the zoo about a third of the lenses
fail the 0.25 threshold on the shared probe; the 397B lens passes comfortably (spreads 0.08 and
0.07). The tool prints the spreads and the verdict every time; treat `NOT identified` as
"report the map and `mid_sep`, not the boundaries".

## How to compare your lens to the zoo

Run with the default shared probe and compare `mid_sep` and `fitted_sep` against
`experiments/jspace_atlas/atlas_out/shared_summary.csv` (columns `slug, family, n_shared,
mid_sep, fitted_sep`) and the per-model maps in `experiments/jspace_atlas/atlas_out/shared_maps/`.
Keep the fitting corpus in mind: the zoo is WikiText-fitted, and the campaign's corpus experiment
found that fitting on code moves the map by two to three orders of magnitude more than a WikiText
resample does (while barely moving the fitted boundaries), so a lens fitted on a different
distribution is not directly comparable on band strength. Fitting budget matters much less: from
about 200 prompts up the map is inside resampling noise.

## Reproducing the cached maps exactly

The cached atlas maps stored each geometry `D_l` in float16 before computing CKA in float32. The
default here keeps `D_l` in float32, which differs from the cache by at most 1.7e-6 on
gpt2-small (measured by `selftest.py`). `--geometry-dtype fp16` mirrors the cache's rounding and
reproduces the cached gpt2-small maps exactly (maximum absolute difference 0.0 on both probes);
`selftest.py` asserts this and prints the measured fp32 distance. Run it with the repository venv:

```bash
../../../../.venv/bin/python selftest.py
```

## Provenance of the recipe

`core.py` names, per function, the experiment file it was vendored from: `common/cka.py`
(linear CKA), `experiments/jspace_atlas/atlas_stage_a.py` (own probe, fitted segmentation,
participation ratio), `experiments/jacobian_lens/cka_layers.py` (band statistic),
`experiments/jspace_atlas/boundary_identifiability.py` (near-optimal spread),
`experiments/fit_our_own/cka_heatmap_397b.py` (Frobenius-matched null). `io.py` follows
`experiments/jacobian_lens/unembed.py` for locating the unembedding tensor (preferring
`lm_head.weight` when the readout is untied) and
`experiments/geometry_causality/run_geometry_causality.py::probe_rows_inline` for reading only
the needed rows.
