# fit_our_own — fitting our own Jacobian lenses (tier 1: validate + stability)

**Goal:** stop depending only on Neuronpedia's pre-fitted lenses — fit our own with
Anthropic's `jlens.fit`, to (1) **prove our pipeline is faithful** and (2) **test whether
J-space is stable** across fitting seeds/corpora. Closes the biggest limitation in the
emergence audit (one lens per model → can't test stability). Frontier (>70B) fitting
became part of this experiment too — the 397B fit lives in `results.md` §5–6 with
`fit_at_scale.py` / `tp_fit.py` / `GAMEPLAN-397B.md` / `MODEL_CARD-397B.md`.

**Compute:** GPU. Fitting is ~`ceil(d_model/8)` backward passes per prompt over ~100
short prompts. Trivial for gpt2 (CPU-feasible), needs a GPU for ≥1B.

## The two runs (one RunPod pod does both)

1. **Validation (the gate) — gpt2.** Fit our own gpt2 lens, CKA it vs Neuronpedia's:
   ```bash
   uv run python fit_lens.py --model openai-community/gpt2 --n-prompts 100 --seed 0 --out lenses/gpt2_ours.pt
   uv run python compare.py --model openai-community/gpt2 \
       --a lenses/gpt2_ours.pt \
       --b "hf:neuronpedia/jacobian-lens::gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"
   ```
   **Expect mean CKA ≳ 0.9.** If not, our fitter is wrong — stop and fix before spending on bigger fits.

2. **Stability — qwen3-4b, 3 seeds.** Fit the same model on 3 different corpus subsets;
   compare pairwise:
   ```bash
   for s in 0 1 2; do uv run python fit_lens.py --model Qwen/Qwen3-4B --n-prompts 100 --seed $s --out lenses/qwen4b_seed$s.pt; done
   uv run python compare.py --model Qwen/Qwen3-4B --a lenses/qwen4b_seed0.pt --b lenses/qwen4b_seed1.pt
   uv run python compare.py --model Qwen/Qwen3-4B --a lenses/qwen4b_seed0.pt --b lenses/qwen4b_seed2.pt
   ```
   High cross-seed CKA = J-space is a stable property (strengthens every emergence claim).
   Low = it's fitting-dependent (a real, honest limitation to report).

## Cost estimate

Single **A100-80GB** on RunPod (~$1.5–2/hr): gpt2 validation ≈ seconds; qwen3-4b × 3 seeds
≈ 4–8 h. **Total ≈ $8–20**, terminated on completion. (Drop to qwen3-1.7b or 50 prompts to
roughly halve it.)

## Deps on the pod

`jlens`, `transformers`, `torch` (CUDA), plus **`datasets`** (for the wikitext corpus) and
`numpy`/`scipy`. The RunPod launcher (`shared/runpod/`) clones this repo and installs
`.[torch]` + these.

## Status

**RUN (complete)** — see [`results.md`](results.md): gpt2 validation gate PASSED (CKA
0.9992 vs Neuronpedia), qwen3-4b 3-seed stability (0.997–0.998), Chinese-corpus lens,
n-scaling convergence curve, and the **397B frontier fit** (mid_sep +0.343 at n=24; lens
published at `praxagent/jacobian-lens-qwen3.5-397b-a17b` with eval receipts +
`consumer_check_397b.py` independent-verification script).
