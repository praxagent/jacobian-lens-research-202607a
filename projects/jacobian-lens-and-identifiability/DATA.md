# Data — what's committed, what's regenerable, how to get it

Everything needed to reproduce or audit our numbers. Committed data is the receipt for
every claim in the writeup/blog.

## Committed to git (the receipts — small, permanent)

| Path | What |
|---|---|
| `experiments/jacobian_lens/emergence.csv` | Per-model band statistic (`mid_sep` + block means), own-vocab probes, uniform final code path. The core result. |
| `experiments/jacobian_lens/emergence_null.csv` | Random-transport null controls (confound floor). |
| `experiments/jacobian_lens/emergence_shared.csv` | Shared-vocabulary (cross-tokenizer) probe re-sweep — the tokenizer-confound mitigation. |
| `experiments/jacobian_lens/emergence_fp32path.csv` | The original fp32-path ledger, kept as the precision A/B reference (see `ab_report.txt`). |
| `experiments/jacobian_lens/ab_report.txt` | fp32-path vs final-path A/B: max/mean |Δ mid_sep| across models. |
| `experiments/fit_our_own/results.md` | Our own-fit results: gpt2 validation gate (CKA 0.9992), seed stability (0.997–0.998), language dependence (zh↔en). |
| `experiments/*/results.md` | Per-experiment ledgers — only runs actually executed. |
| `experiments/behavioral/verbal-report.json`, `ignition.json` | Anthropic's released prompt sets (Apache-2.0), vendored so the behavioral tests are self-contained + shareable. See `ANTHROPIC_EXPERIMENTS_README.md`. |
| `experiments/behavioral/verbal_report_*.json` | Behavioral causal-swap results per model. |

CSV schema (`emergence*.csv`): `slug, hf_id, params, d_model, n_layers, mid_sep,
within_mid, within_early, within_late`. Higher `mid_sep` = more distinct workspace band.

## NOT in git (regenerable or too large)

- **Pre-fitted lenses** (the 38 models): from
  [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens) on HF —
  free, authoritative (official Anthropic×Neuronpedia release). `cka_layers.py` downloads
  the needed file per model automatically.
- **Our own fitted lenses** (`artifacts/lenses/`: gpt2, qwen3-4b seeds 0/1/2, qwen3-4b-zh
  — ~1.8 GB): gitignored (too big for git). Regenerate cheaply with
  `experiments/fit_our_own/fit_lens.py` (recipe + exact commands in that dir's README).
  The Chinese lens (`qwen4b_zh.pt`) is the one genuinely novel artifact others might want
  — **TODO: offer via HF or Git LFS if requested** (currently preserved on the dev box).
- **HuggingFace model cache**: re-downloads free.

## Reproduce from scratch

```bash
uv venv && uv pip install -e .            # + torch (CPU wheel, see infra.md)
cd projects/jacobian-lens-and-identifiability/experiments/jacobian_lens
python ladder.py                          # full 38-model own-vocab sweep -> emergence.csv
python shared_vocab.py --out shared_tokens.json && \
  for s in $(tail -n+2 emergence.csv|cut -d, -f1); do python cka_layers.py --slug $s --shared-probe shared_tokens.json; done
python plot_curve.py                      # figure
```
Own-vocab + shared sweeps are CPU-only (needs ~17 GB RAM for the 70B); behavioral +
fitting need a GPU (see `../fit_our_own/README.md`, `../behavioral/`).
