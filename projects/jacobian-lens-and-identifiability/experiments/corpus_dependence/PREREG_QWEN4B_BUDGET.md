# Pre-registration: does the whole map converge as early as the band statistic? (qwen3-4b)

Frozen 2026-09-08 before any fit exists.

## Why

The 397B release note justifies its 24-prompt fit with a July 2026 ablation of `mid_sep` on
qwen3-4b: the band statistic sits at its reference from n = 16 on (0.060 / 0.050 / 0.058 / 0.060 /
0.061 at 16 / 32 / 64 / 128 / 256; 0.036 at 8; `fit_our_own/results.md` section 4). The fit-budget
sweep (`PREREG_FITBUDGET.md`) later measured the **whole layer-by-layer map** on gpt2-small and
gemma-3-270m and found the 25-prompt map 1.8x and 14x their seed nulls. Those are two different
estimands measured on different models. This test measures both on the same model and the same
fits, at the budgets the release note leans on (16 and 24), and asks whether "band statistic
converged" implies "map converged". It is a **proxy** for the open 397B question at about $2, not
an answer to it: qwen3-4b is dense and two orders of magnitude smaller.

## Design

Model `Qwen/Qwen3-4B` (in the zoo: public Neuronpedia lens, 35 lens layers, d = 2560, shared-probe
boundaries 3 / 23, **identified**; registry shared_mid_sep 0.0383, shared_fitted_sep 0.0567; tied
embeddings, so the probe rows are the embedding rows on every loader). Six fits with
`fit_our_own/fit_lens.py`, `--max-seq-len 128 --match-length --dim-batch 8`, bf16, WikiText-103:

| fit | prompts | seed | role |
|---|---|---|---|
| `q4b_wiki_a` | 100 | 0 | reference (the fit-budget sweep's reference budget) |
| `q4b_wiki_b` | 100 | 1 | seed null |
| `q4b_n8`, `q4b_n16`, `q4b_n24`, `q4b_n48` | 8, 16, 24, 48 | 0 | budgets; nested prefixes of `wiki_a`, as in the fit-budget sweep |

Measures, all on the shared-vocabulary probe with the corrected statistic (`analyze.py` helpers,
identity-checked against `common.cka.linear_cka`), by `analyze_qwen4b_budget.py`:

- **map distance** to the reference (`1 - CKA` between off-diagonal map profiles);
- **fitted boundaries** and **fitted band separation** (`atlas_stage_a.fitted_seg`);
- **thirds `mid_sep`** (the statistic the release note quotes);
- anchor distance of every fit to the public shared map (descriptive: the reference scale here is
  our own seed null, and a 100-prompt 4B fit may sit outside the 0.01 that sub-1B fits met, as the
  8B did at 0.011).

**Frozen bars.** Map converged at budget n iff map distance(n, reference) <= 2x the seed-null map
distance (the fit-budget sweep's bar, unchanged). Band converged at n iff both
|mid_sep(n) - mid_sep(ref)| and |fitted_sep(n) - fitted_sep(ref)| are <= 2x their seed-null shifts,
with a floor of 0.002 on each seed shift so that a degenerate seed null cannot make the bar
unpassable. No other bars.

## Predictions

- **P1** (the July result replicates on the shared probe): band converged at 16, 24 and 48; not at 8.
- **P2** (the map lags the band): map **not** converged at 16 and 24, converged at 48. This is what the
  small-model sweep suggests; a miss in either direction is informative and is reported as such.
- Anchor: the reference fit lands within 0.01 of the public map. Descriptive; not a gate.

## Decision table

| outcome | statement |
|---|---|
| band not converged at 16 or 24 | P1 fails: the July band result does not replicate under this recipe; reported as such, and the release-note addendum says so |
| map converged at 24 | on qwen3-4b the whole map is inside the seed null by 24 prompts; the release note's parenthetical extends to the map on this model; the 397B caveat is weakened, not discharged |
| map not converged at 24, converged at 48 | the band converges before the map: 24 prompts is band-converged and not map-converged on this model; the atlas note's caveat on the 24-prompt 397B map stands and the addendum says so |
| map not converged at 48 | the map is not converged by 48 prompts on a 4B model; the caveat is strengthened and the addendum says so |

The 397B statement in either note changes only in the direction the table names; no 397B claim is
made from a 4B proxy.

## Cost

296 prompts of qwen3-4b fitting. Scaling the measured 8B throughput (117 s per prompt on an RTX A6000)
by parameters x d_model gives roughly 35 to 45 s per prompt, so about 3 to 4 h plus setup on one
RTX A6000 at $0.33 to $0.53 per hour: **about $1.50 to $2.50.** `DONE` marker, sha256 receipt with
all-finite and layer-count checks, scheduled fetch-verify-terminate from the work box.
