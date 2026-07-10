# Results — J-lens (open-model audit)

_Ledger of actual runs. Only executed runs; nothing copied from a paper. This is the
evidence base for the "was the global-workspace framing oversold?" audit — read
[`../../background.md`](../../background.md) and the experiment
[`README.md`](README.md)._

## Sub-experiment A/B — within-model layer×layer CKA (`cka_layers.py`, CPU)

Method: J-lens token geometry `D_l = U @ J_l` over a 4096-token vocab sample; linear CKA
between layers. Pre-fitted Neuronpedia lenses, so no fitting/GPU.

### gpt2-small (2026-07-07, CPU, free)

Lens: `d_model=768, n_prompts=277, 11 source layers`.

**Result: uniformly high cross-layer CKA (0.94–1.00), NO distinct mid-network band.**
mid-band separation = **+0.015** (≈ 0). Within-early / within-mid / within-late block
means are all ~0.97–0.99; every layer's J-lens token geometry is nearly identical to
every other's.

**Null control (random scale-matched `J_l`, same script `--null`):** cross-layer CKA
drops to a **uniform ~0.66** (mid-sep +0.001). This addresses the shared-unembedding
confound: if `U` alone drove the similarity, the null would also sit at ~0.99 — it
doesn't. So the real lens's ~0.99 is a genuine property of the *fitted transports*, above
a 0.66 null floor. Read "high CKA" relative to that 0.66 floor, not to 0.

**Honest read (confound substantially controlled).** gpt2-small has **no workspace
band** — the real geometry is uniformly self-similar (0.99-flat), the null is uniformly
moderate (0.66-flat), and *neither* has a mid-network bump. The method would surface a
band if one existed (it resolves the real vs null gap cleanly), so gpt2's absence is
real, not an artifact. What's still needed is **calibration by scale**: eliebak reports
a sensory→workspace→motor band on *larger* models with this readout, so the live question
is whether a mid-band separation **emerges above this flat baseline as models grow**.

**Status of the audit:** one small model, no band, confound controlled. Not yet
"oversold" evidence — but the setup is now clean enough that the **size-ladder emergence
curve** (next) is decisive: if `mid_sep` stays ≈0 up through mid/large models, the
"universal workspace band" framing is overstated; if it rises sharply at some scale, the
claim holds (as a large-model phenomenon) and we say so.

_Pipeline note: this ran on a 3 GB box because the loader pulls only the unembedding
tensor, not the full model — so the ladder is memory-bound (needs a bigger box), not
GPU-bound._

## Final uniform sweep + precision A/B + SHARED-VOCAB re-sweep (2026-07-10, Lightsail CPU)

The one-code-version, one-precision-path re-run of everything (the methods decision of
2026-07-08), plus the tokenizer-confound mitigation. Receipts: `emergence.csv` (35
models, own-vocab), `emergence_shared.csv` (35 models, shared probes), `ab_report.txt`,
`emergence_curve.png` (35 models, 12 families), full pipeline log archived locally
(`artifacts/lightsail-receipts/uniform.log`, gitignored).

**Coverage: 35 of Neuronpedia's 38 lenses.** Not measured: `qwen3.6-27b` (lens failed to
load), `qwen3-32b` (its lens stores keys in a layout our loader doesn't parse yet — open
item), `gemma-4-31b` (unembedding requires a ~50 GB model pull; exceeds the 30 GB CPU
box). llama3.3-70b-it (own 0.1478 / shared 0.1372) is the new 70B anchor.

**Precision A/B: a clean zero.** fp32-path ledger vs the final fp16-storage/fp32-compute
path, 34 overlapping models: **max |Δ mid_sep| = 0.00000** (mean 0.000000). The mid-run
precision change had no effect on any published number.

### ⚠️ The tokenizer confound was REAL — own-vocab probing understated Gemma

Re-running every model with probes restricted to the 4096 token strings shared by ALL
tokenizers in the study (`shared_vocab.py` → `--shared-probe`) moves Gemma dramatically
while barely moving anyone else:

| model | own-vocab | shared-probe | Δ |
|---|---|---|---|
| gemma-3-27b | 0.025 | **0.298** | +0.273 — now the strongest band in the sweep |
| gemma-3-270m | 0.010 | **0.134** | +0.124 — a solid band at 0.27B(!) |
| gemma-3-12b | 0.0007 | **0.114** | +0.113 |
| gemma-2-27b | 0.043 | **0.113** | +0.071 |
| gemma-4-e4b | 0.050 | 0.098 | +0.048 |
| qwen3-14b | 0.211 | 0.206 | −0.005 |
| qwen3.5-27b | 0.197 | 0.196 | −0.000 |
| pythia-70m / gpt2 | 0.015 / 0.015 | 0.016 / 0.015 | ~0 (floor unchanged) |

Full table in `emergence_shared.csv`; mean Δ across 35 models +0.024, max +0.273
(gemma-3-27b).

**Family-gap verdict: the headline mostly does NOT survive.** Base-model family means,
own-vocab: Qwen 0.152 vs Gemma 0.024 — a **6.2×** gap. Shared probes: Qwen 0.139 vs
Gemma 0.099 — **1.4×**; excluding the two KD-pretrained Gemma-2 bases (2B/9B, which stay
at the floor: 0.0074/0.0047), **1.14× — essentially nothing**. At matched size the
ordering even flips: gemma-3-27b 0.298 > qwen3.5-27b 0.196. The "~300× at 12–14B" own-
vocab contrast (qwen3-14b 0.21 vs gemma-3-12b 0.0007) becomes **1.8×** (0.206 vs 0.114).
Gemma's own 262k-token vocabulary was diluting its probe set; probed on common strings,
Gemma has bands too.

**What SURVIVES shared probes (and strengthens):**
1. **The within-Gemma-2 KD natural experiment.** KD-pretrained 2B/9B stay at the floor
   (0.0074/0.0047) while from-scratch 27B rises to 0.113 — now **15–24×** with
   architecture, norm scheme, tokenizer AND probe set all held constant. (But see the
   Gemma-3 complication in `hypotheses.md`: Gemma-3 is also distillation-trained and now
   shows strong bands, so KD-suppression is a Gemma-2-recipe fact, not a KD universal.)
2. **Instruct-tuning shrinks the band — all 8 measurable pairs**, often harder than
   own-vocab suggested: gemma-3-27b 0.298→0.104, gemma-3-12b 0.114→0.030, gemma-3-270m
   0.134→0.025, gemma-3-1b 0.077→0.019, gemma-3-4b 0.061→0.013, gemma-2-2b 0.0074→0.0014,
   gemma-2-9b 0.0047→0.0027, llama3.1-8b 0.106→0.081.
3. **The sub-0.2B floor.** pythia-70m and gpt2 are unchanged (~0.015) — though
   gemma-3-270m's shared-probe band (0.134) now puts the emergence onset lower than the
   own-vocab story implied.

Published cross-family comparisons must use the shared-probe numbers from here on.

## Next (all CPU, no GPU spend)

- Size ladder across families (pythia→gpt2→gemma-3-270m/1b→qwen) → emergence curve.
- Random-`J_l` null control for the shared-`U` confound.
- Frequency confound (do J-lens top tokens just track token frequency?).
- Dimensionality: effective rank / variance concentration of J-space (is it really a
  compact "space"?).

GPU (only if the free evidence is inconclusive, and only with explicit per-run approval):
ignition + capacity on a live small open model.
