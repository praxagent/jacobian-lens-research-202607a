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

## Next (all CPU, no GPU spend)

- Size ladder across families (pythia→gpt2→gemma-3-270m/1b→qwen) → emergence curve.
- Random-`J_l` null control for the shared-`U` confound.
- Frequency confound (do J-lens top tokens just track token frequency?).
- Dimensionality: effective rank / variance concentration of J-space (is it really a
  compact "space"?).

GPU (only if the free evidence is inconclusive, and only with explicit per-run approval):
ignition + capacity on a live small open model.
