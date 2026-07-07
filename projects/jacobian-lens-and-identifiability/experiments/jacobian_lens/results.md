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

**Honest read — preliminary, NOT yet a finding.** Taken at face value this says gpt2 has
no "workspace band," consistent with the band being a larger-model phenomenon. **But two
things must be controlled before this means anything:**

1. **Shared-unembedding confound (the big one).** `D_l = U @ J_l` shares the *same* `U`
   across all layers. If `U` dominates the geometry (the transports `J_l` being near-
   identity-like), cross-layer CKA is high *by construction* and says little about a
   workspace. **Control needed:** a null with random `J_l` (does CKA stay ~0.99?), and/or
   a geometry that divides out the shared-`U` component.
2. **Calibration.** eliebak reports a sensory→workspace→motor block structure on larger
   models with this exact readout — so the method *can* resolve a band. We must reproduce
   a band on ≥1 model before "gpt2 has none" is informative (else it's just a small/weak
   model, or our pipeline differs from eliebak's).

**So the honest status of the audit right now: one small-model data point suggesting no
band, with a live confound. Not evidence of "oversold" yet — evidence that we need the
size ladder + the null control.** That's the next CPU work (free): run
pythia-70m / gemma-3-270m / gemma-3-1b / qwen-small on the same script, add the random-J
null, and see whether a band *emerges with scale*. Only if a band clearly appears in
larger models and is absent/weak in smaller ones does the "band = large-model phenomenon,
general framing overreached" argument get real support.

## Next (all CPU, no GPU spend)

- Size ladder across families (pythia→gpt2→gemma-3-270m/1b→qwen) → emergence curve.
- Random-`J_l` null control for the shared-`U` confound.
- Frequency confound (do J-lens top tokens just track token frequency?).
- Dimensionality: effective rank / variance concentration of J-space (is it really a
  compact "space"?).

GPU (only if the free evidence is inconclusive, and only with explicit per-run approval):
ignition + capacity on a live small open model.
