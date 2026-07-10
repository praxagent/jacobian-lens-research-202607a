# Results — SAE × J-space bridge (ledger of actual runs)

## CPU mock (gpt2 + random SAE, free) — plumbing + null behavior

All three stages ran; both geometry asserts live. Random-SAE κ ≈ 3.1–3.4 = the
Gaussian baseline. `sae_x_jspace_mock.json`.

## 8B tier (3090, ~$0.15, terminated) — mechanics on real weights

Goodfire 8B SAE + `llama3.1-8b-it` lens. No assert violations. κ discriminates on a
real SAE: an email feature (f30035) reads ` mailbox/ mail/ emails` at κ=10.4; a
"types" feature hits κ=73; unstructured features sit near 3. (Target IDs are
meaningless at this tier — they index the 70B dictionary.) `saex_8b.json`, `saex.log`.

## 70B tier (2×H200, ~20 min ≈ $2.4, terminated) — THE SCIENCE RUN

Goodfire/Llama-3.3-70B-Instruct-SAE-l50 + neuronpedia `llama3.3-70b-it` lens (pairing
assert passed). One rerun: pass 1 crashed at Stage B on a cross-GPU device mismatch
(layer-50 resid on cuda:1, SAE on cuda:0 — fixed, committed); model cache made the
rerun ~8 min. Receipt: `saex_70b.json`, log `saex70.log`.

### Stage 0 — κ triage: the J-lens NAMES four of the six deception features

| feature (TJ's validated label) | κ_max (band) | J-lens top tokens at peak |
|---|---|---|
| 30686 tactical deception/misdirection | **37.4** | ` deception, misleading, dece, deceptive, deceive, deceived` |
| 58667 maintaining cover stories | **10.0** | ` fake, disguise, disgu, Fake, disguised` |
| 41533 acts of deception/dishonesty | **8.1** | ` deception, lied, deceit, falsehood, lies` |
| 30032 pretending/feigning | 5.0 | ` innocent, fake, supposedly` (+ whitespace) |
| 23893 concealing artificial nature | 6.5 | non-Latin junk — NOT verbalizable |
| 22004 assistant roleplaying persona | 6.1 | whitespace/punctuation — NOT verbalizable |

Controls: adjacent median κ 6.5, random 5.5. The three strongest deception features
are *directly legible* through the lens — Anthropic's Fig-74/90-style analysis
reproduced on fully open artifacts. The two that are NOT legible are precisely the
persona/concealment features (see "the dissociation candidates" below).

### Stage A — SAE→workspace steering: dose-dependent, geometry-clean

Steering each feature at layer 50, deception-probe-token score delta at band layers
≥ hook (mean over carriers, c=32): **30686 +12.0 · 58667 +8.5 · 41533 +7.7** ·
30032 +2.3 · 22004 +0.9 · 23893 +0.7 — vs **+0.4 for control features**. Monotonic
dose-response (58667: +1.8/+3.5/+5.0/+7.1/+8.5 at c=4/8/12/20/32). **below-hook
readout drift = 0.0 exactly, every run** — the falsification geometry held.
Effect ordering matches Stage 0 verbalizability exactly.

### Stage B — workspace→SAE steering: the reverse bridge, at 20–130× controls

J-lens deception-token directions steered at band layers 26–49 (all upstream of the
hook), feature activations read at layer 50 (max-over-positions, the llm_selfref_pre
convention):

| feature | mean Δ activation | max Δ |
|---|---|---|
| 30686 | **+1132** | +4780 |
| 58667 | **+610** | +1293 |
| 41533 | +166 | +832 |
| 22004 | +109 | +445 |
| 30032 | +46 | +336 |
| 23893 | **0.000 exactly** | 0.000 |
| adjacent controls (mean) | +8.9 | — |
| random controls (mean) | +28.5 | — |

Zero-control (identical steering applied only ABOVE the hook): **84/84 rows exactly
zero** — the hard assert TJ specified held universally.

### Reading

**The bridge is real and bidirectional.** Independently validated deception features
are, for the most part, *verbalizable workspace content*: the J-lens names them
(Stage 0), steering them injects deception content into the workspace readout
dose-dependently (Stage A), and steering the workspace with deception *tokens*
re-activates exactly those features at 20–130× control levels (Stage B) — with
effect sizes rank-consistent across all three stages.

**The dissociation candidates.** Features 22004 (persona roleplay) and 23893
(concealing artificial nature) act like the OTHER kind of feature: non-verbal
readouts, weak workspace injection, and — for 23893 — exactly zero response to
workspace steering. If those labels are accurate, the two features most associated
with *hiding what the model is* are the two least visible to the verbalizable
workspace. Honest caveats before anyone quotes that sentence: (a) autointerp labels
can be noisy; (b) 23893's exact-zero pattern is consistent with a dead/never-firing
latent on our carriers (its baseline activation needs checking — Goodfire zeroed
toxic latents in this SAE and encoder thresholds vary); (c) two features is an
anecdote, not a distribution. Follow-up: baseline-activation audit for 23893/22004 on
natural deception text (TJ's 1,120-text corpus), and a κ-stratified sweep beyond the
deception cluster.

**Comparability note:** Stage B deltas are on the same max-activation scale as TJ's
clean-room study; expressing them as percentiles of his natural-text distributions
needs the per-feature stats from `llm_selfref_pre` — queued as the write-up step.

Total experiment cost: ≈ **$2.6**.
