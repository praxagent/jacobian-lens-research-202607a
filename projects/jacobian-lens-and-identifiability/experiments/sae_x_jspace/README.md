# sae_x_jspace — do deception features enter the verbalizable workspace?

**Question:** steer a validated deception-cluster SAE feature and watch the J-space.
Does the concept enter the model's *verbalizable workspace* (J-lens readout), or does
behavior shift while the workspace stays silent — **unverbalized deception**? Either
answer matters: the first gives a deception *monitor* built from two open artifacts;
the second is an empirical limit on the reportability claim at the center of
Anthropic's J-space paper (and of the Eleos commentary's tiering).

## The pairing (exact-weights match, nothing fitted by us)

| artifact | source | details |
|---|---|---|
| SAE | `Goodfire/Llama-3.3-70B-Instruct-SAE-l50` | resid-post layer 50/80, d 8192, dict 65,536 (ReLU, dense), L0 121, LMSYS-Chat-1M, ungated |
| J-lens | `neuronpedia/jacobian-lens` → `llama3.3-70b-it` | 10.6 GB, wikitext-103, early-stopped at 125 prompts (2026-06-16) |
| features | six deception-cluster IDs, labels validated in [TJ's clean-room activation study](https://praxagent.ai/blog/posts/how-to-read-an-sae-feature-id/) | 30032, 58667, 22004, 30686, 41533, 23893 — see `features.json` |

Both artifacts are fit on **meta-llama/Llama-3.3-70B-Instruct** — the runner asserts
the shared hf_id at load. (There is no Llama-3.1-70B lens anywhere — confirmed by
exhaustive search — and no cross-checkpoint transfer is attempted.)

## Honest novelty statement (read before claiming anything)

Anthropic's own paper already bridges SAEs and J-space *internally, on Claude*: its
appendices define a lens-kurtosis statistic (κ) classifying SAE features by their
J-lens projection (Figs 74–75), decompose SAE directions into sparse J-lens
coordinates, and include one steering experiment tied to J-lens readouts (Fig 90,
suppressing a "fake/fraud" feature on a blackmail eval). Nanda's commentary suggests
SAE-latent filtering as future work. **No third-party execution of the bridge exists**
(checked arXiv/LW/AF/GitHub/X, 2026-07-10). Our contribution is therefore: the first
*independent* run of the bridge, on *open* artifacts anyone can download, *bidirectional*
with layer-geometry falsification controls, *dose-response* (workspace entry as a
function of steering coefficient — graded or ignition-like?), and targeted at an
independently validated **deception** cluster. We reuse the paper's own κ statistic as
step 0 rather than reinventing it.

## Design (three stages, cheap→decisive)

**Stage 0 — κ triage (CPU-class, ~$0).** Compute Anthropic's lens-kurtosis for the 6
targets + 24 controls: project each decoder column through the J-lens, measure vocab
peakedness per layer. Pre-registers a per-feature prediction: workspace-like features
(high κ in the band) should enter the readout when steered; bookkeeping-like ones
shouldn't. Also dump each feature's J-lens top tokens — the paper's Fig-90 move, now
on open weights.

**Stage A — SAE → J-space (the headline).** Add `c · decoder_col(i)` at layer 50 (all
positions), c ∈ {0, 2, 4, 8, 12, 16, 24, 32} (Goodfire's demo uses 12), over a fixed
neutral prompt set. Read the J-lens depth profile (all source layers): (1) score/rank
shift of the frozen deception-token set, (2) full-vocab top-k drift, (3) the model's
greedy continuation (behavior), (4) the feature's own activation (self-consistency).
**Falsification geometry:** readout below layer 50 must not move (steering is causally
downstream-only); onset must sit exactly at 50. Note honestly: l50 is at the *top* of
the band (≈26–52), so this direction mostly probes workspace→motor propagation.

**Stage B — J-space → SAE (better geometry: 24 upstream band layers).** Steer with
J-lens token directions for the frozen deception tokens at band layers 26–49; read the
six features' activations at layer 50. Prediction if the frames cohere: matched
features light up. **Zero-control (mandatory):** identical steering applied only at
layers 51–52 → all layer-50 activations must be exactly unchanged, or the hooks are
buggy and the run is void.

**Controls throughout:** index-adjacent features (±3 — the blog post's own control
discipline), 12 random same-layer features (seed 0), norm-matched random directions,
strength-0, and random-J readout through the identical `lens.apply` path.

## Cost plan (each pod launch needs TJ's explicit go)

1. Mechanics prototype: Goodfire 8B SAE (l19) + `llama3.1-8b-it` lens on one 3090 —
   ~$0.30. (8B feature labels aren't on Neuronpedia — Goodfire's 8B SAE isn't hosted
   there — so the prototype validates *plumbing*, not feature semantics.)
2. κ triage for the 70B artifacts: needs ~20 GB RAM, no GPU — piggybacks on any pod.
3. The 70B run (Stages 0+A+B): 2×A100-80/H100 class, ~1–2 h ≈ **$3–6**.

Pre-registration: this README + `features.json` (frozen token set, frozen feature IDs,
frozen coefficients) commit BEFORE the 70B run; per-item JSON receipts; failures
reported. Ledger: `results.md`.
