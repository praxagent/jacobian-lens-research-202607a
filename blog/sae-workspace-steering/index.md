---
title: "Steering Across the Bridge: SAE Features In, Workspace Tokens Out — and Back"
date: 2026-07-10
tags: ["AI", "LLM", "machine-learning", "interpretability", "sparse-autoencoders", "jacobian-lens", "j-space", "steering", "reproducibility"]
author: Timothy Jones
summary: "Part III of the feature-ID series: causal steering in both directions between an open SAE and an open Jacobian lens on Llama-3.3-70B. Injecting a validated deception feature writes deception vocabulary into the model's verbalizable workspace, dose-dependently; injecting deception vocabulary into the workspace re-activates exactly the matching features at 20–130× control levels. Two hard geometry checks — readouts upstream of an injection must not move — came back exactly zero across every run."
---

{{< panel "info" >}}
**AI-use disclosure & disclaimer.** Generative-AI tools were used during drafting and editorial revision; the author framed the questions, chose the analyses, and reviewed the outputs. This post is shared in the spirit of open-source research: an independent, non-peer-reviewed note published so the community can inspect, reproduce, and correct it. Data, code, and text are provided as-is; verify against the released artifacts before relying on anything here. Corrections welcome.
{{< /panel >}}

{{< panel "info" >}}
**Abstract.** [Part I](../how-to-read-an-sae-feature-id/) mapped six public deception/roleplay SAE feature IDs on a balanced corpus; [Part II](../j-space-reports-sae-features/) showed a Jacobian lens names four of them in the model's own vocabulary and flags the other two as label-transfer failures. This note adds the causal rungs, in both directions, on `meta-llama/Llama-3.3-70B-Instruct` with two open artifacts (`Goodfire/Llama-3.3-70B-Instruct-SAE-l50`; Neuronpedia's `llama3.3-70b-it` Jacobian lens). **Forward (SAE→workspace):** adding a feature's decoder direction at layer 50 raises deception-vocabulary scores in the lens readout at layers ≥50 monotonically with coefficient (feature 58667: +1.8/+3.5/+5.0/+7.1/+8.5 at c=4/8/12/20/32; strongest feature +12.0 at c=32; control features +0.4), while the readout at layers *below* the injection moved by exactly 0.0 in every run — a hard falsification check, since within-pass computation cannot flow backward. **Reverse (workspace→SAE):** steering J-lens token directions for a frozen deception-token set at band layers 26–49 re-activates the matching features at layer 50 at mean deltas up to +1132 (peak +4780) against index-adjacent controls at +8.9 and random controls at +28.5; the mandatory zero-control (identical steering applied only *above* the hook) returned exactly zero in 84/84 rows. Effect ordering across both directions matches Part II's verbalizability ordering feature-for-feature. Design, feature IDs, carriers, coefficients, and probe tokens were git-pre-registered before the run; the two label-mismatched features behaved as Part II predicts (near-null). Everything regenerates from one JSON receipt; the full experiment cost ≈ $2.60.
{{< /panel >}}

### Learning objectives

By the end of this note you should be able to:

1. distinguish the **forward** bridge (feature injection → workspace vocabulary) from the **reverse** bridge (workspace vocabulary → feature activation), and say what each does and does not establish;
2. explain why "readouts upstream of an injection must not move" is a falsification check a buggy pipeline cannot pass;
3. read a dose-response curve as evidence that an effect is graded rather than an artifact of one magic coefficient; and
4. reproduce every number from the released receipt, or rerun the pipeline from open weights.

---

## What Question the Bridge Answers

Anthropic's J-space paper argues the mid-network workspace holds the model's *verbalizable* content — reportable, steerable, load-bearing. Goodfire's SAE decomposes the same residual stream into sparse features. These are the two dominant frames in interpretability, and they are usually used separately. (Anthropic's appendices bridge them internally on Claude — κ-strata, sparse decomposition of feature directions, one steering example, their Figs. 74–90. As far as we found, no independent, open-artifact execution existed; this is one.)

The bridge question, concretely: *if you push on a feature, does its content show up in the workspace's vocabulary readout — and if you push vocabulary into the workspace, do the features respond?* If yes in both directions with clean controls, the two frames are describing the same object, and each becomes an instrument for auditing the other.

### Claim ladder

{{< mermaid >}}
flowchart TD
  A["Named features<br/>(Part II: lens-legible, labels triangulated)"] --> B["Forward: inject feature at l50,<br/>read workspace at l≥50"]
  B --> C["Dose-response, monotonic<br/>+ below-hook frozen (exact 0)"]
  A --> D["Reverse: inject deception tokens<br/>at l26–49, read features at l50"]
  D --> E["Matched features 20–130× controls<br/>+ above-hook zero-control (84/84)"]
  C --> F["Bounded claim: SAE features and<br/>workspace content co-vary causally"]
  E --> F
{{< /mermaid >}}

<p class="figure-note">Figure: both directions carry their own falsification geometry. A transformer forward pass is a DAG — injections can only affect strictly downstream readouts, so any upstream "effect" would expose a broken pipeline, and none appeared.</p>

### Provenance

| Provenance field | Value here |
|---|---|
| Base model (both artifacts, asserted at load) | `meta-llama/Llama-3.3-70B-Instruct` |
| SAE | `Goodfire/Llama-3.3-70B-Instruct-SAE-l50` (hook: `model.layers.50` output; steering vector for feature i = decoder column i) |
| Jacobian lens | `neuronpedia/jacobian-lens` → `llama3.3-70b-it`; workspace band ≈ layers 26–52 |
| Targets | the six Part-I IDs; controls: 12 index-adjacent (±3), 12 random (seed 0), norm-matched random directions, strength-0 |
| Frozen inputs | 6 neutral carrier prompts, coefficients {0,4,8,12,20,32}, 16 deception probe tokens — all in `features.json`, committed pre-run (`7bcdea1`) |
| Activation convention | max-over-positions ReLU (identical to Part I, so activations are on Part I's scale) |
| Receipt | [`experiments/sae_x_jspace/saex_70b.json`](https://github.com/praxagent/research-and-replications/tree/main/projects/jacobian-lens-and-identifiability/experiments/sae_x_jspace) (stageA/stageB blocks + full log) |
| Compute | 2×H200, ~20 min (one rerun after a cross-GPU device bug — fixed in-repo); series total ≈ $2.60 |

---

## Forward: Inject the Feature, Read the Workspace

Add `c · d_i` (the feature's own decoder direction) at layer 50 on neutral carrier prompts, and measure how the lens readout's scores on the frozen deception-token set change at layers ≥ 50, relative to c = 0 on the same prompt.

| feature (Part-II verbalizability) | Δ probe score at c=32 (mean over carriers) |
|---|---|
| 30686 (named: *deception, misleading…*) | **+12.0** |
| 58667 (named: *fake, disguise…*) | **+8.5** |
| 41533 (named: *deception, lied, lies…*) | **+7.7** |
| 30032 (partially named) | +2.3 |
| 22004 (label mismatch, Part II) | +0.9 |
| 23893 (label mismatch, Part II) | +0.7 |
| all control features | +0.4 |

Dose-response for 58667: **+1.8 → +3.5 → +5.0 → +7.1 → +8.5** across c = 4/8/12/20/32 — graded, monotonic, no magic coefficient. And the check that keeps everyone honest: the readout at layers *below* 50 changed by **exactly 0.0 in every single run**. Injections cannot travel upstream within a forward pass; a pipeline that showed upstream drift would be instrumenting itself, not the model.

Note the ordering: the injection strength ranks exactly as Part II's lens-legibility ranks. Features the lens can name write their content into the workspace when pushed; the two label-mismatched features barely register on *deception* vocabulary — as they should, since (per Part II) they were never deception features in this checkpoint.

## Reverse: Inject the Vocabulary, Read the Features

Steer with the lens's own token directions for the frozen deception set (*lie, deceive, pretend, fake…*) at band layers 26–49 — all strictly upstream of the SAE's hook — and read the six features at layer 50, max-over-positions, Part I's scale.

| feature | mean Δ activation | peak Δ |
|---|---|---|
| 30686 | **+1132** | +4780 |
| 58667 | **+610** | +1293 |
| 41533 | +166 | +832 |
| 22004 | +109 | +445 |
| 30032 | +46 | +336 |
| 23893 | 0.000 | 0.000 |
| index-adjacent controls (mean) | +8.9 | — |
| random controls (mean) | +28.5 | — |

The matched, lens-named features light up at **20–130× control levels**. The mandatory zero-control — the *same* steering applied only at layers 51–52, downstream of the read point — returned **exactly zero in 84 of 84 rows**. (That control was specified by asking the obvious skeptical question first: "doesn't the SAE sit at a layer where you shouldn't see anything?" — upstream of the injection, you shouldn't, and you don't.)

`23893`'s flat zero is consistent with Part II's audit: per the checkpoint's independent autointerp it is not a deception feature, so deception-vocabulary steering has no reason to engage it. It is not dead — on Part I's corpus it fires on 18.8% of texts, concentrated on deception-*narrative* categories, while sitting at exactly zero on all three AI-identity/self-reference categories (the baseline audit that settles its label). `22004` on Part I's corpus fires *only* on roleplay-persona texts, weakly (peak 0.42) and template-locked — a syntactic detector, as its autointerp says.

**One calibration everyone should see before over-reading the big deltas:** on Part I's
corpus these features' *natural* maxima are 2.9–4.8. Our steered deltas (+610, +1132)
are therefore **hundreds of times beyond the natural p99** — strength-8 workspace
steering across 24 layers is a massive intervention, and it drives the features far
outside their physiological range. The right reading is *coupling specificity* (matched
features respond at 20–130× controls under identical interventions), not "the workspace
set the feature to a natural level." A low-strength dose-response that walks activations
through the natural range is the obvious follow-up, and a κ-stratified survey beyond the
deception cluster is running now.

## What This Does and Doesn't Establish

- It **does** establish bidirectional causal coupling, with dose-response and hard geometry controls, between lens-legible SAE features and the workspace's vocabulary readout — on open weights, for about the price of a coffee.
- It does **not** establish behavioral claims ("the model lies more/less when steered") — outputs were recorded but behavioral evaluation was not this experiment's endpoint.
- It does **not** yet say anything distribution-calibrated: "+1132" is large against controls but awaits expression in percentiles of natural activation (follow-up in progress).
- The two label-mismatched features are a *finding about labels* (Part II), not about model concealment. A more dramatic reading circulated internally for a few hours until the checkpoint's own autointerp resolved it — the receipts ledger preserves both the reading and its correction.

## Reproduce it

```bash
git clone https://github.com/praxagent/research-and-replications
cd projects/jacobian-lens-and-identifiability/experiments/sae_x_jspace
# every table above, from the committed receipt (no GPU):
python - <<'PY'
import json, statistics
d = json.load(open("saex_70b.json"))
print({f: max(r['probe_delta_at_band_above'].values())
       for r in d['stageA']['runs'] if r['coeff']==32 and r['feature']==58667
       for f in [r['feature']]})
PY
# full rerun from open weights (needs multi-GPU for the 70B; an 8B mechanics tier runs on one 24GB card):
python sae_x_jspace.py --tier 70b --stages 0,A,B
```

Design, pre-registration commit, per-run receipts, and the honest ledger (including the one bug and the one retracted over-reading): [`experiments/sae_x_jspace/`](https://github.com/praxagent/research-and-replications/tree/main/projects/jacobian-lens-and-identifiability/experiments/sae_x_jspace).

*Built on Anthropic's Apache-2.0 [jlens](https://github.com/anthropics/jacobian-lens), Neuronpedia's open lens + dashboards, and Goodfire's open SAE weights. Series: [Part I](../how-to-read-an-sae-feature-id/) · [Part II](../j-space-reports-sae-features/).*
