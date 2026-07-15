# Features as Rewards — reader benchmark for hallucinated-entity detection

Motivated by **"Features as Rewards: Scalable Supervision for Open-Ended Tasks via
Interpretability"** (Goodfire, arXiv 2602.10067). A frontier Pro plan review
([`ADJUDICATION.md`](ADJUDICATION.md)) established that what we can do cheaply and honestly
is **not** a replication of the paper's Gemma-3 number (their data/grader/probe code aren't
public) but a **benchmark of readers** — supervised probe vs. logit lens vs. fitted Jacobian
lens vs. label-selected SAE latent — for **discriminating** hallucinated from supported
entity spans on public gold labels, plus an own-graded Gemma-3-12B arm. See
[`PROTOCOL.md`](PROTOCOL.md) (v2, hardened) for the frozen design.

- **Paper assessment + tiered plan:** [`PROPOSAL.md`](PROPOSAL.md)
- **Artifact access snapshot:** [`FEASIBILITY.md`](FEASIBILITY.md)
- **What the paper's "features" are:** attention **probes on raw residual-stream
  activations** — NOT SAE features (see PROPOSAL). SAEs are our *extension*, not theirs.

## Scope (TJ, 2026-07-14)

- **Tiers 0–2 + extension** authorized; **Tier 3 (the ~$4k RL run) excluded.**
- **Three models** compared: Llama-3.1-8B-It, Llama-3.3-70B-It, Gemma-3-12B-It.
- **Four readers** of the same held-out entity spans: supervised **attention probe**
  (theirs) · unsupervised **logit lens** · unsupervised **fitted Jacobian lens** (ours) ·
  sparse **SAE latent** (Goodfire's own instrument, unused in the paper).
- Cheapest-model-first sequencing; every paid pod gated on a fresh cost estimate.

## Replication status

| # | Claim | Model(s) | Status |
|---|---|---|---|
| C1 | Detection probes read calibrated hallucination belief (Localize/Classify AUC) | 8B → 70B → Gemma | proposed |
| C1x | Which reader carries the signal: probe vs logit-lens vs J-lens vs SAE | all 3 | proposed (our extension) |
| C2 | Reward probes (retract/correct) grade interventions | 8B → 70B | proposed |
| C3 | Probe-reward best-of-N > model-self-judge at test time (no RL) | 8B → 70B | proposed |
| C4 | Activation-source invariance (probe ordering base vs policy) | 8B | proposed |
| C5 | Full RLFR training → decomposed 58% reduction | — | **excluded (Tier 3)** |

Status vocabulary: proposed → frozen (outcome-masked) → running → confirmatory → reported.

## Method note

The paper's probes are attention probes trained on residual-stream activations to imitate
a grader's hallucinated-entity labels. We reuse the **public gold labels** from
`obalcells/longfact-annotations` (the LongFact++ / Obeso–Nanda work) where they exist
(Llama-3.1-8B, Llama-3.3-70B, gemma-2-9b), avoiding the paper's grader cost and
non-determinism. The Gemma-3-12B arm's label path is decided when we sequence it (see
FEASIBILITY §label gap).

## How to run

Nothing is frozen or run yet. The pipeline is being built and CPU-smoked first
(integrity/GPU playbooks: prove the path free before any paid compute). Run commands will
land in `experiments/<name>/README.md` once the protocol is frozen.
