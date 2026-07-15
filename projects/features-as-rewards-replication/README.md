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

## Scope (TJ, 2026-07-14/15; framing per the Pro review, `ADJUDICATION.md`)

- A **reader benchmark** for hallucinated-entity **discrimination** (not a replication of the
  paper's number; RL / the 58% is out of scope).
- **Readers** on the same held-out entity spans: supervised **attention probe** · unsupervised
  **logit lens** · unsupervised **fitted Jacobian lens (ours)** · label-selected **SAE latent**.
- **Arms** (cheapest first; the arm set is fixed at freeze):
  1. **Llama-3.1-8B** — primary confirmatory (public gold labels), 4 readers
  2. **Llama-3.3-70B** — public gold labels, 4 readers, our J-lens
  3. **gemma-2-9b** — public gold labels, 4 readers
  4. **Gemma-3-12B** — own grounded-grader labels, 4 readers (+ Gemma Scope 2 SAE)
  5. **Qwen3.5-397B (flagship)** — **pre-registered CONDITIONAL extension** (TJ, 2026-07-15):
     go/no-go decided *after* the four confirmatory arms, as a cost-vs-value call; if run, its
     result is reported unconditionally (no cherry-picking). Own grounded-grader labels,
     **3 readers** (no public SAE): probe / logit lens / **our fitted J-lens**
- **Deliverable beyond the paper:** publish the LongFact hallucination-annotation sets we
  create for **gemma-3-12b** and **Qwen3.5-397B** as an open HF dataset (`praxagent-org`),
  extending the public LongFact++ family to two un-annotated models (PROTOCOL §11).

## Status

| Item | Status |
|---|---|
| Reader benchmark (probe vs logit vs J-lens vs SAE), 5 arms | protocol v2, pre-freeze |
| Grounded grader + local Wikipedia labels (Gemma-3, Qwen-397B) | built, CPU-proven |
| Public label release (gemma-3-12b, Qwen-397B) | planned deliverable |
| Best-of-N reader-as-reward at test time (no RL) | registered, secondary |
| RLFR training / the 58% | **out of scope** |

Status vocabulary: proposed → frozen (outcome-masked) → running → confirmatory → reported.

## Method note

We reuse the **public gold labels** from `obalcells/longfact-annotations` (LongFact++ /
Obeso–Nanda) for Llama-3.1-8B, Llama-3.3-70B, gemma-2-9b. Gemma-3-12B and Qwen3.5-397B have no
public labels, so we generate our own with a **grounded grader** (cheap OpenRouter model + a
pinned local Wikipedia snapshot; `GROUNDING.md`), disclosed as grader-agreement not truth and
validated against the public labels — and we release those two label sets publicly.

## How to run

Nothing is frozen or run yet. The pipeline is being built and CPU-smoked first
(integrity/GPU playbooks: prove the path free before any paid compute). Run commands will
land in `experiments/<name>/README.md` once the protocol is frozen.
