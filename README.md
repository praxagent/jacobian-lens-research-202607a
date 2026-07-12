# Jacobian Lens Research

The repo `jacobian-lens-research-202607a` — a focused, open research campaign by
**[praxagent](https://praxagent.ai)**, an independent, safety-first AI research &
consultation org — on **reading the internals of large language models with the Jacobian
lens**: the LLM "global workspace" (Anthropic's J-lens / J-space), what it does and does
*not* reveal under pressure once you run the controls, and the nonlinear-ICA
**identifiability** lineage it is read as vindicating — is the Jacobian's support really
what makes a model's latents recoverable? One line of work carried end to end: fit our own
lens on a frontier open-weights model, stress-test it, run the controls, and report
honestly what held. **Open by default, honest about what reproduced and what didn't,
reproducible on modest hardware.**

This is one campaign, not a catch-all. Other praxagent research lines live in their own
repositories; the dated slug (`…202607a`) marks this one, and later campaigns get their
own.

This repository is deliberately **separate from the Prax product harness**. Prax's
code stays a lean, safety-first agentic harness; exploratory research — reproducing
papers, running experiments, testing ideas before they're mature — lives here. Nothing
in this repo is imported by, or required for, Prax. When a replication produces a result
that should change the harness, the *conclusion* (a principle, a flag, a metric) gets
written up in `prax/docs/research/` and implemented there; the experimental code stays
here.

**Standing on shoulders, honestly.** We build on others' released code (with attribution
and license care — e.g. Anthropic's Apache-2.0 `jacobian-lens`), reproduce their results
openly, and aim to *do better only where we genuinely can* — theory grounding, open-model
robustness, reproducibility, rigor — never by overclaiming or out-scaling what we can't.

## Why the split

- **Prax is a product.** Its dependency tree, CI, and code surface are audited by
  strangers; research code (heavy ML deps, half-finished experiments, throwaway
  notebooks) does not belong in that trusted base.
- **Research has a different lifecycle.** Replications are allowed to be incomplete,
  to fail, to be rewritten. The product cannot.
- **Honesty is easier when they're separate.** A replication that "doesn't reproduce"
  is a valid, publishable outcome here; it would be noise in the product repo.

## Layout

Each project is a self-contained directory with its own README, a replication-status
table, and its experiments.

| Project | What it covers | Status |
|---|---|---|
| [`jacobian-lens-and-identifiability/`](jacobian-lens-and-identifiability/) | The LLM "global workspace" (Anthropic J-lens/J-space) and the nonlinear-ICA **identifiability** lineage it's read as vindicating — is the Jacobian's support really what makes latents recoverable? | in progress |

## Conventions

- **Package manager: `uv`.** Deps declared in `pyproject.toml`; heavy/optional ones
  (torch) behind extras so the light replications run without them.
- **Honesty over green.** Every replication README states plainly what was **run** vs.
  **written-but-not-executed**, what reproduced, and what didn't. Never report a
  metric that wasn't actually produced by running the code.
- **Cite the source, reproduce the metric.** Each replication names the exact claim /
  theorem / figure it targets and the paper's own success metric.
- **No product coupling.** Do not `import prax` here; do not add this repo as a Prax
  dependency.

## Running

```bash
uv venv && uv pip install -e .          # light replications (numpy only)
uv pip install -e ".[torch]"            # add the nonlinear (flow-based) replications
```
