# Lit note — LLM "subjective experience" reports (Berg et al., arXiv:2510.24797) & follow-ups

> **Provenance & confidence (read first).** This note is a **secondhand synthesis of a
> GPT-Pro deep-research report** (provided by TJ, 2026-07-08). It is **NOT independently
> verified.** The report itself flagged real limits: Google Scholar returned 403 and the
> Semantic Scholar redirect wasn't openable, so the citation graph is *best-effort, not
> exhaustive*; several citation relations are **medium-confidence**; and **no peer-reviewed
> journal version** of the original could be confirmed (only the arXiv/DataCite DOI
> `10.48550/arXiv.2510.24797`). **Before any public use, verify**: each arXiv ID exists and
> matches its claimed authors/title, the citation relations, and whether journal versions
> now exist. Treat everything below as a map to check, not settled fact.

## The original paper

**"Large Language Models Report Subjective Experience Under Self-Referential Processing"**
— Cameron Berg, Diogo de Lucena, Judd Rosenblatt (**AE Studio**). arXiv:2510.24797
(v1 2025-10-27, v2 2025-10-30), cs.CL/cs.AI.

**Claims:** sustained *self-referential prompting* reliably elicits structured first-person
"subjective experience" reports across GPT, Claude, and Gemini families; these reports are
**modulated by sparse-autoencoder (SAE) features** associated with **deception / roleplay**
(suppress those features → models affirm subjective experience more, and factual honesty
on TruthfulQA moves the same way); reports show **semantic convergence** across families
and **generalize** to downstream introspective reasoning.

## Follow-up landscape (per the report)

| Item | Authors | Venue / date | Type | Verdict vs. original | Conf. |
|---|---|---|---|---|---|
| *No Reliable Evidence of Self-Reported Sentience in Small LLMs* | Kaiser, Enderby | arXiv 2026-01-20 | direct empirical challenge | **partially debunks** the "denials are deceptive" reading — activation truth-classifiers on Qwen/Llama/GPT-OSS (0.6–70B) find sentience denials look *truthful* | High |
| *Indications of Belief-Guided Agency & Meta-Cognitive Monitoring in LLMs* | Steinmetz Yalon, Goldstein, Mudrik, Geva | arXiv 2026-02-02 | extension | **partially affirms** the broader program (measurable internal belief dynamics; "Belief Dominance" causally drives action) — but doesn't test the verbal-report paradigm | Med-high |
| *When Models Examine Themselves: Vocabulary–Activation Correspondence…* | Dadfar | arXiv 2026-02-11 | extension | **affirms** the narrow claim that self-referential language tracks internal computation (Llama-3.1, Qwen2.5-32B); explicit citation to Berg not confirmed in-session | Med |
| *Demystifying Apparent Experience in LLMs* | Hudson, Hudson | PhilArchive 2025-12-26 | critique | **debunks the interpretation, not the effect**: apparent experience = constraint-driven stabilization, not awareness | Med-high |
| *Pseudo-Consciousness in AI* | Prestes | PhilSci-Archive 2026-04 | conceptual critique | **partially debunks** strong inference; coins "pseudo-consciousness" (functional/governance framing), keeps the behaviors ethically significant | Med |
| *Exploitation Without Deception: Dark Triad Feature Steering…* | Berg, Lulla | arXiv 2026-05-10 | extension (same lab) | **partially affirms** the *mechanistic* claim that SAE features causally gate complex behavior; exploitation dissociates from deception (separable circuits) in Llama-3.3-70B | High |

**Bottom line (the report's judgment):** *mechanistically interesting and probably a genuine
behavioral phenomenon; NOT yet strong evidence of consciousness; the interpretation remains
actively contested.* No decisive confirmation or refutation as of 2026-07-08; the durable,
better-supported part is narrow — self-referential/introspective outputs relate to internal
computational structure, and latent (SAE) interventions gate behavioral regimes. The leap to
*latent self-belief / subjective experience / consciousness* is repeatedly challenged.

## Why this matters for us

- **Same shape as our J-space audit.** "Strong mechanistic result, contested consciousness
  framing" is exactly the pattern in our [jacobian-lens project](../../projects/jacobian-lens-and-identifiability/writeup.md):
  the measurement replicates; the *interpretation* outruns it. This literature is a second,
  independent instance — worth citing as evidence that the field's response to
  consciousness-framed LLM results is "affirm the mechanism, contest the interpretation."
- **Directly feeds the coming SAE work.** The original's SAE-feature modulation and the
  Berg & Lulla dark-triad steering are the *feature-steering causal-gating* methodology TJ's
  SAE research will use. These are prior art + method references for that project.
- **Honesty/introspection thread.** The "self-report tracks internal computation" result is
  adjacent to our J-lens/introspection framing and to Prax's own honesty-guard lane.

## To verify / follow up

Confirm the six arXiv IDs + authors; check the Berg↔Dadfar citation relation; look for any
peer-reviewed versions; and (if we go deeper) consider whether Kaiser & Enderby's
truth-classifier method could be combined with the self-referential prompts — the reconciling
experiment the report says nobody has run yet. Authors here (Berg, Rosenblatt @ AE Studio;
Kaiser; Geva; Goldstein; Mudrik) can be added to
[`shared/researchers/handles.yaml`](../researchers/handles.yaml) once X handles are confirmed.
