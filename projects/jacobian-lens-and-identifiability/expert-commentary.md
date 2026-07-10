# Deep read — Anthropic's invited expert commentary on J-space

_Deep-read 2026-07-08 of Anthropic's 53-page **["External commentary on Verbalizable
Representations Form a Global Workspace"](https://www-cdn.anthropic.com/files/4zrzovbb/website/cc4be2488d65e54a6ed06492f8968398ddc18ebe.pdf)**
(a curated set of three invited commentaries). Two of the three are new to us and **two
findings here directly reshape our open-model audit** — flagged ⚠️ below._

The three commentaries span a spectrum from **warmly affirming** (Dehaene & Naccache) to
**graded/structured** (Eleos) to **method-focused and framing-skeptical** (Nanda). All
three separate a real mechanistic finding from the "global workspace / consciousness"
framing — the same split our audit is built on.

## 1. Dehaene & Naccache (GNW originators) — affirm with caveats

The commentary we already summarize in [background.md](background.md) (this PDF is the
authoritative host of it). Additions worth noting:

- **Co-evolution caveat (methodological).** Their note states the Anthropic report "was
  still evolving, partly in response to our queries," and several confirmatory results
  were **added after their first draft** (marked in italic). So some supporting
  experiments were generated *during* the review exchange, not pre-registered — a mild
  caution on the evidence's independence.
- Their verdict is **functionalist and cautiously affirmative**: Claude "approximates the
  functional architecture of conscious processing" (access, not phenomenal); the "hard
  problem" they treat as a likely crypto-dualist intuition. Ignition still not shown;
  capacity ~6 not ~25; subframe not dedicated neurons; no recurrence/body/episodic memory.
- They read the spontaneous emergence of a workspace as evidence it is a **"universal
  computational solution."** ⚠️ That universality assumption is exactly what our
  family-dependence result puts to the test.

## 2. Butlin, Shiller, Plunkett & Long (Eleos AI / Rethink Priorities) — graded verdict

The authors of *"Consciousness in AI: Insights from the Science of Consciousness"* (2023)
apply their **consciousness-indicator framework** and reach a careful, tiered conclusion.

**Their core move — three claims of increasing strength:**
1. **Privileged SET** — some representations are cognitively accessible (report,
   flexibility, broad influence). **Firmly supported.**
2. **Privileged STREAM** — those form a *unified* stream. **Suggestive, not conclusive.**
3. **GWT WORKSPACE** — that stream has full GWT structure (modules + broadcast-to-all +
   state-dependent selection). **Not established** (the paper concedes no evidence of
   encapsulated *modules*; LLM "broadcast" via MLP neurons + some attention heads differs
   from canonical broadcast).

They call it *"the most significant evidence of consciousness in LLMs so far uncovered by
mechanistic interpretability research"* — but repeatedly warn the term "global workspace"
*"can naturally be read as making a stronger claim"* than the evidence supports. They
sharply separate **access** (evidenced) from **phenomenal** consciousness (*"we remain
highly uncertain"*), and argue the work should still raise the community's attention to
**moral status / model welfare** — via phenomenal consciousness if present, but also via
access-consciousness-as-morally-significant (Levy 2024) and **agency**.

> ⚠️ **The critical caution for OUR audit — the tokenizer/vocabulary confound.** *"The
> central issue is that the J-space is defined in terms of the model's token vocabulary."*
> J-lens can only surface concepts that map to a single vocab token; it conflates
> Dog/DOG/ dog/chien and misses multi-token or non-lexical concepts. It is only an
> approximation to a true vocabulary-independent "W-space." **Different model families
> have different tokenizers/vocabularies, so J-lens-based "workspace" measures may not be
> commensurable across families — our Qwen≫Gemma gap could be *partly* a
> J-space-vs-W-space approximation artifact, not a real difference in workspace
> structure.** This is now a top-priority confound to address before we publish the
> family-dependence claim (control for tokenizer/vocab; ideally a vocab-independent proxy;
> at minimum, report it as a limitation and test sensitivity).

They also flag the **base-vs-posttrained** result (base model represents the *user's*
properties; post-trained represents the *Assistant's* reactions) — a *qualitative content*
change from post-training that **complements our finding that instruct-tuning reduces band
strength** (their language: post-training pulls toward "a coherent persisting point of
view").

## 3. Neel Nanda (DeepMind interp lead) — the independent open-model replication ⚠️

**The most audit-relevant item in the entire PDF.** Nanda (with MATS scholars Camila Blank
and Agam Bhatia) **independently replicated core J-space findings on an open model**, and
his results substantially corroborate ours.

**His four-claim decomposition** (and how much he buys each):
- **Scientific** (a "cognitive space"/working memory stores intermediate variables in a
  forward pass) — **strongly persuaded.**
- **Methodological** (J-Lens works, beats logit/tuned lens; the Jacobian is near-causal
  because "no time for nonlinearities") — **strongly persuaded.**
- **Pragmatic** (useful auditing tool) — **somewhat** (good for hypothesis *generation*,
  not *validation*; expects many false positives — "but basically no existing
  interpretability technique meets this bar").
- **Philosophical** (global workspace) — **explicitly declines to endorse**, calls it "the
  least interesting claim." ⚠️ A DeepMind interp lead separating the real finding from the
  GWT branding is strong support for our audit's central distinction.

**The replication (details we can act on):**
- **Model:** Qwen 3.6 27B (dense), Jacobians to the penultimate layer, **n=25** prompts of
  128 tokens (Pile/wikitext), skip first 4 high-norm tokens, 64 layers.
- ⚠️ **CKA bands were "notably less clean" than the paper's** — *"two or three somewhat
  overlapping bands (four or five bands total)."* He frames Qwen as "a different and weaker
  model [where] some results should differ." **Independent corroboration that the crisp
  mid-network band is model/family-dependent, not universal** — consistent with our
  Qwen≫Gemma result, and a caution that even *within* Qwen the bands are messier than
  Anthropic's Claude.
- **Poetry and arithmetic failed to replicate** on Qwen; **multihop-intermediate reversed**
  (answer-swap strictly dominated) because the Qwen dataset had linearly-related pairs like
  France/Paris — a methodological warning: **calibrate task difficulty per model** or you
  get artifacts. Multilingual and typo experiments *did* replicate.
- ⚠️ **Frontier-scale data point (gap since CLOSED, 2026-07-10):** J-Lens "seemed to do
  reasonably" on **Qwen3.5-397B-A17B** (~400B MoE) at n=4 prompts, **~1 hr on 8×H200s.**
  And he argues fitting is **cheap** — `O(n·d_model)` backward passes, n=1000 is
  overkill, **n=10–25 suffices**. *(Follow-through: we fit our own 397B lens at n=24 —
  mid_sep 0.343, the strongest band we measured; published with receipts. His "~1 hr"
  proved optimistic for the naive sharded path — see fit_our_own/results.md §5–6.)*
- **Novel positive extension:** abstract Chinese "interpretative meta-tokens" (什么意思 /
  "what does it mean") that appear and *causally act* during disambiguation — rare
  *algorithm*-level (not just variable-level) interpretability. He counts fast outside
  discovery of something new as validation that "J-Space is … a rich domain."

**His method cautions (adopt as our null hypotheses):** J-Lens is a noisy approximation
biased toward single-token concepts; causal interventions are less reliable than
observation (ablation removes only a fraction; negative steering also steers the error);
apparent broadcast may be an artifact of selecting high-internal-effect vectors;
multilingual results may be a norm/similarity artifact; model-organism results may not
generalize. "Nothing canonical about J-Lens" — SAEs/probes are alternatives.

## What this changes for our audit (action items)

1. ✅ **RESOLVED (2026-07-10) — and Eleos was right.** The shared-probe re-sweep
   (restricting CKA to the 4096 token strings shared by all tokenizers, exactly as
   suggested here) showed the Qwen≫Gemma headline was **mostly tokenizer artifact**:
   base-family means went 6.2× → 1.4×, and gemma-3-27b became the strongest band in the
   sweep (0.298). The surviving cross-family finding is the *function* gate
   (concept-resolution: all 8 behavioral Gemmas at 0.000 regardless of band). *This
   deep-read call was the single most consequential methods decision in the audit.*
2. **Nanda's "less clean bands"** on Qwen 27B, and experiments that fail/reverse on a
   non-Claude model, read post-correction as band strength varying by recipe and scale
   (not the dramatic family split our own-vocab numbers implied). Within-Qwen messiness
   remains a caution for *our* Qwen numbers.
3. ✅ **Frontier fitting DONE** — 397B fit at n=24, mid_sep 0.343, lens published with
   receipts; the transient question is settled (band grows to 0.4T). Nanda's "~1 hr"
   was optimistic for the naive sharded path (fit_our_own/results.md §5–6).
4. **Both moderate the GWT framing** → supports a **calibrated** audit conclusion: the
   mechanistic finding (a working-memory/cognitive space) is real and replicable; the
   *universal global-workspace* reading is not established (Eleos: only the weakest tier;
   Nanda: won't endorse it).
5. **Methods to reuse** (Nanda's recipe): Jacobians to the penultimate layer, ~25×128-token
   Pile/wikitext prompts, drop first 4 tokens; diagnostics = verbal-report Spearman vs a
   95% null band + layerwise CKA; expect signal only in mid-late layers.
6. **New dimension to at least acknowledge:** moral status / model welfare (Eleos) — out of
   our current scope, but the base-vs-instruct "persona/self" framing is worth a sentence.

## Sources & researchers

- The PDF (above). Contributors added to
  [`shared/researchers/handles.yaml`](../../shared/researchers/handles.yaml): Neel Nanda
  (DeepMind), Patrick Butlin / Robert Long / Derek Shiller / Dillon Plunkett (Eleos /
  Rethink Priorities), Camila Blank & Agam Bhatia (Nanda's replication team).
- Butlin, Long et al., *Consciousness in Artificial Intelligence* (2023).
