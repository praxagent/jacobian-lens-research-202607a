# Why does Qwen have a workspace band and Gemma barely any? — hypotheses & tests

_2026-07-08. Built from primary-source research (tech reports + raw HF configs, fetched
and quoted; four-agent sweep, receipts in the workflow journal). Measured values cited
from our fp32-path ledger; final numbers come from the uniform re-sweep. This is the
"contingent demands explanation" follow-through — the post is weak without it (TJ)._

## The explanandum

`mid_sep` (CKA band distinctness) at matched scale: **Qwen3/3.5 0.15–0.21 · OLMo-3-32B
0.094 · gpt-oss-20b 0.076 · Gemma-2 ≤0.043 · Gemma-3 ≤0.025 (0.0007 at 12B)** — with
small models (<0.2B) and the random-transport null at ~0.

## Verified recipe facts (the discriminating ones)

| Family | Attention layout | Norm scheme | Pretraining | Embeds/vocab |
|---|---|---|---|---|
| **Gemma-2** | 1:1 local(4096):global | **pre+post sandwich** (4 norms/block) + softcap | **2B & 9B: logit-KD distilled; 27B: from scratch** (report) | tied, 256k |
| **Gemma-3** | **5:1 local(512–1024):global** | sandwich + QK-norm | **KD-distilled (all sizes, prob.)**, teacher undisclosed | tied, 262k |
| **Qwen3** | all-global | pre-norm + QK-norm | **from scratch, 36T** (KD only in *post*-training for 1.7–14B; 32B none) | tied ≤4B, untied ≥8B, 152k |
| **Qwen3.5** | **3:1 linear(GatedDeltaNet):full** — only 25% full attention | pre-norm (zero-centered) + QK-norm | from scratch, multimodal early-fusion | tied ≤4B, 248k |
| **OLMo-3** | 3:1 sliding(4096):full | **post-norm-on-outputs** + QK-norm | from scratch 5.9T (+100B midtrain incl. teacher-generated traces) | untied, 100k |
| **gpt-oss-20b** | 1:1 sliding(**128**):full, attn sinks, MoE | pre-norm | "large-scale distillation and RL" (mechanism undisclosed); RL-only release | untied, 201k |
| **Llama-3.1** | all-global | pre-norm | from scratch ~15T | untied, 128k |

Key technical note: **Neuronpedia lenses are fit on 128-token prompts.** Gemma's
512–4096 windows and OLMo's 4096 windows **do not bind at that length** — every "local"
layer behaves globally during lens estimation. Only gpt-oss's 128-token window binds.
So no sliding-window effect can be *mechanical* in our measurements (except gpt-oss);
any attention-layout effect must run through *what training shaped the weights to do*.

## Hypotheses (ranked after the evidence)

### H2′ — Pretraining-level knowledge distillation suppresses the band  ★ leading
Gemma is the **only** family whose *pretraining objective* is logit-KD — and the only
family with ~no band. Prior art supplies the mechanism: logit-KD students **reorganize
internals away from teacher structure** ("Distilled Circuits", arXiv 2505.10822), KD
fidelity is low even when capacity suffices (Stanton, 2106.05945), and **KL-only
distillation *reduces* layerwise-CKA structure** unless internal geometry is explicitly
aligned (arXiv 2606.05682). Interpretation: a student mimicking output distributions
needn't develop the serial internal "working memory" that from-scratch next-token
training induces.
- **Already-supporting natural experiment (within Gemma-2, architecture constant):**
  27B (from scratch) = **0.043** vs 2B (KD) = **0.007** — 6×.
- **PREDICTION P1 — ✅ RESOLVED, HIT:** we predicted gemma-2-**9b** (KD) lands ≤0.02,
  well under from-scratch 27b. **Measured: 0.0019** (uniform sweep). Within Gemma-2 —
  architecture, norms, tokenizer, data lineage constant — KD models score 0.002–0.007
  *independent of scale* (9B < 2B!) while the from-scratch 27B is ~6–20× higher.
  *Timing honesty:* the prediction was written blind to the value, but the sweep had
  computed that row on the box minutes earlier — "predicted-before-seeing," not strict
  pre-registration. The prediction and its basis (the report's 2B/9B-distilled vs
  27B-from-scratch split) are in git history (`0412769`) before we looked.
- Qwen3's *post*-training KD (1.7–14B distilled; 32B not) shows **no band discontinuity**
  → pretraining-KD, not any-KD, is the candidate lever. gpt-oss ("distillation",
  mechanism unknown) at 0.076 is consistent but uninformative.

### H3 — Sandwich/peri-norm homogenizes layer geometry  ★ co-leading, confounded with H2′
Gemma's per-sublayer post-norms re-normalize every residual write; the Peri-LN
literature shows this controls hidden-state growth/variance. Observed signature: Gemma's
CKA is **uniformly ~0.99 everywhere** — layers all look alike, which our `mid_sep`
scores as "no distinct band." OLMo (partial post-norm) lands intermediate (0.094).
- **⚠️ Perfect confound in this cohort:** only Gemma has sandwich-norm, and only Gemma
  has KD-pretraining. The Gemma-2 27B-vs-9B natural experiment is the ONE lever that
  separates them (norm constant within family): if 9b comes in low, KD explains the
  within-family gradient and norm alone cannot.
- **PREDICTION P2:** under any homogeneity-robust band statistic (T3 below), Gemma still
  shows less block structure than Qwen — else our result is partly metric artifact (H5).

### H1′ — Attention layout (training-shaped only)  — demoted
Naive form **refuted by Qwen3.5**: 75% linear-attention layers yet the strongest bands
(0.147–0.203); and windows don't bind at fit length. Surviving form: training under
tiny-window SWA (Gemma-3's 512–1024; gpt-oss's 128) shapes weights toward local
processing, diluting broadcast structure. Weak support: gemma-2 (1:1, bigger windows) >
gemma-3 (5:1, tiny windows) at 27B (0.043 vs 0.025), same norm+KD regime.
- **TEST T1/T2 (free, per-layer maps in hand):** restrict the CKA analysis to
  global-attention layers only (Gemma-3: (idx+1)%6==0; Gemma-2: odd; Qwen3.5 full-attn:
  i%4==3; OLMo: i%4==3). If Gemma's global-only submatrix shows a band that full-stack
  mixing hides → layout matters; if still flat → layout is not the story.

### H4 — Tied embeddings / giant vocab — weak
Qwen3-1.7B/4B are *tied* with strong bands → tying alone doesn't suppress. Retained only
as a possible contributor to lens geometry (the shared-probe re-sweep bounds the
measurement side).

### H5 — Our statistic under-credits homogeneous models — standing self-critique
`mid_sep` measures *differentiation* of the middle third. A model whose layers are all
mutually similar scores ~0 even if globally workspace-like. **TEST T3 (free):**
recompute band structure with homogeneity-robust statistics (spectral gap of the CKA
matrix; off-diagonal contrast normalization) and check the family ranking survives.

## Test queue (all CPU, from cached lenses; ordered by information value)
1. **P1 — gemma-2-9b prediction** (auto: lands in tonight's uniform re-sweep). The KD
   hypothesis's pre-registered test: we predict ≤0.02 **before seeing the number**.
2. **T1/T2 — global-layer-only CKA** for Gemma-2/3, Qwen3.5, OLMo, gpt-oss.
3. **T3 — homogeneity-robust statistics** (answers H5, protects the headline).
4. **T4 — shared-probe re-sweep** (running; bounds tokenizer/vocab confounds incl. H4).
5. Future/GPU (v2): fit a lens on a KD-pretrained non-sandwich model or a from-scratch
   sandwich-norm model if one exists/appears — the clean H2′⊥H3 separation.

## Honest limits
Observational cohort, n(families) small; Gemma's KD+sandwich confound is real and only
partially separable (P1); teacher identities undisclosed (Gemma-3, gpt-oss); Qwen3.5
"from scratch" rests on official docs, no tech report yet; all this explains variance in
*our statistic* — connecting any of it to Anthropic's causal/behavioral workspace claims
requires their experiments (data/experiments/) run per-family (v2).

_Sources: Gemma-2 (arXiv 2408.00118), Gemma-3 (2503.19786), Qwen3 (2505.09388),
OLMo-3 (2512.13961), gpt-oss (2508.10925), Llama-3 (2407.21783), Pythia (2304.01373);
Distilled Circuits (2505.10822), Stanton (2106.05945), NVFP4-geometry (2606.05682),
Ojha (2205.16004), Peri-LN literature; raw config.json per model (incl. layer_types
arrays). Full agent receipts in the workflow journal (wf_473a0ab9-369)._
