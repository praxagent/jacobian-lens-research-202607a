# TODO — beef plan for the J-space audit (approved by TJ 2026-07-09)

_The gap: we measure geometry, not behavior. These three upgrades convert the post from
"solid replication" to "definitive independent study." Statuses updated as work lands;
session state in `/checkpoint.md` (gitignored)._

## 1. Behavioral battery per family — **IN PROGRESS** (~$10–30 GPU)
Run Anthropic's released causal experiments (`anthropics/jacobian-lens
data/experiments/`: verbal-report, probe-swap, directed modulation/top-down-summoning;
methods per their README + Nanda's replication recipe) on the **killer triple**:
- `qwen3-4b` (strong band, from-scratch)
- `gemma-2-9b` (bandless, KD-pretrained)
- `gemma-2-27b` (banded, from-scratch — same architecture/norms/tokenizer as 9b)

**Payoff:** if causal workspace behavior (swaps steer answers, reportability) tracks the
band — present in qwen + gemma-2-27b, weak/absent in gemma-2-9b — the KD finding becomes
*functional*, not geometric. If behavior is intact in gemma-2-9b despite no band, our
`mid_sep` misses real workspaces (equally important, changes the post's headline).
Either outcome is publishable; we predict band↔behavior tracking but commit to reporting
the miss.
Steps: [x] plan  [ ] runner (`experiments/behavioral/`)  [ ] CPU smoke on gpt2
[ ] pod run on the triple  [ ] results.md + post section.

## 2. Ignition test on open models — **QUEUED** (~$5–15, shares pod with #1)
Dehaene & Naccache's decisive GWT signature: graded-strength stimuli → is J-space entry
threshold-like/all-or-none (ignition) or monotonic? Anthropic shipped `ignition.json`
but only partially addressed it (late §4.1.1). Nobody has run it independently, on any
open model. Run on the triple (band-havers vs bandless — does ignition exist only where
the band does?). A genuinely novel experimental contribution, not just audit.

## 3. Frontier lens fit — **QUEUED** (~$30–60, one run)
Fit a lens on **Qwen3.5-397B-A17B** (Nanda: ~1 h on 8×H200s, n=4–25 prompts) and compute
our band statistic at 0.4T scale. Converts the "sub-70B transient" limitation into a
measured point. Secondary candidate if weights/hosting awkward: biggest available
DeepSeek/GLM open MoE.

## 4. Free CPU tests (close mechanism holes) — **QUEUED**
- T1/T2: global-attention-layer-only CKA (per-layer maps in `hypotheses.md`) — does
  Gemma's residual band live in its global layers?
- T3: homogeneity-robust band statistics (spectral gap / contrast-normalized) — answers
  the H5 self-critique that `mid_sep` under-credits Gemma's uniform geometry.
- French lens (`--corpus wikipedia-fr`) when a pod is next up — second language pair.

## Landing today from the box (not new work)
70B anchor · shared-vocab confound verdict · precision A/B · final figure → fill the
four `[PENDING-*]` slots in `blog/jspace-audit/index.md`.
