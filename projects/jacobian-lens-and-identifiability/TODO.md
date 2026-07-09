# TODO — MOONSHOT ($150 budget approved 2026-07-09)

TJ: fix the sharding bug (valuable in itself → own blog post), REPLICATE Nanda's 397B
claim (builds respect), fix the 3 behavioral failures. "Shoot for the moon."
LIVE: shard-fix pod 1wdh59m0886l28 (2×A6000 $0.98) — eager-attn fix, validate qwen3-4b
sharded→0.056. behav-retry pod 9lbonb7xuwvo0k (A100 $1.39) — rerun qwen3-14b/qwen3.5-27b/
gpt-oss with accelerate+xet-off. THEN 8×H200 for Qwen3.5-397B once sharding validated.
Budget: ~$50-70 of $150. Both pods TERMINATE on done.

# TODO — beef plan for the J-space audit (approved by TJ 2026-07-09)

_The gap: we measure geometry, not behavior. These three upgrades convert the post from
"solid replication" to "definitive independent study." Statuses updated as work lands;
session state in `/checkpoint.md` (gitignored)._

## 1+2 FOLDED — **RUNNING on pod w01s4uz8s2nqfi** (A6000 $0.49/hr, ssh -p 41556 root@38.147.83.32)
Both behavioral tests on the triple, one pod (saves money). Log `/root/behavioral.log`,
marker `BEHAVIORAL_DONE`. Watcher: task bril6oi33. ON DONE: pull `verbal_report_*.json`
+ `ignition_*.json` → git, write post section, **TERMINATE pod**
(`launch.py terminate --pod w01s4uz8s2nqfi`). HF_TOKEN passed inline (not on pod).

### STRONG RESULT emerging (2026-07-09, A100 batch ~8/13)
Two-level, maps to Eleos privileged-set < workspace:
- **Steerability (verbal-report swap): UNIVERSAL** — works ~0.84-0.91 in banded AND
  bandless (privileged-SET tier present everywhere).
- **Concept-propagation + ignition: TRACKS THE BAND.** share_span diagnostic:
  Qwen 0.96-0.98 (readout cleanly flips A→B), Gemma ≈0.0 (injected concept NEVER
  resolves in readout — real, not metric artifact), gpt2 0.90-but-gradual (ign_sharp 0).
  ign_sharp: qwen3-4b 0.91, qwen3.5-4b 0.75, qwen3.5-2b 0.27, gpt2 0.0, Gemma None.
  → the correlation (correlate.py) will quantify: ignition-sharpness/share-span +tracks
  mid_sep; swap flat. THIS is the frontier behavioral result.

### INTERIM RESULTS (2026-07-09, pod running)
- **Verbal-report causal SWAP does NOT discriminate band vs no-band:** qwen3-4b (band
  0.20) swap **91%**; gemma-2-9b (band **0.0019**) swap **84%**, reportability **0.71**.
  Both strong; strength-0 control 0/140 (clean). → the *privileged-set* tier (Eleos:
  reportable + steerable) is present even where the geometric band is absent. Honest,
  important: our `mid_sep` does NOT track this behavior.
- **Ignition DOES look discriminating so far:** qwen3-4b width **0.123, 93% sharp**
  (strongly all-or-none) vs gpt2 baseline 0.69/0% (gradual). **gemma-2-9b ignition =
  the decisive number, running now.** If gradual → clean story: swaps generic, but
  *ignition (all-or-none workspace entry) tracks the band*. If sharp → reframe: Gemma
  has a functional workspace that just isn't geometrically differentiated (H5).
- **gemma-2-27b FAILED (OOM: 27B bf16 ~54GB > A6000 48GB).** The from-scratch banded
  sibling — needs an A100-80GB rerun for both tests to complete the triple. Decide
  after gemma-2-9b ignition lands.
- This maps onto Eleos's privileged-SET < STREAM < WORKSPACE tiering — likely the
  post's behavioral framing.

## 1. Behavioral battery per family — **RUNNING (folded)** (~$10–30 GPU)
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
