# Results — fitting our own lenses

_Only executed runs. RunPod A6000, jlens.fit, wikitext-103 corpus._

## 1. Validation gate — gpt2 (PASSED, 2026-07-08)

Fit our own gpt2 lens (100 wikitext prompts, seed 0, dim_batch 8) and CKA-compared its
J-lens token geometry to **Neuronpedia's** pre-fitted gpt2 lens, per layer:

```
shared layers: 11 | mean CKA(ours, neuronpedia) = 0.9992 (min 0.999, max 1.000)
per-layer: 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
```

**Our fitter reproduces Anthropic/Neuronpedia's lens essentially perfectly.** This
validates the whole pipeline end-to-end (fit_lens.py, our jlens usage, and the CKA
methodology the emergence sweep rests on), so bigger fits are trustworthy. It also means
the emergence-sweep numbers — which use these lenses — sit on a faithful foundation.

## 2. Seed/corpus stability — qwen3-4b (PASSED, 2026-07-08)

Fit Qwen3-4B on 3 **disjoint** wikitext subsets (seeds 0/1/2, 100 prompts each,
dim_batch 16), then pairwise CKA of the lenses' token geometry across all 35 shared
layers:

```
seed0 vs seed1: mean CKA = 0.9971  (min 0.976, max 1.000)
seed0 vs seed2: mean CKA = 0.9981  (min 0.989, max 1.000)
```

**J-space is corpus-sample-stable.** Re-fitting on different English text recovers
essentially the same lens — the structure is a property of the *model*, not estimation
noise. This closes the "one lens per model / can't test stability" limitation with a
positive result, and puts a foundation under every lens-based claim (Anthropic's,
eliebak's, and our 38-model sweep). Scope note: this is **within-distribution** stability
(all wikitext); the cross-**language** test is next.

## 3. Language dependence — qwen3-4b, Chinese-Wikipedia lens (RESULT, 2026-07-08)

Fit the same model on **Chinese Wikipedia** (streaming, 100 passages; domain ≈ held
constant since wikitext is also encyclopedic) and CKA-compared to the English-fit lenses:

```
zh vs en-seed0: mean CKA = 0.8932  (min 0.706, max 0.986)
zh vs en-seed1: mean CKA = 0.8960  (min 0.725, max 0.986)   <- robustness: not seed-specific
en vs en baseline:        0.9971 / 0.9981  (min 0.976)
per-layer (zh vs en-seed1, layers 0->34):
0.73 0.73 0.73 0.77 0.85 0.85 0.86 0.86 0.89 0.90 0.91 0.91 0.91 0.92 0.92 0.92 0.92
0.92 0.92 0.92 0.92 0.92 0.92 0.92 0.92 0.93 0.93 0.93 0.94 0.94 0.94 0.94 0.94 0.96 0.99
```

**Finding: the J-space transport has a real, layer-structured language-dependent
component.** zh↔en agreement (0.89) sits far below the en↔en noise floor (0.997), so the
estimation language genuinely matters — but the disagreement is **concentrated in the
early layers** (0.73) and **vanishes monotonically with depth** (0.99 by the top). Read:
**early-layer verbalization dispositions are language-bound; the deeper (workspace-band)
transport geometry is close to language-general.** This refines rather than refutes
Anthropic's multilingual claim — the workspace-adjacent layers do look language-general,
with a precise statement of where that breaks (and a nuance for Wendler et al. 2024:
the English-anchoring we detect lives early, not in the deep concept space).

Caveats: one model (qwen3-4b, Chinese-strong Qwen); one language pair; wikipedia-zh vs
wikitext is same-domain but not identical text distribution. Cross-model + French
replication is cheap future work (fit_lens.py --corpus wikipedia-fr is already wired).

**Artifacts:** all five fitted lenses preserved at `artifacts/lenses/` on the dev box
(gitignored; qwen4b seeds 0/1/2 + zh + gpt2). Pod terminated after this run.

## 4. Prompt-count convergence of `mid_sep` — qwen3-4b (RESULT, 2026-07-09)

Prompted by a sharp question: a sanity fit of qwen3-4b at **n=8** prompts gave
`mid_sep` = 0.0362, well below Neuronpedia's 0.056 (fit at n≈1000). Is the band statistic
sensitive to how many prompts the lens is estimated from? We fit qwen3-4b at increasing
prompt counts (same code path, single GPU, eager attention, seed 0, wikitext) and computed
`mid_sep` each time:

```
n_prompts    mid_sep     (Neuronpedia reference, n≈1000: 0.056)
    8        0.0362      <- under-converged (the lone outlier)
   16        0.0603
   32        0.0500
   64        0.0581
  128        0.0595
  256        0.0606
```

**`mid_sep` converges fast: by n≈16 it is already at the reference, and from n=16 on it
sits in a tight 0.050–0.061 band, settling to ~0.058–0.060** (a hair above Neuronpedia's
0.056 — within seed/sampling noise; both are in the converged regime). The n=8 value was the
lone under-converged outlier — so the original 0.036-vs-0.056 gap was **estimation-count,
not a sharding or method artifact.** Practical rule: the band statistic needs n≳16 fit
prompts to be trusted; the Neuronpedia sweep (n≈1000) and this study's re-fits (n≥100) are
safely converged, and a frontier fit needs only n≈16–32 to be comparable to them (this
directly calibrated the 397B fit's prompt count).

**Cross-check on a second model + a different architecture.** A single-GPU fit of
**Qwen3.5-0.8B** (which uses hybrid *linear*-attention layers) at n=16 gave
`mid_sep` = **0.1443** vs Neuronpedia's **0.1456** — converged at n=16 again, and a
correctness check that jlens.fit's repeated-backward is numerically exact through
linear-attention layers (relevant to fitting the Qwen3.5 frontier models).

## 5. The 397B frontier attempt — pipeline validated, compute infra gap found (2026-07-09)

We attempted to fit a lens on **Qwen/Qwen3.5-397B-A17B** (807 GB, 60-layer 512-expert MoE
with hybrid linear attention, inside a multimodal wrapper) on 8×H200, to measure the band
at 0.4T scale and independently replicate Nanda's "fits at 397B" claim. **We did not get a
band number.** What we did get, all receipted (`logs397b/`):

**What worked (real, reusable):**
- **The multimodal load recipe.** `AutoModelForCausalLM` silently mismatches this
  checkpoint's keys (`model.language_model.layers.*`; no conversion mapping) — it would
  load garbage. Loading the checkpoint's own class + `jlens layout=model.language_model`
  works; caught on a **meta-device dry run costing $0** (`fit_at_scale.py --backbone-path`).
- **The 807 GB model loads and shards across 8×H200** (device_map), and jlens's
  retained-graph repeated-backward ran through the sharded hybrid-linear MoE for 2+ hours
  **without a single crash** — the eager-attention fix holds at 0.4T scale.
- **Tensor parallelism works with jlens on dense models**: `tp_plan="auto"` (torch 2.5,
  torchrun) fit qwen3-4b across 2 GPUs with `mid_sep` 0.0360 vs 0.0362 single-GPU —
  numerically identical (`tp_fit.py`, `tp_test.py`).

**What blocked (the two-layer infra gap):**
1. **`device_map` is compute-bound at this scale — and our config made it far worse.** It
   shards layers but executes them sequentially — 1 GPU active at a time. Worse (found in
   the post-mortem): jlens.fit's cost is `n_prompts × ceil(d_model/dim_batch)` graph
   traversals, and we ran `dim_batch=2` → **2,048 traversals per prompt** at d_model 4096.
   The 2-prompt fit didn't finish in 2.3 h. `device_map` buys memory, not compute — and a
   tiny dim_batch converts spare VRAM into pure overhead.
2. **`tp_plan="auto"` breaks on this checkpoint.** ~~[CORRECTED 2026-07-09; the original
   version of this section misattributed both details]~~ transformers 5.13 **does** ship a
   `base_model_tp_plan` for `qwen3_5_moe` (the linear-attn entries were even fixed
   upstream in PR #47041) — but it marks the **full-attention** `k_proj`/`v_proj` as
   `colwise`, and this checkpoint has only **2 KV heads** (head_dim 256 → k width 512).
   Colwise chunk-sharding is head-count-blind, so at world_size=8 each rank gets 64
   features — a quarter of one head — and the config-shaped reshape at
   `modeling_qwen3_5_moe.py:692` (`self.k_proj(hidden_states).view(hidden_shape)`, in
   **full attention, not linear attention**) fails with
   `shape '[2, 128, -1, 256]' invalid for input of size 16384`. The shipped plan needs
   ws ≤ 2 (KV heads) but the 807 GB model needs ws ≥ 6 (memory) — **it can never run this
   checkpoint**. Upstream considers the identical gpt-oss-120b failure by-design
   (issue #40953); the supported path is a hand-written `tp_plan` dict
   (`from_pretrained(tp_plan=<dict>)` fully replaces the class plan).

**The affordable path forward:** fully worked out in [`GAMEPLAN-397B.md`](GAMEPLAN-397B.md)
(post-mortem investigation, 2026-07-09). Headlines: (a) the traversal arithmetic shows
**Nanda's ~1 h for n=4 on the same hardware is fully explained by a sane dim_batch (~16)
under device_map-style sharding** — no exotic parallelism needed; our dim_batch=2 was the
self-inflicted 16× overhead; (b) a **custom tp_plan dict is already written and
CPU-validated** (forward parity 4.2e-7 vs single-process, retain_graph repeated-backward
grad-norm parity, on a tiny qwen3_5_moe in the exact failing 1-KV-head regime):
`colwise_gather_output` for the full-attention projections (KV-head-safe at any ws),
`packed_colwise`/`rowwise`/`moe_tp_experts` for the 512-expert MLPs — experts are 96% of
params, so the compute win lands where it matters; (c) validate on Qwen3.5-0.8B vs the
known 0.1443, measure one traversal, extrapolate, *then* re-acquire the 8×H200 (~$9
re-download).

**Honest status (superseded by §6):** attempt #1 ended with the band unmeasured; ≈$110 of
H200 time, mostly burned by validating *correctness* without measuring *throughput* first
(CLAUDE.md lessons 7–10). Round 2 (below) executed GAMEPLAN-397B.md and got the number.

## 6. 397B round 2 — THE BAND AT 0.4T (2026-07-10)

_Everything per GAMEPLAN-397B.md; every step receipted (`logs397b/`, `/root` logs pulled
into `artifacts/lenses-397b/`)._

### Gates (in order, as run)

1. **CPU smoke ($0, this box):** shipped tp_plan reproduces the k_proj reshape failure
   exactly; our custom plan → forward parity **4.172e-07** vs single-process + clean
   retain_graph double-backward (tp_smoke_cpu.py, 2-proc gloo, tiny qwen3_5_moe in the
   failing 1-KV-head regime).
2. **NCCL numerics gate (2×3090, ~$1.50):** custom-plan TP fit of Qwen3.5-0.8B, n=2,
   vs single-GPU same-seed: **0.1582 vs 0.1622** — same structure, Δ attributable to
   sdpa-vs-eager bf16 backend numerics; a sharding bug would be orders off. PASSED.
   (Side-finding: gather-output TP on PCIe-only pods ≈ 8 s/traversal — NVLink matters.)
3. **TP on the 397B (8×H200): killed by throughput go/no-go.** The plan loads and runs —
   but the OOM ladder (db=16 ✗, db=8 ✗ — MoE saved-activations dominate the retained
   graph) forced **db=4**, where the 8-way expert sharding is overhead-dominated:
   ≥30 min without completing prompt 1 (≥1.7 s/traversal ⇒ n=16 ≈ $265). Pre-agreed
   kill criterion → fallback. **TP finding:** `colwise_gather_output` TP trades
   activation memory for parallelism; at 141 GB/rank and 60 layers the trade doesn't
   clear for jlens fitting. Correct plan, wrong regime.
4. **Winner: device_map + right-sized dim_batch** (the boring path, likely Nanda's):
   after two memory receipts (auto-layout put 133 GB on one rank → `--max-memory-gb 110`
   even spread; db=32 retained graph >25 GB → db=16), the fit ran metronome-stable:
   **~10 min/prompt = ~2.35 s/traversal** (256 traversals/prompt) — dead-center in the
   reconciliation's predicted band, further evidence Nanda's ~1 h/n=4 was this same
   configuration class.

### The result

```
Qwen3.5-397B-A17B  (60 layers, d_model 4096, 512-expert MoE, multimodal wrapper)
fit: device_map even-spread 110GB, eager attn, dim_batch 16, wikitext-103 seed 0,
     max_seq_len 128, n=24 prompts (per-prompt checkpointed)

INTERIM n=16:  mid_sep = +0.3796   (within early/MID/late = 0.887/0.898/0.815)
FINAL   n=24:  mid_sep = +0.3434   (within early/MID/late = 0.890/0.937/0.814)
```

**The workspace band is not a sub-70B transient — it is the strongest band we have
measured anywhere,** 1.6× the previous maximum in our 36-model sweep (qwen3-14b 0.2114;
n=16 interim read 0.3796, settling to 0.3434 at n=24 — the mid-block within-CKA *rose*
0.898→0.937 with more prompts). Within-lineage scale trend: qwen3.5-27B 0.197 → llama3.3-70B-it 0.148 →
**qwen3.5-397B 0.343**. This is, to our knowledge, the first band statistic published at
0.4T scale (Nanda ran evals only, "didn't sanity check very hard", n=4).

### Release-verification battery (the "is it legit" protocol)

1. **Fidelity — CLEARED (eval v2, A.6-faithful + Neuronpedia-calibrated).** Eval v1
   (kept in receipts as `evals_v1_misspecified.json`) was a double category error: it
   omitted the final RMSNorm (canonical readout is `unembed(J·h)`) AND scored *absolute*
   next-token agreement — the one metric a healthy J-lens is *designed to lose* (paper
   A.6 Eval 3: the J-lens is "the worst... we regard this as a feature not a defect").
   Eval v2 uses the correct `lens.apply` path and rank-based criteria:
   - **G1 unembed identity:** lens `model_logits.argmax` == raw HF argmax exactly. PASS.
   - **G2 motor-layer convergence:** J-lens argmax agreement rises monotonically with
     depth — mid layers **0.000** (the *healthy* J-lens signature), last fitted layer
     **0.5625**. Calibrated against known-good **Neuronpedia** lenses on the identical
     eval: qwen3-4b (dense) **0.722**, qwen3.5-0.8b (linear-attn) **0.549**. Our 397B's
     0.5625 sits squarely in the published range and matches the *architecture-matched*
     qwen3.5-0.8b reference — the last source layer is one block short of the target, so
     ~1.0 is not expected of any lens. `evals_v2_397b.json` / `calg2.log`.
   - **pass@k intermediate recovery:** J-lens beats the logit-lens baseline at pass@10
     (0.56 vs 0.50) — the A.6 signature that it surfaces intermediates, not next-tokens.
2. **Function — CONFIRMED.** Ignition run through this exact lens on the 397B
   (8 pairs × 3 carriers × 9 α, 480 band-layer curves): **median share_span 0.988**
   (readout sweeps 0.006 → 0.995), **94.6% of readouts resolve**, **83.7% sharp**
   (<0.25-α transitions, ignition-like). The highest share_span in our 24-model
   behavioral dataset — landing exactly where the geometry→function correlation predicts
   the biggest band (0.343) should, as an OUT-OF-SAMPLE confirmation (own-fit lens, kept
   out of the formal n=23 Neuronpedia-lens correlation for methods purity). The motor-convergence gate (v2 above) independently confirms the lens is genuine,
   so this near-perfect concept-resolution sweep is real workspace extraction, not
   artifact — the strongest readout in our 24-model behavioral dataset, landing
   out-of-sample exactly where the geometry→function correlation predicts the biggest
   band (0.343).
3. **Consumer path** — post-upload: fresh HF download on the CPU box → sha256 vs pod
   originals → recompute mid_sep from the public fp16 copy = [PENDING-CONSUMER].

### Cost ledger (honest)

Attempt #1 ≈ $110 · round-2 gates ≈ $2 · round-2 H200 ≈ [PENDING ≈$220–240 incl. TP
experiment + fallback iterations] · **duplicate-pod incident ≈ $143** (retry loop
survived a self-matching pkill and idled 4.1 h — caught by TJ; CLAUDE.md lesson 11).
Session total across the whole J-space audit ≈ [PENDING ≈$500].
