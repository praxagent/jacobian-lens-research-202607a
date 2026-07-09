# Game plan — 397B J-lens replication, round 2 (2026-07-09)

_Product of a 5-agent post-mortem investigation (Nanda's primary sources; jlens source
cost model; our receipts; transformers TP internals; synthesis). TJ's mandate:
replication is a first-class product — recreate Nanda's finding properly. Cost rules
apply (CLAUDE.md 5, 7–10): validate cheap, measure throughput before spending._

## 1. Diagnosis — why Nanda got ~1 h (n=4) where we burned 2.3 h (n=2, unfinished)

jlens.fit cost (fitting.py:152): **traversals = n_prompts × ceil(d_model / dim_batch)**;
one forward per prompt (replicated dim_batch× in batch), then that many retained-graph
backwards. Backward FLOPs are dim_batch-*invariant* — dim_batch converts spare VRAM into
*fewer graph traversals* (less launch/routing/eager overhead per row of J).

| Config | Traversals | Implied s/traversal |
|---|---|---|
| Ours: n=2, db=2 (2.3 h, unfinished) | 4,096 | ~2–3.3 |
| Nanda: n=4, ~1 h, finished — if db=16 | 1,024 | **3.5** |

**Same per-traversal time.** Nanda's hour needs no exotic parallelism — device_map-style
sequential sharding with a sane dim_batch reproduces it. We ran the worst point on the
curve (db=2 → 2,048 traversals/prompt of overhead). His actual parallelism/db/seq_len/
layers are **undisclosed** (LW post + commentary PDF say only: n=4, ~1 h, 8×H200, "didn't
sanity check very hard"); anything beyond "consistent with db≈16–32" is inference.

Non-factors: penultimate-target (conditioning trick, ~1.7% of span); layer subsetting
(only helps by raising **min(source_layers)** — the graph is rooted there; a sparse subset
including an early layer saves ~nothing); TP (possible but unnecessary to explain 1 h).

## 2. What the TP failure actually was (corrects our first write-up)

- transformers 5.13 **does** ship a `base_model_tp_plan` for qwen3_5_moe (linear-attn
  entries fixed upstream in PR #47041 — those were fine).
- The break: **full-attention** `k_proj`/`v_proj` marked `colwise`; this checkpoint has
  **2 KV heads** (k width 512). Colwise is head-count-blind → at ws=8 each rank holds 64
  features = ¼ head → `view(2,128,-1,256)` at modeling line 692 fails (16384 ≠ k·65536).
- Shipped plan needs **ws ≤ 2** (KV heads) but the 807 GB model needs **ws ≥ 6** (memory)
  → it can never run this checkpoint. Upstream treats the identical gpt-oss-120b case as
  by-design (#40953). Supported route: `from_pretrained(tp_plan=<dict>)` **replaces** the
  class plan; omitted modules stay replicated.
- 5.13's TP is NOT DTensor at runtime — plain sliced tensors + hand-written autograd
  collectives; retain_graph repeated-backward works if all ranks run backwards in
  lockstep (jlens does). Known torch-DTensor bugs (#157606 etc.) don't apply.

## 3. The validated custom tp_plan (CPU-proven, $0)

Validated on this box (`tp_smoke_cpu.py`: tiny qwen3_5_moe in the failing 1-KV-head
regime, 2-proc gloo): shipped plan reproduces the exact reshape error; custom plan gives
**forward parity 4.2e-7** vs single-process and **retain_graph double-backward grad-norm
parity**. The plan (keys carry the `model.language_model.` prefix for the CondGen wrapper):

```python
TP_PLAN_397B = {
  # full attention: weight sharded 8-way for MEMORY, output all-gathered so the module
  # computes config-shaped attention redundantly (KV-head-safe at any world size)
  "model.language_model.layers.*.self_attn.q_proj": "colwise_gather_output",
  "model.language_model.layers.*.self_attn.k_proj": "colwise_gather_output",
  "model.language_model.layers.*.self_attn.v_proj": "colwise_gather_output",
  "model.language_model.layers.*.self_attn.o_proj": "colwise_gather_output",
  # linear attention: keep the shipped (working) entries
  "model.language_model.layers.*.linear_attn.in_proj_qkv": "colwise_gather_output",
  "model.language_model.layers.*.linear_attn.in_proj_z":   "colwise_gather_output",
  "model.language_model.layers.*.linear_attn.in_proj_b":   "colwise_gather_output",
  "model.language_model.layers.*.linear_attn.in_proj_a":   "colwise_gather_output",
  "model.language_model.layers.*.linear_attn.out_proj":    "colwise_gather_output",
  # the MoE experts — 96% of params, the actual 8-way COMPUTE win
  "model.language_model.layers.*.mlp.experts.gate_up_proj": "packed_colwise",
  "model.language_model.layers.*.mlp.experts.down_proj":    "rowwise",
  "model.language_model.layers.*.mlp.experts":              "moe_tp_experts",
  "model.language_model.layers.*.mlp.shared_expert.gate_proj": "colwise",
  "model.language_model.layers.*.mlp.shared_expert.up_proj":   "colwise",
  "model.language_model.layers.*.mlp.shared_expert.down_proj": "rowwise",
  "lm_head": "colwise_gather_output",
}
```

## 4. Ranked paths

- **B (primary): custom tp_plan** — experts sharded 8-way → est. 5–7× over device_map;
  n=24 ≈ 1–2 h ≈ **$35–70**. P(success) ~80–85%. Kills: CondGen wrapper TP load
  (vision/rotary meta-init unexercised), per-rank memory (~112–120/141 GB before
  activations), NCCL expert path at 60-layer scale.
- **A (mandatory fallback): device_map + dim_batch 32–64** — zero new code, correctness
  already gated. n=24 → 128 traversals at db=32; est. **$90–245** (n=16: $60–165).
  P(complete) ~85%. Kill: retained-graph OOM at high db (memory ∝ db — measure first).
- **C: hybrid** (TP + tuned db) — free once B works; tune on the pod.
- Dead ends, don't revisit: vLLM (no backward), waiting for upstream (#40953 by-design).

## 5. The sequence (gates + costs)

1. **Local, $0 (~2 h):** wire `TP_PLAN_397B` into tp_fit.py; add `--dim-batch/--source-layers/
   --checkpoint-path` plumbing; guard the resume footgun (checkpoint doesn't validate
   max_seq_len across resume — assert it); re-run tp_smoke_cpu.py green.
   **Gate: CPU parity holds.**
2. **Cheap pod ≤$10 (2–3 h):** 2×GPU — reproduce shipped-plan failure under NCCL on
   qwen3.5-0.8b; custom plan → mid_sep must match single-GPU **0.1443**; dim_batch sweep
   (s/traversal + graph GB at db=2/8/32/64). **Gate: numerics match or STOP.**
3. **Optional ≤$15:** 4-GPU `moe_tp_experts` smoke on a small qwen3_5_moe checkpoint if
   one exists on the Hub; else accept the residual risk knowingly.
4. **H200 session** (poll-acquire; ~$9 re-download): **first 45 min = measurement only**
   (~$26). TP load of the CondGen wrapper; if load fails → ≤15 min triage → fall back to
   device_map+db=32 (known-good). Time 3 traversals at candidate db.
   **Go/no-go: projected n=24 ≤ ~3.5 h (~$125) → GO; else n=16; else terminate.**
   No debugging on this pod beyond the 15-min triage.
5. **Fit** (checkpoint_every=1; eager attn required on the device_map path — dense TP
   passed without it). Band stat at the n=16 checkpoint; extend to n=24 if in budget.
6. **Validate + write up:** appendix-A.6-style evals on the lens (Nanda skipped this —
   we won't); band vs the 27B reference; file the upstream issue (shipped qwen3_5* plan
   unusable at any ws for 2-KV-head checkpoints).

**Cost: most-likely ~$130–150, hard cap $180 for the H200 session.**

## 6. Blog framing (either outcome)

State the reconciliation as inference (labeled); publicly correct our own two
misattributions (full-attn not linear-attn; plan-exists-but-unusable); publish the plan
dict + CPU repro; if the fit lands, report the band with full config + evals and note
Nanda's own n=4 no-sanity-check asterisk; if it dies, publish the throughput receipts and
the shipped-plan finding — independently useful either way. That's the cross-verification
mission.
