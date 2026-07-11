# CLAUDE.md — research-and-replications

This is a **research** repo, separate from the Prax product harness. Different rules
apply here than in `prax/`. Read this before working.

## Cost discipline (the founder is self-funded — treat money as scarce)

GPU time is paid out of pocket. Optimize for the cheapest path that doesn't compromise
the research:

1. **CPU-first, always.** If an experiment can run on this box's CPU (synthetic data,
   small models, linear/analytic checks), it runs here — **free**. A CPU venv with
   `torch==2.4.1+cpu`, `numpy`, `scipy` already exists (`.venv/`). Prototype, debug, and
   validate the whole pipeline on CPU before touching a GPU.
2. **Never compromise the research to stay on CPU.** When a result genuinely requires a
   GPU — real-LLM experiments (J-lens), or training at a scale CPU can't reach — use
   one. Don't fake it, don't down-scope the science to dodge the cost; scope the *spend*
   instead (smallest GPU that fits, shortest run that answers the question).
3. **Terminate every RunPod pod the moment you're done.** A forgotten running pod bills
   by the second for nothing. The launcher (`shared/runpod/`) has `--terminate-on-done`
   (default on) and a `terminate` subcommand. After any GPU run, confirm the pod is gone
   (`python shared/runpod/launch.py pods`). Prefer on-demand/spot over persistent.
4. **Never start a paid pod without the user's explicit go-ahead** for that specific run.
   Estimate the cost first (GPU $/hr × expected minutes) and say it out loud.
   Full compute inventory, access, and the snapshot/teardown cost analysis live in
   [`shared/infra.md`](shared/infra.md).
5. **⚠️ Validate the code path on the CHEAPEST hardware that exercises it, BEFORE the
   expensive run.** Never debug on an expensive node. A bug found at $0.98/hr is free;
   the same bug on 8×H200 at ~$29/hr is money set on fire while you read a traceback.
   This has already caught two real bugs (an incomplete fp16 lens loader; an
   accelerate CPU-offload crash in the sharded fit) — each would have surfaced on the
   costly node otherwise. Concretely:
   - **Smoke every runner on CPU / a tiny model first** (e.g. gpt2) — proves imports,
     shapes, and control behavior. Do this before ANY GPU pod.
   - **Exercise the actual mechanism at minimum scale on the cheapest GPU that has it.**
     Multi-GPU sharding? Validate on 2×A6000 ($0.98/hr) forcing a small model to split —
     NOT on the 8×H200. Big fit? Prove the fit+readout pipeline on a small model first.
   - **Make the cheap test a CORRECTNESS check, not just a smoke test** where possible:
     reproduce a known number (e.g. our sharded qwen3-4b fit must land at Neuronpedia's
     mid_sep ≈ 0.056). A green run that produces the wrong number is still a caught bug.
   - **Mirror the expensive path exactly.** The cheap validation must use the *same code
     path* (same sharding mode, same precision, same dtype) — a validation that skips the
     risky part (e.g. tests CPU-offload when the real run is pure-GPU) validates nothing.
   - Only after the cheap test passes: launch the expensive node, and terminate it the
     instant the run (or a failure) is confirmed.
6. **Budget ceiling behavior (TJ, 2026-07-09): never kill a RUNNING experiment to stay
   under budget.** If cumulative spend nears the ceiling (currently $150/session),
   **stop launching anything NEW and bring it to TJ to discuss** — do not cut off live
   work mid-run. Terminating a pod the moment its job *completes* (to stop idle billing)
   is always correct and is NOT "killing an experiment" — that's ending waste. The rule
   only protects in-progress runs.
5. **Get everything ready on CPU so a GPU run is short.** Code, configs, a CPU smoke of
   the exact command, and results plumbing should all be done *before* the pod starts —
   so the paid pod only does the irreducibly-GPU part.

## Lessons learned the expensive way (2026-07-09, the 397B fit — ~$80 lesson)

7. **⚠️ Validate THROUGHPUT at scale, not just correctness.** The cheap-validation ladder
   (rule 5) proved the 397B pipeline *correct* (load, sharding, numerics) but never *timed*
   it — and a correctness-clean path can still be 10–100× too slow. A "quick n=2 smoke" on
   8×H200 ran 2.3h at $35/hr before being killed. Before any expensive run: **measure one
   unit of work on the cheap node** (one backward pass, one prompt) **and multiply** —
   walk away from the launch if the extrapolated wall-clock × $/hr isn't affordable.
8. **`device_map="auto"` buys MEMORY, not COMPUTE.** It shards whole layers and executes
   them *sequentially* — on an 8-GPU pod, 7 GPUs idle at any instant. It exists to make
   too-big models *fit*, not go fast. For big-model **fitting/training-style workloads, plan
   tensor parallelism from the start**: `tp_plan="auto"` (transformers ≥5.x native TP,
   needs **torch≥2.5**, `torchrun --nproc_per_node=N`). Validated: jlens.fit's
   retain_graph + repeated-backward survives TP with numerically identical results
   (`tp_fit.py` / `tp_test.py` in fit_our_own). vLLM is inference-only — no backward, not
   an option for lens fitting.
9. **Big-GPU supply is scarce — treat an acquired pod as an asset.** Getting an 8×H200
   took ~1h of poll-looping (`scratchpad/h200_retry.sh` pattern: retry a config list every
   90s). Before terminating a rare pod mid-investigation, weigh re-acquisition time + the
   re-download of any cached giant model against the idle $/hr of keeping it briefly.
   Corollary: **RunPod "stop" does NOT preserve the container disk** — only *volume* disk
   survives a stop, and our pods are created with `volumeInGb: 0`. Stopping ≠ caching.
10. **Long SSH commands to pods hang on detach** (nohup-launch then exit). Launch with a
   short `timeout`, expect exit 143, and **verify with a separate fresh SSH** — don't
   re-run the launch because the channel hung (double-launching a fit is an expensive bug).
11. **⚠️ `pkill -f <pattern>` SELF-MATCHES the shell that runs it** when the pattern
   appears in your own command line — the kill dies mid-command and the target SURVIVES.
   Cost us twice on 2026-07-09/10: an orphaned TP fit, and a retry-loop that outlived its
   "kill" and silently created a DUPLICATE 8×H200 pod (~$143 idle before caught). Rules:
   kill by exact PID (`pgrep -f pat | head` first, inspect, then `kill <pid>`), use the
   bracket trick (`pgrep -f "[h]200_retry"`), and after ANY create-capable loop runs,
   **audit `launch.py pods` and count RUNNING rows against what you expect** — a pod you
   don't recognize is money burning.

## Lessons learned the expensive way (2026-07-11, the thinking-on truncation — ~$30 lesson)

12. **⚠️ VALIDATE THE OUTPUTS ARE USABLE BEFORE YOU TERMINATE A WARM POD.** A run that
   reaches its "done" marker can still have *unusable* results. The thinking-on divergence
   batch finished cleanly, but every reasoning trace was truncated at `--continue-tokens 160`
   before the model committed an answer — the one thing the experiment needed. The pod was
   **still warm with the model loaded**; continuing those 12 generations would have cost ~$5.
   Instead it was terminated the instant the marker fired, so recovering the answers means
   re-paying the whole ~$25 model download. **New hard gate: before `terminate`, open the
   receipt and confirm the headline quantity is actually present and well-formed** (not just
   that the process exited 0). "The job finished" ≠ "the data is good." Teardown is the last
   step *after* validation, never coincident with the done-marker.
13. **Reasoning models need a generously sized generation window — and a stop condition.**
   A token budget that's ample for a non-reasoning model truncates a reasoning model
   mid-`<think>`. Qwen3.5 deliberates **hundreds of tokens** before `</think>` even on a
   trivial factual question under pressure. For any behavioral read of a reasoning model:
   size `max_new_tokens`/`continue-tokens` to *reach and pass* `</think>` (256–1024+, or
   generation-until-`</think>`), and **assert the committed answer parsed** (`</think>` seen
   AND a token after it) for every item before trusting the run. See lesson 12 — the assert
   is what would have caught this while the pod was still warm.
14. **Recovering a committed answer is generation-only and cheap — but still needs the model
   in memory.** Greedy decoding is deterministic, so re-generating (or resuming from saved
   `continuation_ids`) with a bigger window reproduces the exact answer a longer original run
   would have given. Use a **standalone generation script (no jlens, no control lenses — that
   setup alone is ~15 min/$8 on the 397B)**, **batch all prompts through one `generate` call**
   (batched greedy = same per-sequence answer, ~1 forward-pass-set instead of N), and verify
   determinism against the saved ids. The dominant cost is re-downloading the weights, which
   is only avoidable if they were cached on a **network volume** (container disk never
   survives teardown) — so for repeated behavioral work, download once to the volume.

## Lessons learned the expensive way (2026-07-11, the Llama-70B readout perf wall)

15. **⚠️ The lens TRANSPORT must run on GPU — naive CPU transport is pathological for a
   large-`d` lens.** `jlens.lens.apply` (and thus demo2's readout) keeps `lens.jacobians[l]`
   on CPU and does `residual @ J_l.T` there. For the d=4096 397B lens that's ~22 s/condition;
   for the **d=8192 Llama-70B lens across 26 band layers it's ~3+ min/condition (~5 h / $44
   for 90)** — a super-linear blow-up (cache thrash on 8192² matrices). Fix: **move each band
   layer's J to that layer's device** (`J = lens.jacobians[l].to(h.device)`) and do `h@J.T +
   unembed` on GPU — the pattern in `peek_thinking.py` / `llama70b/llama_battery.py`. This
   works under `device_map` (each layer's residual is on the GPU that holds the layer; `.to`
   lands J there). Rule of thumb: **cost of a naive lens readout scales with d² — budget for a
   GPU-transport runner before pointing the lens at any d≥8192 model.**
16. **Control lenses via `copy.deepcopy` are O(d²·n_layers) and blow up for large lenses.**
   `make_control_lens` deepcopies the whole lens + fills a random-J per source layer; for
   d=8192 that stalled setup ~40 min at 1900% CPU (looked hung; was grinding). Added
   `--skip-controls` to demo2. Better: **generate the identity (skip-transport) and random-J
   (inline `randn(d,d)` Frobenius-matched) controls per BAND layer at readout time** (~20
   layers, not all 80) — cheap, GPU-resident. Don't deepcopy a giant lens for controls.
17. **Redirected stdout is BLOCK-buffered — a slow run looks like "0 conditions / hung."**
   Launch pod runs with **`python3 -u`** (or `PYTHONUNBUFFERED=1`) so `== condition:` prints
   flush immediately; otherwise you can't tell "slow" from "stuck" and waste money terminating
   a healthy pod (or grinding a dead one). Diagnose stuck-vs-slow with **GPU-util + the
   process's %CPU**: GPU 0% + CPU high = slow CPU-side work (transport/decode/controls);
   CPU ~0% = truly hung. `--skip-per-layer-topk` does NOT speed the readout (it only drops the
   field from the JSON; `decode_topk` still runs) — cut `--topk` and drop per-position clouds
   to actually reduce readout cost.

## 💸 Cost lessons (2026-07-11 — a ~$180 session that should have been ~$120)

18. **⚠️ For a big model, the DOWNLOAD is the cost driver — every separate pod re-pays it.**
   The 397B (~800GB) downloads in ~40 min = **~$25 idle on 8×H200 each time**; the actual
   readout/generation compute is cheap by comparison. This session ran the 397B on THREE
   separate pods (main battery, thinking-on recovery, peek-inside-thinking) that each
   re-downloaded — **~$75 in downloads alone**. RULE: when you anticipate more than one
   experiment on a big model, **do them in ONE pod session (one download)** or keep the pod
   warm between them; do NOT terminate-then-reacquire per experiment. If the asks arrive over
   time (they will), weigh "keep the rare/expensive pod warm at idle $/hr" against "re-download
   next time" — for the 397B, keeping it warm ~30 min ($18) beats a fresh $25 download+setup.
19. **⚠️ A DC-pinned network volume is worthless if its datacenter has no GPU supply.** The
   campaign volume `jh4vh4cx4o` is in US-NC-1, which was **supply-dry all day** (SECURE US-NC-1
   returned SUPPLY_CONSTRAINT for even a cheap A4000). So the volume — the whole point of which
   is to cache the 397B and skip the download — was **unusable every time**, and each 397B run
   fell back to cloud-ALL and re-downloaded. LESSON: **before designing around a volume, confirm
   its DC actually gives you pods**; if the DC is chronically dry, **recreate the volume in a DC
   where cloud-ALL keeps landing you H200s** (volumes are cheap to make/destroy). A cache you
   can't mount is $63/mo of nothing.
20. **⚠️ Track CUMULATIVE session spend against the stated ceiling and SURFACE the crossing.**
   The overnight budget was $120; a chain of individually-authorized follow-on asks (recovery,
   the $35 peek, the Llama replication that needed two pod attempts) pushed the session to
   **~$180**. Each ask was authorized, but no single ask "felt" like the one that blew the
   ceiling. Per house rule 6: keep a running tally and, **when cumulative spend crosses the
   stated budget, say so explicitly and confirm before the next NEW launch** — even mid-flow,
   even when the user is engaged and asking for more. "Each piece was approved" is not the same
   as "the total was approved."
21. **Audit COMBINED $/hr across parallel pods, not per-pod.** Running peek (8×H200, $35/hr) and
   Llama (2×H200, $8.78/hr) at once = **~$44/hr**. Parallelism is good for wall-clock, but
   `launch.py pods` shows the sum for a reason — a $9/hr pod feels cheap until it's stacked on a
   $35/hr one. Two failed/slow Llama attempts (perf wall, transient HF download timeout) also
   quietly added pod-hours; a run that dies at load still billed the download.

## Folder structure (built to add research easily)

```
research-and-replications/
  README.md            # index of all projects
  CLAUDE.md            # this file
  pyproject.toml       # deps (numpy/scipy; torch behind the CPU-wheel note below)
  shared/              # cross-project, reusable infra
    runpod/            # GPU launcher (stdlib only — runs on this box, no deps)
  projects/            # one self-contained folder per research area
    <project-slug>/
      README.md        # overview + replication-status table + how-to-run
      background.md    # lit review / assessment / why we're reproducing it
      common/          # project-local shared code (data, metrics, nets)
      experiments/
        <experiment-slug>/
          README.md    # what claim/figure it targets + exact command
          train.py     # or run.py — the experiment
          results.md   # LEDGER of actual runs (never fabricate a number)
```

**To add a new research project:** copy the shape of
`projects/jacobian-lens-and-identifiability/` — a `README.md` (with a status table), a
`background.md`, and `experiments/<name>/` folders each with their own README + code +
`results.md`. Add a row to the top-level `README.md` index. Keep every project
self-contained so humans and agents can navigate by folder alone.

## Checkpoint discipline (session survival)

Maintain **`checkpoint.md`** at the repo root (gitignored — state, not history). It exists
so a disrupted session can be resumed by a fresh Claude or by TJ without re-derivation.
**Update it frequently** — after every significant step, not just at milestones: when an
experiment starts/finishes, when a result lands, when infrastructure spins up or down
(especially anything BILLING, with its kill-command), when a finding or confound changes
the plan, and when the next-steps order changes. It must always answer: what's running
right now (and what it costs), what the results so far are, what's next in order, and the
gotchas a cold session would trip on. If checkpoint.md and reality disagree, trust
reality, then fix checkpoint.md.

## House rules

- **Pre-register before running, formally when reasonable (TJ, 2026-07-10).** Before
  any experiment whose outcome we might publish: freeze the design (prompts, metrics,
  gates, controls, predictions) and commit it BEFORE results exist — git pre-registration
  is the floor (see `experiments/lens_demo/` for the pattern). When the experiment is
  substantial enough that we'd blog/tweet the result, ALSO file an OSF or AsPredicted
  pre-registration if reasonable (needs TJ's account — ask him; cite the registration
  ID in the results ledger and blog). "Not reasonable" = quick mechanics smokes,
  debugging runs, exploratory sweeps explicitly labeled exploratory in the ledger.
- **Save the raw ingredients, not just derived stats (TJ, 2026-07-11 — learned at ~$20/lesson).**
  Every model-touching runner must write into its receipt: the model OUTPUT-HEAD top-k
  (ids + scores) at every readout position and generation step — not just decoded text;
  per-layer lens top-k WITH scores for every transport; generated token IDs; all args,
  seeds, artifact hashes/revisions, and package versions. Test: "could a new analysis be
  done from this receipt alone, without re-renting the GPU?" If a control is named after
  something it is not (e.g. `logit_lens` = identity transport, not the output head),
  say so IN the receipt metadata. Storage is pennies; re-runs are not.
- **`uv` for envs.** The CPU torch wheel isn't on PyPI for cp312 under the default
  resolver; install it by direct URL (see README "Running") — pyproject keeps torch in
  an extra so the light path stays clean.
- **Honesty over green.** `results.md` records only runs you actually executed. A
  replication that *doesn't* reproduce is a valid, valuable result — report it plainly
  (see the nonlinear-ICA `results.md` for the tone).
- **No product coupling.** Never `import prax`; never make Prax depend on this repo. When
  a finding should change the harness, write the *conclusion* into `prax/docs/research/`
  and implement it there — the experiment code stays here.
- **Cite the claim, reproduce the metric.** Each experiment names the exact
  theorem/figure and the paper's own success metric.
- **Secrets never land in git.** `RUNPOD_API_KEY` / `HF_TOKEN` live in the environment or a
  gitignored `shared/runpod/.env` — never committed. Sweep staged files before commit.
- **⚠️ Secrets never land on a remote box either — `.gitignore` does NOT protect
  `rsync`/`tar`/`scp`.** Those copy whatever is on disk, including the gitignored
  `shared/runpod/.env`. **When transferring code to a RunPod pod or Lightsail, ALWAYS
  exclude the env file** — `tar --exclude=.venv --exclude='.env' …` /
  `rsync --exclude='.env' …`. A tar-over-SSH to a fit pod once shipped `shared/runpod/.env`
  (our RunPod + HF keys) onto a third-party GPU box; if it happens, **`rm` it from the box
  immediately**. For a model that needs a token, pass it inline for that one command, not
  as a file left on the pod.
