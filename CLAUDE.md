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
5. **Get everything ready on CPU so a GPU run is short.** Code, configs, a CPU smoke of
   the exact command, and results plumbing should all be done *before* the pod starts —
   so the paid pod only does the irreducibly-GPU part.

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
