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
- **Secrets never land in git.** `RUNPOD_API_KEY` lives in the environment or a
  gitignored `shared/runpod/.env` — never committed. Sweep staged files before commit.
