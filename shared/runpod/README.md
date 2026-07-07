# shared/runpod — GPU launcher

A stdlib-only launcher so it runs on the (GPU-less) dev box. **Read
[`../../CLAUDE.md`](../../CLAUDE.md) cost discipline first** — GPUs are paid out of
pocket; CPU-first, terminate immediately, never spend without an explicit OK.

## How you empower Claude to run GPU jobs

1. Get a key: runpod.io → Settings → **API Keys** → create one (read/write).
2. Give it to this box **without putting it in the transcript or git** — either:
   - type `! echo 'RUNPOD_API_KEY=YOURKEY' > shared/runpod/.env` in the session
     (the file is gitignored), or
   - `! export RUNPOD_API_KEY=YOURKEY` (note: env vars don't persist across Claude's
     separate shells, so the `.env` file is more reliable).
3. Then Claude can:
   ```bash
   python shared/runpod/launch.py gpus     # see GPU types + $/hr, pick the cheapest that fits
   python shared/runpod/launch.py pods     # see anything currently billing
   python shared/runpod/launch.py run --repo <git-url> --ref <branch> \
       --gpu "<display name>" --max-minutes 60 --cmd "<experiment command>" --yes
   python shared/runpod/launch.py terminate --pod <id>
   ```

## The guardrails (built in)

- `run` **prints the worst-case cost** ($/hr × cap) and **refuses without `--yes`** —
  Claude will not spend on a specific run without your go-ahead for *that* run.
- Pods are meant to **terminate on completion**; `pods` always shows what's still
  running and the total $/hr burning, so nothing is silently left on.
- The launcher needs the repo reachable by `git clone` (push this repo to GitHub, or
  extend the launcher to `runpodctl send` local files) — the pod clones `--ref`,
  installs `.[torch]`, runs `--cmd` on the GPU, tees `/workspace/run.log`.

## First-live-run note (honest)

The `create-pod` GraphQL mutation is deliberately left to execute on the first real
run (on the cheapest GPU), because pinning the exact `gpuTypeId`/image fields needs a
real API response — I won't hard-code an unvalidated paid call. `pods`, `gpus`,
`terminate`, cost-printing, and the pod startup script are ready now. Once we do one
cheap live run together, `run` becomes fully automatic.
