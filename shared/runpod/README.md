# shared/runpod — GPU launcher

A stdlib-only launcher (`launch.py`) that runs on the dev box and drives RunPod GPU pods.
**Read [`../../CLAUDE.md`](../../CLAUDE.md) cost discipline first** — the key can spend money.

## How the API key is used (transparency)

**Storage.** `RUNPOD_API_KEY` lives only in **`shared/runpod/.env`**, which is
**gitignored** (see `.gitignore`) — it is never committed, and the launcher never prints
it. The same file also holds `HF_TOKEN`. Give the key to the box with, e.g.,
`! echo 'RUNPOD_API_KEY=...' >> shared/runpod/.env` typed in the session.

**Loading.** `launch.py:_load_key()` reads `RUNPOD_API_KEY` from the environment, else from
that `.env` file. Nothing else reads it.

**Transport.** Every call is one HTTPS POST to RunPod's GraphQL endpoint,
`https://api.runpod.io/graphql?api_key=<KEY>`, with a `User-Agent: praxagent-research/1.0`
header (RunPod's Cloudflare returns error 1010 to the default Python user-agent). The key
is a query parameter on that URL — standard for RunPod's GraphQL API. Errors are surfaced;
the key is not logged.

**What the key can do** (so you know the blast radius): it authenticates as your RunPod
account, so it can **read** (`pods`, `gpus`) and, via mutations, **create pods (starts
billing)** and **terminate pods**. Treat it as spend-capable and account-scoped — hence the
approval + terminate-on-done discipline below.

**SSH into pods.** `create` injects **our existing `~/.ssh/id_rsa.pub`** as the pod's
`PUBLIC_KEY` env var (RunPod adds it to the pod's `authorized_keys`). We then reach the pod
over plain SSH at its public IP/port — the private key never leaves this box; the pod only
ever sees the public half. Same trust model as the Lightsail box.

## Commands

```bash
python launch.py pods                      # list pods + total $/hr burning (free read)
python launch.py gpus                      # GPU types, VRAM, $/hr (free read)
python launch.py create --gpu-id "<id>" [--cloud ALL] [--disk 40] [--name ...]  # create a pod (BILLS)
python launch.py sshinfo --pod <id>        # poll until the pod exposes public SSH; prints the ssh line
python launch.py terminate --pod <id>      # stop billing NOW
```

Typical run: `create` → `sshinfo` → `ssh`/`rsync` to the pod (clone or push code, install
deps, run the job, fetch results) → **`terminate`**. GPU-type ids come from `gpus`
(e.g. `"NVIDIA GeForce RTX 3090"`).

## Guardrails (cost discipline)

- **Never create a pod without explicit per-run approval** and a stated $/hr estimate.
- **Terminate the moment a job finishes.** `pods` shows anything still billing — nothing
  should be left running.
- Prefer the cheapest GPU that fits the job's VRAM; validate on a tiny/cheap pod first.

## Status

**Validated live 2026-07-08** — created an RTX A2000 pod, SSH'd in with our key, confirmed
GPU + torch/CUDA, and terminated (total ~$0.02). `create`/`sshinfo`/`terminate` all work.
