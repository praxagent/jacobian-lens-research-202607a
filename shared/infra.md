# Infrastructure — compute we use, how, and what it costs

How praxagent research runs, and the cost discipline around it (see also the repo
[`CLAUDE.md`](../CLAUDE.md)). Two compute backends: **Amazon Lightsail** (CPU, RAM-bound
sweeps) and **RunPod** (GPU, when the science needs it). Everything durable lives in
**git**; model caches are regenerable and never worth storing.

## Philosophy

- **Code → GitHub** (`github.com/praxagent/research-and-replications`). **Results → git**
  (CSVs, plots are KB). Both free and permanent.
- **Model weights / lenses → regenerable.** They re-download from HuggingFace / Neuronpedia
  for free. Never snapshot or back them up — that's paying to store free data.
- **CPU-first.** RAM-bound work (our CKA emergence sweep only needs the unembedding + lens,
  not a forward pass) runs on Lightsail — no GPU-hours. GPU (RunPod) only for real
  forward/backward passes (fitting lenses, ignition/capacity experiments).
- **Tear down when idle.** Instances bill by the hour; delete/stop when a run is done.

## Amazon Lightsail — the CPU box (used since 2026-07-07)

Chosen because it's the cheapest AWS-family option (helps for tax/accounting).

- **Spec (observed):** 30 GB RAM, 8 vCPU, 619 GB SSD, Ubuntu, Python 3.12. (~$164/mo per
  TJ, **billed hourly** to that cap — a 2–3 day sweep is a few dollars.) Lightsail has
  **no GPU**.
- **Purpose:** the full 38-model J-lens CKA emergence sweep — memory-bound (holds the
  unembedding U + lens J_l in RAM; the 32B tier peaks ~16–18 GB), not GPU-bound.
- **Access (from the dev box, same VPC):**
  `ssh ubuntu@172.26.2.127` using `~/.ssh/id_rsa` (the passphrase-free key; same one used
  for GitHub). Private IP, same zone → fast rsync/ssh.
- **Setup that works** (Python 3.12 + the exclude-newer buffer break the default torch
  resolve, so install the CPU wheel by direct URL):
  ```bash
  uv venv --python 3.12
  uv pip install "https://download.pytorch.org/whl/cpu/torch-2.4.1%2Bcpu-cp312-cp312-linux_x86_64.whl" --exclude-newer 2030-01-01
  uv pip install --exclude-newer 2030-01-01 numpy scipy pyyaml safetensors matplotlib "jlens @ git+https://github.com/anthropics/jacobian-lens"
  ```
- **Run detached** (survives SSH drops): `setsid nohup .venv/bin/python -u ladder.py ... </dev/null >~/run.log 2>&1 &`.
  Gotcha: `pkill -f ladder.py` **matches its own shell** and kills the SSH session — use
  the bracket trick `pgrep -f "[l]adder.py"` and kill by PID.
- **Gated models (Gemma/Llama):** need `HF_TOKEN` (in the gitignored `shared/runpod/.env`)
  **and** the account to have accepted the model licence on HuggingFace.

## RunPod — the GPU backend (launcher validated live 2026-07-08)

- **Purpose:** the genuinely GPU-first work — fitting our own lenses, and the
  ignition/capacity experiments (live forward passes on real LLMs). Lightsail can't do this.
- **Launcher:** [`shared/runpod/`](runpod/) — stdlib-only, runs on the dev box. `create` /
  `sshinfo` / `terminate` are validated (RTX A2000 test, ~$0.02). **How the API key is
  stored/used is documented in [`shared/runpod/README.md`](runpod/README.md)** — gitignored
  `.env`, GraphQL over HTTPS with a User-Agent header, our `id_rsa.pub` injected for SSH.
- **Discipline:** never start a paid pod without explicit per-run approval + a $/hr
  estimate; **terminate the moment a run finishes** (`pods` shows what's billing); cheapest
  GPU that fits the VRAM.

## Cost model + the snapshot question (2026-07-07)

**What's actually on disk** (at 14/38 models, 76 GB used): the cache is dominated by
**full model files**, not lenses — e.g. gemma-4-E4B 15 GB, Qwen3.5-4B 5 GB, gemma-2-2b
4.7 GB. The **lenses** (the thing our method centres on) are only **4.4 GB total**. We
download the models because the CKA needs each model's **unembedding matrix `U`**, which
lives inside the model checkpoint. Final full-disk size is uncertain — **~150–400 GB**
depending on how many big models download as a single file vs. one shard (sharded big
models fetch only the embedding shard; single-file ones pull the whole checkpoint).

**Lightsail snapshot pricing: $0.05 per GB-month** on the stored disk size.

The lever: **we never actually need the full models kept around — only the lenses + each
model's `U`.** That's a small fraction of the disk. So:

| Option | ~Monthly cost | Convenience | Notes |
|---|---|---|---|
| Keep **instance running** | ~$164 | instant | pays full compute even when idle |
| **Full-disk snapshot** (~300 GB) | **~$15/mo** | instant restore, everything ready | pays to store ~250 GB of re-downloadable full models |
| **Minimal snapshot** (lenses + saved `U`, ~60–100 GB) | **~$3–5/mo** ✅ | instant re-analysis, no re-download | needs `U` cached to disk (cheap to build — models are already local) |
| Snapshot lenses only (~30–60 GB) | ~$1.5–3/mo | re-analysis re-downloads models | lenses ready, `U` not |
| **Delete, keep only git** | **$0** | ~1–3 h re-download on resume | results already in git |

**Recommendation for cost-effective "models at the ready": the minimal snapshot
(~$3–5/mo).** We build a small `unembeddings/` cache (each model's `U`, ~0.5–3 GB, saved
once — the models are already downloaded, so this needs **no** re-download), keep the
Neuronpedia lens cache, drop the full models, and snapshot that. Future analyses
(base-vs-instruct, the brain-alignment bridge, new metrics) then run instantly with zero
re-download, at ~1/3–1/5 the cost of snapshotting the full 300 GB of models we don't
need. If instead we won't resume for a while, **delete everything** — results are in git
and a full re-download is free.

> **Disk-fill watch:** the loader falls back to a *full-model* download if the
> embedding-shard partial-load fails. On a 27–70B model that's tens of GB and could fill
> the 619 GB disk. Monitor `df -h /` during the sweep; if a big model triggers the
> fallback, skip it or fix the shard path rather than let the disk fill.

## Teardown checklist (when a sweep is done)

1. Commit results to git (`emergence*.csv`, PNGs) and push.
2. `rsync` any artifacts you want off the box back to the dev box / git.
3. Delete the Lightsail instance (stops the hourly bill). Do **not** snapshot the cache.
4. Confirm no RunPod pods are running (`python shared/runpod/launch.py pods`).
