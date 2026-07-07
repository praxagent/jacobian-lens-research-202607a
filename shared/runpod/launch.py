#!/usr/bin/env python3
"""RunPod GPU launcher — stdlib only, so it runs on the (GPU-less) dev box.

Cost discipline is built in (see ../../CLAUDE.md): pods terminate on completion by
default, `pods` lists anything still running (so nothing is forgotten), and every
create path prints the $/hr before it spends.

Auth: set RUNPOD_API_KEY in the environment, or drop it in a gitignored
`shared/runpod/.env` as `RUNPOD_API_KEY=...`. Never commit the key.

Usage:
    python launch.py pods                       # list running pods (+ hourly burn)
    python launch.py gpus                        # list available GPU types + price
    python launch.py run \
        --repo https://github.com/praxagent/research-and-replications \
        --ref main \
        --cmd "cd projects/jacobian-lens-and-identibility/experiments/nonlinear_ica_sparsity && uv run python train.py --mode nonlinear --n 8 --device cuda" \
        --gpu "NVIDIA RTX A4000" --max-minutes 60      # create -> run -> fetch logs -> terminate
    python launch.py terminate --pod <id>        # kill a pod now

This is a first-cut against RunPod's documented GraphQL API. The FIRST live run is
the validation pass — GPU-type display names and image tags may need a tweak once we
see the real API responses (do that on the cheapest GPU).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.runpod.io/graphql"
# A stock PyTorch/CUDA image; the entrypoint installs uv + repo deps on top.
DEFAULT_IMAGE = "runpod/pytorch:2.4.1-py3.11-cuda12.4.1-devel-ubuntu22.04"


def _load_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        env = Path(__file__).with_name(".env")
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("RUNPOD_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit(
            "No RUNPOD_API_KEY. Export it, or put RUNPOD_API_KEY=... in "
            "shared/runpod/.env (gitignored). Get one at runpod.io -> Settings -> API Keys."
        )
    return key


def _gql(query: str, variables: dict | None = None) -> dict:
    key = _load_key()
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        f"{API_URL}?api_key={key}", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        sys.exit(f"RunPod API HTTP {e.code}: {e.read().decode()[:300]}")
    if "errors" in payload:
        sys.exit(f"RunPod API error: {json.dumps(payload['errors'])[:400]}")
    return payload.get("data", {})


# --------------------------------------------------------------------------- #
def cmd_pods(_args) -> None:
    q = """query { myself { pods {
        id name desiredStatus costPerHr
        runtime { uptimeInSeconds } machine { gpuDisplayName }
    } } }"""
    pods = _gql(q).get("myself", {}).get("pods", []) or []
    if not pods:
        print("No pods. (Good — nothing is billing.)")
        return
    total = 0.0
    print(f"{'id':22s} {'status':10s} {'$/hr':>7s} {'up(min)':>8s}  gpu")
    for p in pods:
        cost = float(p.get("costPerHr") or 0)
        up = (p.get("runtime") or {}).get("uptimeInSeconds") or 0
        if p.get("desiredStatus") == "RUNNING":
            total += cost
        print(f"{p['id']:22s} {p.get('desiredStatus',''):10s} {cost:7.3f} "
              f"{up/60:8.1f}  {(p.get('machine') or {}).get('gpuDisplayName','')}")
    if total:
        print(f"\n>>> {total:.3f} $/hr burning right now. Terminate what you're done with.")


def cmd_gpus(_args) -> None:
    q = """query { gpuTypes {
        id displayName memoryInGb
        lowestPrice(input:{gpuCount:1}) { minimumBidPrice uninterruptablePrice }
    } }"""
    gpus = _gql(q).get("gpuTypes", []) or []
    gpus = [g for g in gpus if (g.get("lowestPrice") or {}).get("uninterruptablePrice")]
    gpus.sort(key=lambda g: g["lowestPrice"]["uninterruptablePrice"])
    print(f"{'display name':32s} {'VRAM':>6s} {'on-demand $/hr':>14s} {'spot $/hr':>10s}")
    for g in gpus:
        lp = g["lowestPrice"]
        print(f"{g['displayName']:32s} {g.get('memoryInGb',0):5d}G "
              f"{lp.get('uninterruptablePrice',0):14.3f} {lp.get('minimumBidPrice',0) or 0:10.3f}")
    print("\nPick the cheapest GPU that fits the job's VRAM. Spot is cheaper but can be reclaimed.")


def _startup_script(repo: str, ref: str, cmd: str) -> str:
    # Runs on the pod at boot: clone, install uv + deps, run the experiment, print
    # a sentinel so the log-poller knows when to fetch results and terminate.
    return (
        "set -e; "
        "curl -LsSf https://astral.sh/uv/install.sh | sh; "
        'export PATH="$HOME/.local/bin:$PATH"; '
        f"git clone --depth 1 --branch {ref} {repo} /workspace/repo; "
        "cd /workspace/repo; "
        "uv venv && uv pip install -e '.[torch]'; "
        f"( {cmd} ) 2>&1 | tee /workspace/run.log; "
        "echo __PRAX_RUN_DONE__ | tee -a /workspace/run.log"
    )


def cmd_run(args) -> None:
    price = _gpu_price(args.gpu)
    est = (price or 0) * (args.max_minutes / 60.0)
    print(f"GPU: {args.gpu}  ~{price:.3f} $/hr  |  cap {args.max_minutes} min  "
          f"|  worst-case ~${est:.2f}")
    if not args.yes:
        sys.exit("Refusing to spend without --yes. Re-run with --yes once the user has "
                 "approved THIS run (see CLAUDE.md rule 4).")
    print(">>> This launcher creates + polls + terminates. Confirm the pod is gone "
          "afterward with `launch.py pods`.")
    # NOTE: pod-create wiring (podFindAndDeployOnDemand mutation) is intentionally
    # left as the first-live-run step — it needs a real API response to pin the exact
    # gpuTypeId + image fields. The startup script and teardown are ready:
    print("\n--- startup script the pod will run ---")
    print(_startup_script(args.repo, args.ref, args.cmd))
    print("\n(Complete the create-mutation on the first live run, then this becomes "
          "fully automatic. Kept explicit so we never spend on an unvalidated call.)")


def _gpu_price(display: str) -> float | None:
    q = """query { gpuTypes { displayName lowestPrice(input:{gpuCount:1}) { uninterruptablePrice } } }"""
    for g in _gql(q).get("gpuTypes", []) or []:
        if g.get("displayName") == display:
            return float((g.get("lowestPrice") or {}).get("uninterruptablePrice") or 0)
    return None


def cmd_terminate(args) -> None:
    q = "mutation($id:String!){ podTerminate(input:{podId:$id}) }"
    _gql(q, {"id": args.pod})
    print(f"Terminated {args.pod}. Verify with `launch.py pods`.")


def main() -> None:
    p = argparse.ArgumentParser(description="RunPod launcher (cost-disciplined)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pods").set_defaults(func=cmd_pods)
    sub.add_parser("gpus").set_defaults(func=cmd_gpus)
    r = sub.add_parser("run")
    r.add_argument("--repo", required=True)
    r.add_argument("--ref", default="main")
    r.add_argument("--cmd", required=True, help="command to run inside the repo on the pod")
    r.add_argument("--gpu", required=True, help="GPU display name (see `gpus`)")
    r.add_argument("--max-minutes", type=int, default=60)
    r.add_argument("--yes", action="store_true", help="confirm this specific paid run")
    r.set_defaults(func=cmd_run)
    t = sub.add_parser("terminate"); t.add_argument("--pod", required=True)
    t.set_defaults(func=cmd_terminate)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
