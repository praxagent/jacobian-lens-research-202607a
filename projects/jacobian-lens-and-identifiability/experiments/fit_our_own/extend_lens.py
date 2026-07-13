"""Extend a published Jacobian lens by continuing its prompt stream — with the
per-prompt convergence logging the original fit lacked, and volume-backed
kill-anywhere recoverability (a pod dying mid-prompt-30 leaves n=29 restorable).

Design: no reimplemented math. We reconstruct a `jlens.fit`-compatible checkpoint
from the published lens (jacobian_sum = J*n, n_done = next_idx = n; documented fp16
caveat: the published J is fp16, so the reconstructed sum carries that rounding at
weight n/target_n), then repeatedly call Anthropic's own `jlens.fit` with a one-longer
prompt prefix and resume=True — their loop processes exactly one new prompt per call
and checkpoints. Between calls we compute:
  mean_rel_change(n) = mean_l ||J_l(n) - J_l(n-1)||_F / ||J_l(n-1)||_F
  identity_distance(n) = mean_l ||J_l(n) - I||_F / ||I||_F
(formulas documented here; same spirit as Neuronpedia's convergence CSV — theirs may
normalize differently, so compare shapes not absolute values), append a CSV row, and
atomically sync ckpt+CSV to --sync-dir (the network volume). Corpus prefix property:
load_corpus(N, seed) shuffles the full corpus then takes the first N, so the original
n prompts are reproduced exactly and continuation starts at index n.

Run (397B campaign, on the pod, /workspace = network volume):
  python extend_lens.py --model Qwen/Qwen3.5-397B-A17B --backbone-path model.language_model \\
    --lens-file /workspace/qwen35_397b_n24.pt --target-n 100 --dim-batch 16 \\
    --work /root/extend --sync-dir /workspace/extend_397b
Every completed prompt leaves /workspace/extend_397b/{extend.ckpt,convergence.csv}
consistent; relaunching the same command resumes losslessly.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_lens import load_corpus  # noqa: E402


def hf_push(repo: str, local: Path, path_in_repo: str, msg: str) -> None:
    """Best-effort background-safe upload; a failed push never kills the fit."""
    try:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo,
                            repo_id=repo, commit_message=msg)
    except Exception as e:  # noqa: BLE001 — durability lives on the volume, not HF
        print(f"  (hf push failed, continuing: {type(e).__name__}: {str(e)[:80]})")


def write_config_yaml(path: Path, args, n_done: int, ident: float, mrc) -> None:
    """Neuronpedia-format fit record (their config.yaml shape), updated per push."""
    path.write_text(f"""# Jacobian lens EXTENSION — praxagent extend_lens.py
# Continues the published n=24 lens (praxagent-org/jacobian-lens-qwen3.5-397b-a17b,
# sha256 668c3bf1...) on the same seed-0 wikitext stream, via Anthropic's jlens.fit
# resume loop. Bootstrap note: the published J is fp16, so the reconstructed running
# sum carries that rounding at weight 24/n. Formulas: mean_rel_change(n) =
# mean_l ||J_l(n)-J_l(n-1)||_F/||J_l(n-1)||_F ; identity_distance =
# mean_l ||J_l - I||_F/||I||_F (may differ from Neuronpedia's normalization —
# compare curve shapes, not absolute values).
hf_model_name: "{args.model}"
dataset:
  name: "Salesforce/wikitext"
  config: "wikitext-103-raw-v1"
  split: "train"
fit:
  n_prompts_requested: {args.target_n}
  dim_batch: {args.dim_batch}
  max_seq_len: {args.max_seq_len}
  seed: {args.seed}
  dtype: "bfloat16"
  attn_implementation: "eager"
  resumed_from_published_n: 24
results:
  prompts_fitted: {n_done}
  final_identity_distance: {ident:.6f}
  final_mean_rel_change: {mrc if mrc else 'null'}
attribution: "jlens by Anthropic PBC (Apache-2.0); extension loop by praxagent"
""")


def atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def mean_rel_change(prev: dict, cur: dict) -> float:
    vals = []
    for l, Jp in prev.items():
        d = (cur[l].float() - Jp.float()).norm() / (Jp.float().norm() + 1e-12)
        vals.append(float(d))
    return sum(vals) / len(vals)


def identity_distance(cur: dict) -> float:
    vals = []
    for l, J in cur.items():
        J = J.float()
        d = J.shape[0]
        eye_norm = d ** 0.5
        vals.append(float((J - torch.eye(d)).norm() / eye_norm))
    return sum(vals) / len(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backbone-path", default=None)
    ap.add_argument("--lens-file", required=True,
                    help="published lens .pt to extend (local path)")
    ap.add_argument("--target-n", type=int, required=True)
    ap.add_argument("--dim-batch", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--work", required=True, help="fast local dir (ckpt lives here)")
    ap.add_argument("--sync-dir", required=True,
                    help="durable dir (network volume) — ckpt+CSV synced per prompt")
    ap.add_argument("--snapshot-every", type=int, default=10,
                    help="also export an fp16 lens snapshot every N prompts")
    ap.add_argument("--hf-repo", default=None,
                    help="push convergence.csv + config.yaml per prompt (and lens "
                         "snapshots per --snapshot-every) to this HF repo under "
                         "jlens/wikitext/extension/ — never touches the n=24 path")
    args = ap.parse_args()

    import jlens
    import transformers

    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
    sync = Path(args.sync_dir); sync.mkdir(parents=True, exist_ok=True)
    ckpt_local = work / "extend.ckpt"
    ckpt_sync = sync / "extend.ckpt"
    csv_path = sync / "convergence.csv"

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    n_gpu = torch.cuda.device_count() if dev != "cpu" else 0
    kw = dict(torch_dtype=torch.bfloat16)
    if args.backbone_path or n_gpu > 1:
        cfg = transformers.AutoConfig.from_pretrained(args.model)
        cls = getattr(transformers, cfg.architectures[0])
        kw.update(device_map="auto", attn_implementation="eager",
                  max_memory={i: "110GiB" for i in range(max(1, n_gpu))})
        hf = cls.from_pretrained(args.model, **kw).eval()
    else:
        hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, **kw)
        hf = hf.to(dev).eval()
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    if args.backbone_path:
        from jlens import Layout
        model = jlens.from_hf(hf, tok, compile=False,
                              layout=Layout(path=args.backbone_path))
    else:
        model = jlens.from_hf(hf, tok, compile=False)

    # ---- bootstrap checkpoint: volume copy > local copy > reconstruct from lens
    if ckpt_sync.exists() and not ckpt_local.exists():
        print(f"bootstrap: pulling checkpoint from volume {ckpt_sync}")
        atomic_copy(ckpt_sync, ckpt_local)
    if not ckpt_local.exists():
        base = jlens.JacobianLens.load(args.lens_file)
        n0 = base.n_prompts if hasattr(base, "n_prompts") else None
        if n0 is None:
            d0 = torch.load(args.lens_file, map_location="cpu", weights_only=False)
            n0 = d0["n_prompts"]
        state = {
            "jacobian_sum": {l: J.float() * n0 for l, J in base.jacobians.items()},
            "n_done": n0, "next_idx": n0,
            "source_layers": list(base.source_layers),
            "target_layer": getattr(base, "target_layer", None),
            "skip_first": 16,
        }
        if state["target_layer"] is None:
            state.pop("target_layer")  # let fit fill its default, skip the check
        torch.save(state, ckpt_local)
        print(f"bootstrap: reconstructed fit-checkpoint from {args.lens_file} at "
              f"n={n0} (fp16-published J -> fp32 sum; rounding carried at weight "
              f"{n0}/{args.target_n})")

    prompts = load_corpus(args.target_n, args.seed, corpus="wikitext")
    state = torch.load(ckpt_local, map_location="cpu", weights_only=True)
    n_done, next_idx = state["n_done"], state["next_idx"]
    prev_mean = {l: s.float() / n_done for l, s in state["jacobian_sum"].items()}
    print(f"resuming at n_done={n_done} next_idx={next_idx} target={args.target_n}")

    new_csv = not csv_path.exists()
    fcsv = open(csv_path, "a", newline="")
    w = csv.writer(fcsv)
    if new_csv:
        w.writerow(["n_done", "next_idx", "elapsed_s", "identity_distance",
                    "mean_rel_change", "note"])
        w.writerow([n_done, next_idx, 0.0, round(identity_distance(prev_mean), 6),
                    "", "baseline (reconstructed from published lens)"])
        fcsv.flush()

    import threading
    stop = threading.Event()
    state_lock = {"prev": prev_mean, "n": n_done}

    def sidecar():
        """Watch THEIR checkpoint; on each new prompt: CSV row + volume sync + HF push.
        Read-only wrt the fit — never touches fit state."""
        last_mtime = 0.0
        while not stop.is_set():
            stop.wait(5.0)
            try:
                m = ckpt_local.stat().st_mtime
                if m == last_mtime:
                    continue
                last_mtime = m
                st = torch.load(ckpt_local, map_location="cpu", weights_only=True)
                nd = st["n_done"]
                if nd <= state_lock["n"]:
                    continue
                cur = {l: s.float() / nd for l, s in st["jacobian_sum"].items()}
                mrc = mean_rel_change(state_lock["prev"], cur)
                w.writerow([nd, st["next_idx"], "", round(identity_distance(cur), 6),
                            round(mrc, 8), ""])
                fcsv.flush()
                print(f"  [sidecar] n={nd} mean_rel_change={mrc:.6f}")
                state_lock["prev"], state_lock["n"] = cur, nd
                atomic_copy(ckpt_local, ckpt_sync)
                if args.hf_repo:
                    cfg = sync / "config.yaml"
                    write_config_yaml(cfg, args, nd, identity_distance(cur),
                                      round(mrc, 8))
                    hf_push(args.hf_repo, csv_path,
                            "jlens/wikitext/extension/convergence.csv",
                            f"extension: n={nd}")
                    hf_push(args.hf_repo, cfg,
                            "jlens/wikitext/extension/config.yaml",
                            f"extension: n={nd}")
                if nd % args.snapshot_every == 0:
                    snap_state = {l: (s.float() / nd) for l, s in
                                  st["jacobian_sum"].items()}
                    tmp = work / f"snap_n{nd}.pt"
                    torch.save({"J": {l: v.half() for l, v in snap_state.items()},
                                "n_prompts": nd,
                                "source_layers": st["source_layers"],
                                "d_model": next(iter(snap_state.values())).shape[0]},
                               tmp)
                    atomic_copy(tmp, sync / f"lens_n{nd}.pt")
                    if args.hf_repo:
                        hf_push(args.hf_repo, sync / f"lens_n{nd}.pt",
                                f"jlens/wikitext/extension/lens_n{nd}.pt",
                                f"extension lens snapshot n={nd}")
            except Exception as e:  # noqa: BLE001 — sidecar must never kill the fit
                print(f"  [sidecar] skipped a tick: {type(e).__name__}: {str(e)[:70]}")

    th = threading.Thread(target=sidecar, daemon=True)
    th.start()
    # ONE call into Anthropic's own loop — the resume path proven bitwise-exact
    lens = jlens.fit(model, prompts, dim_batch=args.dim_batch,
                     max_seq_len=args.max_seq_len,
                     checkpoint_path=str(ckpt_local), checkpoint_every=1,
                     resume=True)
    stop.set(); th.join(timeout=30)
    atomic_copy(ckpt_local, ckpt_sync)
    lens.save(str(work / "final.pt"))
    atomic_copy(work / "final.pt", sync / f"lens_n{lens.n_prompts}.pt")
    n_done = lens.n_prompts
    fcsv.close()
    print(f"DONE at n={n_done}. Durable state: {ckpt_sync}, {csv_path}")


if __name__ == "__main__":
    main()
