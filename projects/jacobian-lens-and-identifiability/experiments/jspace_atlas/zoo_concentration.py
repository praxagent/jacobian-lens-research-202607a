"""Zoo-wide readout concentration: is a concentrated readout covariance a gemma-2 trait?

The readout covariance M_c = U_c^T U_c depends ONLY on the unembedding, so this needs each
model's embedding matrix and no lens at all. For every atlas model we compute the
participation ratio of M_c's spectrum on a common probe, which is the quantity that predicted
how much the top readout directions mask depth structure (readout_ablation.py).

Design notes:
  * COMMON PROBE. Each tokenizer resolves the 4,096 shared strings to a slightly different
    subset (the canonical resolver drops strings needing >1 token). We take the INTERSECTION
    across every model, so each arm is measured on an identical probe and n cannot differ by
    arm. Per-model resolvable counts are recorded anyway (integrity playbook 10B).
  * CANONICAL RESOLUTION. Uses jacobian_lens.shared_vocab.resolve_ids, the function the
    shared-token artifact was built with, not a lookalike re-derivation.
  * NORMALIZATION. A participation ratio is bounded by min(n_probe, d), so the comparable
    quantity is PR / min(n_probe, d), not raw PR.
  * RAM/DISK FRUGAL. Eigenvalues of U_c U_c^T (n x n) equal the nonzero eigenvalues of
    U_c^T U_c (d x d), so we never form a d x d matrix; embeddings are streamed and deleted
    after use, so disk stays bounded. Resumable: existing rows are skipped.

  python zoo_concentration.py --stage ids      # pass 1: tokenizers only (cheap)
  python zoo_concentration.py --stage embed    # pass 2: stream embeddings, compute, delete
"""
from __future__ import annotations
import argparse, json, sys, gc
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
JL = HERE.parent / "jacobian_lens"
sys.path.insert(0, str(JL))
from shared_vocab import resolve_ids                      # noqa: E402  (canonical resolver)

OUT = HERE / "decompose_out"
SHARED = json.load(open(JL / "shared_tokens.json"))
IDS_CACHE = OUT / "zoo_shared_ids.json"
RESULTS = OUT / "zoo_concentration.json"
TMP = OUT / "_tmp_embed.npy"
CHUNK = 8192
TGT = "model.embed_tokens.weight"


def stage_ids():
    """Resolve the shared strings per model (tokenizers only). Resumable."""
    cache = json.loads(IDS_CACHE.read_text()) if IDS_CACHE.exists() else {}
    strings = SHARED["strings"]
    for slug, hf_id in sorted(SHARED["models"].items()):
        if slug in cache:
            continue
        try:
            ids = resolve_ids(hf_id, strings)
            cache[slug] = ids
            print(f"[{slug:16s}] resolved {len(ids)}/{len(strings)}", flush=True)
        except Exception as e:
            print(f"[{slug:16s}] TOKENIZER FAILED: {type(e).__name__}: {e}", flush=True)
        IDS_CACHE.write_text(json.dumps(cache))
    ok = {k: v for k, v in cache.items() if v}
    common = set.intersection(*(set(v) for v in ok.values())) if ok else set()
    print(f"\n{len(ok)} models; COMMON probe = {len(common)} strings "
          f"(min per-model {min(len(v) for v in ok.values())}, max {max(len(v) for v in ok.values())})")
    return cache, sorted(common)


def fetch_embed(hf_id):
    """Download only the shard holding embed_tokens and return it as a float16 memmap path."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    import torch
    try:
        idx = hf_hub_download(hf_id, "model.safetensors.index.json")
        shard = json.load(open(idx))["weight_map"][TGT]
    except Exception:
        shard = "model.safetensors"
    sp = hf_hub_download(hf_id, shard)
    with safe_open(sp, framework="pt") as f:
        key = TGT if TGT in f.keys() else next(k for k in f.keys()
                                               if k.endswith("embed_tokens.weight"))
        W = f.get_tensor(key)
    np.save(TMP, W.to(torch.float16).numpy())
    del W; gc.collect()
    return sp


def stage_embed(common):
    cache = json.loads(IDS_CACHE.read_text())
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for slug, hf_id in sorted(SHARED["models"].items()):
        if slug in res or slug not in cache or not cache[slug]:
            continue
        shard_path = None
        try:
            shard_path = fetch_embed(hf_id)
            U = np.load(TMP, mmap_mode="r")
            ids = np.array([cache[slug][s] for s in common])
            Us = np.asarray(U[ids], dtype=np.float32)
            Uc = Us - Us.mean(0, keepdims=True)
            # eigenvalues of U_c U_c^T (n x n) == nonzero eigenvalues of U_c^T U_c (d x d)
            G = Uc @ Uc.T
            ev = np.linalg.eigvalsh(G)[::-1]
            ev = np.clip(ev, 0, None)
            pr = float((ev.sum() ** 2) / (np.square(ev).sum() + 1e-12))
            d = int(U.shape[1]); vocab = int(U.shape[0])
            cap = min(len(ids), d)
            res[slug] = {
                "slug": slug, "hf_id": hf_id, "vocab": vocab, "d": d,
                "n_common": int(len(ids)), "n_resolvable": len(cache[slug]),
                "vocab_over_d": round(vocab / d, 1),
                "readout_pr": round(pr, 2), "pr_norm": round(pr / cap, 4),
                "top1_share": round(float(ev[0] / ev.sum()), 4),
                "top16_share": round(float(ev[:16].sum() / ev.sum()), 4),
            }
            print(f"[{slug:16s}] d={d:5d} vocab={vocab:6d} v/d={vocab/d:6.1f} "
                  f"PR={pr:8.1f} PR/cap={pr/cap:.4f} top16={res[slug]['top16_share']:.3f}",
                  flush=True)
            del U, Us, Uc, G; gc.collect()
        except Exception as e:
            print(f"[{slug:16s}] FAILED: {type(e).__name__}: {str(e)[:90]}", flush=True)
            res[slug] = {"slug": slug, "hf_id": hf_id, "error": f"{type(e).__name__}"}
            shard_path = locals().get("shard_path")
        finally:
            # stream-and-delete: drop BOTH the temp npy and the cached HF blob, or the
            # hub cache alone would exceed this box's free disk across 37 models.
            TMP.unlink(missing_ok=True)
            try:
                if shard_path:
                    real = Path(shard_path).resolve()
                    if "models--" in str(real) and real.suffix == ".safetensors":
                        real.unlink(missing_ok=True)
            except Exception:
                pass
            shard_path = None
        RESULTS.write_text(json.dumps(res, indent=1))
    print("ZOO_CONCENTRATION_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all",
                                                    choices=["ids", "embed", "all"])
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    cache, common = stage_ids()
    (OUT / "zoo_common_probe.json").write_text(json.dumps(common))
    if a.stage in ("embed", "all"):
        stage_embed(common)


if __name__ == "__main__":
    main()
