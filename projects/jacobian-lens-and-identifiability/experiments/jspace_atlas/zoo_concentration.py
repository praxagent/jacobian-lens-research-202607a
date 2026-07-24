"""Zoo-wide readout concentration: is a concentrated readout covariance a gemma-2 trait?

The readout covariance M_c = U_c^T U_c depends ONLY on the unembedding, so this needs each
model's embedding matrix and no lens at all. For every atlas model we compute the
participation ratio of M_c's spectrum on a common probe, which is the quantity that predicted
how much the top readout directions mask depth structure (readout_ablation.py).

Design notes:
  * COMMON PROBE. Uses jacobian_lens.shared_vocab.resolve_ids, the canonical function the
    shared-token artifact was built with (not a lookalike re-derivation). All 37 models
    resolve all 4,096 strings, so every arm is measured on an identical probe by
    construction; per-model resolvable counts are recorded anyway (integrity playbook 10B).
  * NORMALIZATION. A participation ratio is bounded by min(n_probe, d), so the comparable
    quantity is PR / min(n_probe, d), not raw PR.
  * STREAMING. Probe rows are read from the shard in row-chunks via safetensors get_slice,
    so a 15 GB single-file checkpoint never has to be materialized or mmapped whole. The
    downloaded blob is deleted after each model, so disk stays bounded across 37 models.
  * RAM. Eigenvalues of U_c U_c^T (n x n) equal the nonzero eigenvalues of U_c^T U_c (d x d),
    so we never form a d x d matrix.
  Resumable: models already present in the results json are skipped.

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
ROWCHUNK = 8192
# embedding-weight key suffixes across architectures (gpt2 wte, pythia embed_in, llama/gemma
# embed_tokens, gemma-3 nests under language_model.)
KEY_SUFFIXES = ("embed_tokens.weight", "wte.weight", "embed_in.weight", "tok_embeddings.weight")


def find_embed_file(hf_id):
    """Return (repo_filename, tensor_key) for the shard holding the embedding matrix,
    without downloading any weights: prefer the safetensors index, else the single file."""
    from huggingface_hub import HfApi, hf_hub_download
    files = HfApi().list_repo_files(hf_id)
    idx = next((f for f in files if f.endswith(".safetensors.index.json")), None)
    if idx:
        wmap = json.load(open(hf_hub_download(hf_id, idx)))["weight_map"]
        for k in wmap:
            if k.endswith(KEY_SUFFIXES):
                return wmap[k], k
        raise KeyError(f"no embedding key in index; sample={list(wmap)[:3]}")
    st = sorted(f for f in files if f.endswith(".safetensors"))
    if not st:
        raise FileNotFoundError("no .safetensors in repo")
    return st[0], None            # key discovered after open


def stream_probe_rows(hf_id, ids):
    """Download the embedding shard and return ONLY the probe rows as float32 (n, d).
    Reads in row-chunks so a giant checkpoint is never materialized whole. Returns
    (rows, d, vocab, blob_path)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    import torch
    fname, key = find_embed_file(hf_id)
    path = hf_hub_download(hf_id, fname)
    with safe_open(path, framework="pt") as f:
        if key is None or key not in f.keys():
            key = next(k for k in f.keys() if k.endswith(KEY_SUFFIXES))
        sl = f.get_slice(key)
        vocab, d = sl.get_shape()
        order = np.argsort(ids)                       # read in ascending row order
        sorted_ids = np.asarray(ids)[order]
        out = np.empty((len(ids), d), dtype=np.float32)
        pos = 0
        for start in range(0, vocab, ROWCHUNK):
            stop = min(start + ROWCHUNK, vocab)
            want = sorted_ids[(sorted_ids >= start) & (sorted_ids < stop)]
            if want.size == 0:
                continue
            block = sl[start:stop].to(torch.float32).numpy()
            out[order[pos:pos + want.size]] = block[want - start]
            pos += want.size
            del block
        del sl
    gc.collect()
    return out, int(d), int(vocab), path


def stage_ids():
    cache = json.loads(IDS_CACHE.read_text()) if IDS_CACHE.exists() else {}
    strings = SHARED["strings"]
    for slug, hf_id in sorted(SHARED["models"].items()):
        if slug in cache:
            continue
        try:
            cache[slug] = resolve_ids(hf_id, strings)
            print(f"[{slug:16s}] resolved {len(cache[slug])}/{len(strings)}", flush=True)
        except Exception as e:
            print(f"[{slug:16s}] TOKENIZER FAILED: {type(e).__name__}: {e}", flush=True)
        IDS_CACHE.write_text(json.dumps(cache))
    ok = {k: v for k, v in cache.items() if v}
    common = sorted(set.intersection(*(set(v) for v in ok.values()))) if ok else []
    print(f"\n{len(ok)} models; COMMON probe = {len(common)} strings", flush=True)
    return cache, common


def stage_embed(common, only=None):
    cache = json.loads(IDS_CACHE.read_text())
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    todo = [(s, h) for s, h in sorted(SHARED["models"].items())
            if s in cache and cache[s] and (only is None or s in only)
            and not (s in res and "error" not in res[s])]
    print(f"{len(todo)} models to do", flush=True)
    for slug, hf_id in todo:
        blob = None
        try:
            ids = [cache[slug][s] for s in common]
            Us, d, vocab, blob = stream_probe_rows(hf_id, ids)
            Uc = Us - Us.mean(0, keepdims=True)
            ev = np.clip(np.linalg.eigvalsh(Uc @ Uc.T)[::-1], 0, None)
            pr = float((ev.sum() ** 2) / (np.square(ev).sum() + 1e-12))
            cap = min(len(ids), d)
            res[slug] = {"slug": slug, "hf_id": hf_id, "vocab": vocab, "d": d,
                         "n_common": len(ids), "n_resolvable": len(cache[slug]),
                         "vocab_over_d": round(vocab / d, 1),
                         "readout_pr": round(pr, 2), "pr_norm": round(pr / cap, 4),
                         "top1_share": round(float(ev[0] / ev.sum()), 4),
                         "top16_share": round(float(ev[:16].sum() / ev.sum()), 4)}
            print(f"[{slug:16s}] d={d:5d} vocab={vocab:6d} v/d={vocab/d:6.1f} "
                  f"PR={pr:8.1f} PR/cap={pr/cap:.4f} top16={res[slug]['top16_share']:.3f}",
                  flush=True)
            del Us, Uc, ev
        except Exception as e:
            print(f"[{slug:16s}] FAILED: {type(e).__name__}: {str(e)[:90]}", flush=True)
            res[slug] = {"slug": slug, "hf_id": hf_id, "error": f"{type(e).__name__}: {str(e)[:120]}"}
        finally:
            # HF blobs have NO file extension, so match on the blobs/ directory, not a suffix.
            try:
                if blob:
                    real = Path(blob).resolve()
                    if "/blobs/" in str(real) and real.is_file():
                        real.unlink()
            except Exception:
                pass
            gc.collect()
        RESULTS.write_text(json.dumps(res, indent=1))
    print("ZOO_CONCENTRATION_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["ids", "embed", "all"])
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    cache, common = stage_ids()
    (OUT / "zoo_common_probe.json").write_text(json.dumps(common))
    if a.stage in ("embed", "all"):
        stage_embed(common, only=a.only)


if __name__ == "__main__":
    main()
