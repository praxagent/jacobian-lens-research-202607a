"""Loading: lenses, unembedding rows, probes, receipts. No model forward pass is ever run and
the full model is never downloaded; only the (un)embedding tensor's rows are read from the
safetensors shard that holds it (vendored from experiments/jacobian_lens/unembed.py and
experiments/geometry_causality/run_geometry_causality.py::probe_rows_inline)."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import numpy as np
import torch

NEURONPEDIA_REPO = "neuronpedia/jacobian-lens"
DATA = Path(__file__).resolve().parent / "data"
SHARED_TOKENS = DATA / "shared_tokens.json"

# readout matrix first (lm_head), then tied-embedding names, most specific first
UNEMBED_NAMES = ("lm_head.weight", "model.embed_tokens.weight", "embed_tokens.weight",
                 "transformer.wte.weight", "wte.weight", "gpt_neox.embed_in.weight",
                 "model.embed_in.weight")
UNEMBED_SUFFIXES = ("lm_head.weight", "embed_tokens.weight", "wte.weight", "embed_in.weight",
                    "tok_embeddings.weight")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _revision_of(path: str):
    m = re.search(r"/snapshots/([0-9a-f]{40})/", str(path))
    return m.group(1) if m else None


# ---- lenses --------------------------------------------------------------------------------
def load_lens_file(path):
    """(J: {int layer: fp16/fp32 tensor (d_final, d_layer)}, meta dict). jlens format: a dict
    with 'J' and usually 'd_model', 'source_layers', 'n_prompts'."""
    try:
        d = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        d = torch.load(path, map_location="cpu", weights_only=False)
    if "J" not in d:
        raise ValueError(f"{path}: not a jlens checkpoint (no 'J')")
    J = {int(l): t for l, t in d["J"].items()}
    meta = {k: (v if isinstance(v, (int, float, str, list)) else str(type(v).__name__))
            for k, v in d.items() if k != "J"}
    meta["d_final"] = int(next(iter(J.values())).shape[0])
    meta["dtype_stored"] = str(next(iter(J.values())).dtype).replace("torch.", "")
    return J, meta


def resolve_neuronpedia(slug: str):
    """(lens_path, hf_model_id, revision) for a Neuronpedia collection slug, via the slug's own
    config.yaml (vendored from experiments/jacobian_lens/cka_layers.py::resolve)."""
    import yaml
    from huggingface_hub import HfApi, hf_hub_download
    files = [f for f in HfApi().list_repo_files(NEURONPEDIA_REPO) if f.startswith(slug + "/")]
    if not files:
        raise SystemExit(f"no files for slug {slug!r} in {NEURONPEDIA_REPO}")
    lens_file = next(f for f in files if f.endswith(".pt"))
    cfg_file = next(f for f in files if f.endswith("config.yaml"))
    cfg = yaml.safe_load(open(hf_hub_download(NEURONPEDIA_REPO, filename=cfg_file)))
    path = hf_hub_download(NEURONPEDIA_REPO, filename=lens_file)
    return path, cfg["hf_model_name"], _revision_of(path)


def download_hf_file(repo_id: str, filename: str, revision=None):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id, filename=filename, revision=revision)
    return path, _revision_of(path)


# ---- unembedding rows ----------------------------------------------------------------------
def _locate_unembedding(hf_id: str):
    """(local safetensors path, tensor name). Prefers lm_head.weight when the model has one
    (untied readout), otherwise the tied embedding."""
    from huggingface_hub import HfApi, hf_hub_download
    files = HfApi().list_repo_files(hf_id)
    idx = next((f for f in files if f.endswith("model.safetensors.index.json")), None)
    if idx:
        wmap = json.load(open(hf_hub_download(hf_id, idx)))["weight_map"]
        name = _pick(wmap.keys())
        return hf_hub_download(hf_id, wmap[name]), name
    single = [f for f in files if f.endswith(".safetensors")]
    if not single:
        raise SystemExit(f"{hf_id}: no safetensors weights found")
    path = hf_hub_download(hf_id, sorted(single)[0])
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        name = _pick(f.keys())
    return path, name


def _pick(keys) -> str:
    keys = set(keys)
    for n in UNEMBED_NAMES:
        if n in keys:
            return n
    for k in keys:
        if k.endswith(UNEMBED_SUFFIXES):
            return k
    raise KeyError(f"no unembedding tensor among {len(keys)} tensors")


def unembedding_shape(hf_id: str):
    """(vocab, d, tensor_name) without reading the tensor."""
    from safetensors import safe_open
    path, name = _locate_unembedding(hf_id)
    with safe_open(path, framework="pt") as f:
        vocab, d = f.get_slice(name).get_shape()
    return int(vocab), int(d), name, path


def unembedding_rows(hf_id: str, ids) -> tuple[np.ndarray, str]:
    """Only the requested rows of the (un)embedding, as float32 (n, d), read in row chunks so a
    256k x 5k matrix is never materialised. Returns (rows in the order of `ids`, tensor name)."""
    from safetensors import safe_open
    path, name = _locate_unembedding(hf_id)
    ids = np.asarray(ids, dtype=np.int64)
    with safe_open(path, framework="pt") as f:
        sl = f.get_slice(name); vocab, d = sl.get_shape()
        order = np.argsort(ids); sid = ids[order]
        out = np.empty((len(ids), d), dtype=np.float32); pos = 0
        for st in range(0, vocab, 8192):
            sp = min(st + 8192, vocab)
            want = sid[(sid >= st) & (sid < sp)]
            if want.size == 0:
                continue
            blk = sl[st:sp].to(torch.float32).numpy()
            out[order[pos:pos + want.size]] = blk[want - st]; pos += want.size
    return out, name


# ---- probes --------------------------------------------------------------------------------
def resolve_shared_ids(hf_id: str, strings) -> dict:
    """Canonical rule (experiments/jacobian_lens/shared_vocab.py::resolve_ids): map each shared
    string to a single token id, space-prefixed form preferred, bare form fallback; drop strings
    that need more than one token in this tokenizer."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    ids = {}
    for s in strings:
        for form in (" " + s, s):
            enc = tok.encode(form, add_special_tokens=False)
            if len(enc) == 1:
                ids[s] = enc[0]
                break
    return ids


def build_probe(kind: str, hf_id: str, n_probe: int = 4096, seed: int = 0,
                shared_tokens_path=None):
    """Returns (U_probe float32 numpy (n, d), info). 'own': n_probe vocabulary rows sampled
    once with numpy default_rng(seed), no replacement (atlas_stage_a convention). 'shared':
    the 4,096 shared strings that resolve to one token in this tokenizer, in file order."""
    if kind == "own":
        vocab, d, name, _ = unembedding_shape(hf_id)
        rng = np.random.default_rng(seed)
        ids = rng.choice(vocab, size=min(n_probe, vocab), replace=False)
        rows, name = unembedding_rows(hf_id, ids)
        return rows, {"probe": "own", "n_probe": int(len(ids)), "seed": seed, "vocab": vocab,
                      "unembedding_tensor": name}
    if kind == "shared":
        spec = json.load(open(shared_tokens_path or SHARED_TOKENS))
        strings = spec["strings"]
        ids_map = resolve_shared_ids(hf_id, strings)
        kept = [s for s in strings if s in ids_map]
        rows, name = unembedding_rows(hf_id, [ids_map[s] for s in kept])
        return rows, {"probe": "shared", "n_probe": int(len(kept)), "n_shared_strings": len(strings),
                      "n_dropped": len(strings) - len(kept), "shared_tokens_sha256": sha256(
                          shared_tokens_path or SHARED_TOKENS), "unembedding_tensor": name}
    raise ValueError(kind)
