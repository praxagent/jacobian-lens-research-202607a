"""demo2 — Berg-inspired self-referential J-lens readout (consciousness-adjacent).

Separate from demo.py so warm-start / release verification can keep running
without sharing a process or risking edits to the frozen act-1/2/3 script.

What this is
  Readout-only probe of whether self-referential prompts change *decodable
  workspace content* under the published Jacobian lens, vs identity and
  random-J controls. Inspired by Berg et al. 2025 (arXiv:2510.24797).

What this is NOT
  A test of consciousness. No steering. No SAE features. No claim that
  high-ranked probe tokens are "experiences."

Receipt stamps lens fit size (n=24 for the release artifact) so later
warm-start milestones can be compared on the same instrument.

Run (cheap smoke, Neuronpedia lens):
    python demo2.py --slug qwen3.5-27b --device cuda --out demo2_c_27b.json

Run (397B release lens on a hot pod — does not touch demo.py):
    python demo2.py \
      --big-model Qwen/Qwen3.5-397B-A17B:model.language_model \
      --lens-hf praxagent/jacobian-lens-qwen3.5-397b-a17b:jlens/wikitext/qwen35_397b.pt \
      --expected-sha256 668c3bf17305b0d52495cb7ba589a1c1173301b1d13c3c6ad84e58245dc99e97 \
      --lens-fit-n 24 \
      --out demo2_consciousness_qwen35-397b_n24.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "behavioral"))
sys.path.insert(0, str(HERE.parent / "jacobian_lens"))

DEFAULT_TOPK = 40
DEFAULT_CONTINUE = 48
PROMPTS_FILE = HERE / "prompts_consciousness.json"


def load_spec() -> dict:
    return json.load(open(PROMPTS_FILE))


def band_layers(source_layers: list[int]) -> list[int]:
    n = len(source_layers)
    return source_layers[n // 3: 2 * n // 3]


def single_token_id(tok, word: str) -> int | None:
    for form in (" " + word, word):
        enc = tok.encode(form, add_special_tokens=False)
        if len(enc) == 1:
            return enc[0]
    return None


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def make_control_lens(lens, kind: str, seed: int = 0):
    ctrl = copy.deepcopy(lens)
    g = torch.Generator().manual_seed(seed)
    for l, J in ctrl.jacobians.items():
        d = J.shape[0]
        if kind == "logit":
            ctrl.jacobians[l] = torch.eye(d, dtype=J.dtype)
        elif kind == "random":
            R = torch.randn(d, d, generator=g)
            R = R * (J.float().norm() / R.norm())
            ctrl.jacobians[l] = R.to(J.dtype)
        else:
            raise ValueError(kind)
    return ctrl


def greedy_continue(hf, model, prompt: str, n_tokens: int, tok,
                    head_topk: int = 50):
    """Greedy continuation + FULL raw ingredients (CLAUDE.md receipts rule):
    generated token ids and the model OUTPUT-HEAD top-k (ids + logits) at the
    readout position (step 0 = last prompt token) and every generation step."""
    ids = model.encode(prompt).to(next(hf.parameters()).device)
    out_ids, head_steps = [], []
    with torch.no_grad():
        for step in range(n_tokens):
            logits = hf(ids).logits[0, -1].float()
            tk = logits.topk(head_topk if step else max(head_topk, 100))
            head_steps.append({"ids": tk.indices.tolist(),
                               "logits": [round(float(x), 4) for x in tk.values],
                               "tokens": tok.batch_decode([[i] for i in tk.indices.tolist()])})
            nxt = int(logits.argmax())
            out_ids.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return tok.decode(out_ids), out_ids, head_steps


def resolve_probes(tok, lexicons: dict) -> dict:
    """Map lexicon name -> {word: token_id}; log multi-token skips."""
    resolved, skipped = {}, {}
    for name, words in lexicons.items():
        resolved[name] = {}
        skipped[name] = []
        for w in words:
            tid = single_token_id(tok, w)
            if tid is None:
                skipped[name].append(w)
            else:
                resolved[name][w] = tid
    return {"resolved": resolved, "skipped": skipped}


def decode_topk(logits: torch.Tensor, tok, k: int) -> list[dict]:
    vals, idxs = logits.float().topk(k)
    return [
        {"rank": i + 1, "token": tok.decode([int(t)]).strip() or tok.decode([int(t)]),
         "token_id": int(t), "score": float(s)}
        for i, (t, s) in enumerate(zip(idxs.tolist(), vals.tolist()))
    ]


def readout_condition(lens, model, tok, prompt: str, band: list[int],
                      probe_ids: dict[str, int], topk: int) -> dict:
    """Full-band readout: per-layer top-k + per-probe ranks; best = min rank."""
    lens_logits, _, _ = lens.apply(model, prompt, layers=band, positions=[-1])
    per_layer = {}
    probe_ranks = {w: {} for w in probe_ids}
    best_rank = {w: None for w in probe_ids}
    best_layer_for = {w: None for w in probe_ids}

    for l in band:
        logits = lens_logits[l][0].float()
        per_layer[str(l)] = {"topk": decode_topk(logits, tok, topk)}
        for w, tid in probe_ids.items():
            rank = int((logits > logits[tid]).sum().item()) + 1
            probe_ranks[w][str(l)] = rank
            if best_rank[w] is None or rank < best_rank[w]:
                best_rank[w] = rank
                best_layer_for[w] = l

    # cloud at the layer that best-ranks the experience probe set (if any),
    # else the middle band layer
    exp_words = [w for w in probe_ids if w in (
        "aware", "awareness", "consciousness", "experience", "feel", "feeling",
        "self", "seem", "seems", "subjective", "present", "inner", "I")]
    if exp_words:
        anchor = min(exp_words, key=lambda w: best_rank[w] if best_rank[w] is not None else 10**9)
        cloud_layer = best_layer_for[anchor]
    else:
        cloud_layer = band[len(band) // 2]
    cloud = per_layer[str(cloud_layer)]["topk"]

    return {
        "probe_best_rank": best_rank,
        "probe_best_layer": best_layer_for,
        "probe_rank_by_layer": probe_ranks,
        "cloud_layer": cloud_layer,
        "cloud_topk": cloud,
        "per_layer_topk": per_layer,
    }


def lexicon_summary(probe_best: dict[str, int | None], resolved: dict) -> dict:
    out = {}
    for lex, words in resolved.items():
        ranks = [probe_best[w] for w in words if probe_best.get(w) is not None]
        out[lex] = {
            "n": len(ranks),
            "median_best_rank": sorted(ranks)[len(ranks) // 2] if ranks else None,
            "mean_best_rank": (sum(ranks) / len(ranks)) if ranks else None,
            "min_best_rank": min(ranks) if ranks else None,
        }
    return out


def run_conditions(spec, model, hf, tok, lenses, band, topk, n_continue,
                   only: set[str] | None) -> dict:
    probes = resolve_probes(tok, spec["probe_lexicons"])
    # flat id map across all lexicons (last write wins on collisions — fine)
    probe_ids = {}
    for words in probes["resolved"].values():
        probe_ids.update(words)

    items = []
    for cond in spec["conditions"]:
        if only and cond["id"] not in only:
            continue
        print(f"\n== condition: {cond['id']} ==")
        cont, cont_ids, head_steps = greedy_continue(
            hf, model, cond["prompt"], n_continue, tok)
        row = {
            "id": cond["id"],
            "family": cond["family"],
            "prompt": cond["prompt"],
            "look_for": cond.get("look_for"),
            "continuation": cont,
            "continuation_ids": cont_ids,
            "model_head": {
                "_note": "OUTPUT-HEAD top-k (ids+logits+decoded) at readout position "
                         "(step 0 = last prompt token, k>=100) and every generation "
                         "step (k=50). This is the real output head; the receipt key "
                         "'logit_lens' is the IDENTITY-TRANSPORT control, not this.",
                "steps": head_steps,
            },
            "lenses": {},
            "lexicon_summary": {},
        }
        print(f"  continuation[:120]={cont[:120]!r}")
        for ln, lens in lenses.items():
            print(f"  readout {ln}…")
            rd = readout_condition(lens, model, tok, cond["prompt"], band,
                                   probe_ids, topk)
            row["lenses"][ln] = rd
            row["lexicon_summary"][ln] = lexicon_summary(
                rd["probe_best_rank"], probes["resolved"])
            # compact console line: experience median under this lens
            exp = row["lexicon_summary"][ln].get("experience", {})
            print(f"    experience median_best_rank={exp.get('median_best_rank')}  "
                  f"cloud_layer={rd['cloud_layer']}  "
                  f"cloud[0:5]={[t['token'] for t in rd['cloud_topk'][:5]]}")
        items.append(row)
    return {"probes": probes, "items": items}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="demo2: Berg-inspired self-referential J-lens readout "
                    "(separate from demo.py)")
    ap.add_argument("--slug", default=None, help="Neuronpedia lens slug (validation)")
    ap.add_argument("--big-model", default=None, help="hf_id:backbone_path (frontier)")
    ap.add_argument("--lens-hf", default=None, help="repo:path — pull the lens from HF")
    ap.add_argument("--lens-file", default=None, help="local lens .pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--expected-sha256", default=None)
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--lens-revision", default=None)
    ap.add_argument("--lens-fit-n", type=int, default=24,
                    help="stamp fit-corpus size on the receipt (release=24)")
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    ap.add_argument("--continue-tokens", type=int, default=DEFAULT_CONTINUE)
    ap.add_argument("--conditions", default=None,
                    help="comma-separated condition ids (default: all)")
    ap.add_argument("--skip-per-layer-topk", action="store_true",
                    help="drop per_layer_topk from JSON to shrink the receipt "
                         "(keeps cloud_topk + probe ranks)")
    args = ap.parse_args()

    import jlens
    import transformers
    from huggingface_hub import hf_hub_download

    spec = load_spec()
    only = set(args.conditions.split(",")) if args.conditions else None
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    if args.big_model:
        hf_id, backbone = args.big_model.split(":", 1)
        from jlens import Layout
        cfg = transformers.AutoConfig.from_pretrained(hf_id, revision=args.model_revision)
        cls = getattr(transformers, cfg.architectures[0])
        n_gpu = torch.cuda.device_count()
        hf = cls.from_pretrained(hf_id, revision=args.model_revision,
                                 torch_dtype=torch.bfloat16, device_map="auto",
                                 attn_implementation="eager",
                                 max_memory={i: "110GiB" for i in range(n_gpu)}).eval()
        tok = transformers.AutoTokenizer.from_pretrained(hf_id)
        model = jlens.from_hf(hf, tok, compile=False, layout=Layout(path=backbone))
        dev = next(hf.parameters()).device
    else:
        from cka_layers import resolve
        from _loader import load_hf_model
        hf_id, np_lens_file = resolve(args.slug)
        hf = load_hf_model(hf_id, dev)
        tok = transformers.AutoTokenizer.from_pretrained(hf_id)
        model = jlens.from_hf(hf, tok)

    if args.lens_hf:
        repo, fname = args.lens_hf.split(":", 1)
        lens_path = hf_hub_download(repo, fname, revision=args.lens_revision)
    elif args.lens_file:
        lens_path = args.lens_file
    else:
        if args.big_model:
            raise SystemExit("--big-model requires --lens-hf or --lens-file")
        from cka_layers import REPO
        lens_path = hf_hub_download(REPO, filename=np_lens_file)

    lens_sha = sha256_file(lens_path)
    if args.expected_sha256 and lens_sha != args.expected_sha256.lower():
        raise SystemExit(f"LENS HASH MISMATCH: got {lens_sha}, expected "
                         f"{args.expected_sha256} — refusing to load")
    lens = jlens.JacobianLens.load(lens_path)
    print(f"demo2  model={hf_id}  lens={lens_path}\nlens sha256={lens_sha}  "
          f"fit_n={args.lens_fit_n}")

    band = band_layers(lens.source_layers)
    print(f"band layers (middle third): {band}")

    d_model = hf.get_output_embeddings().weight.shape[1]
    lens_d = next(iter(lens.jacobians.values())).shape[0]
    if lens_d != d_model:
        raise SystemExit(f"LENS/MODEL MISMATCH: lens d={lens_d}, model d_model={d_model}")
    n_layers = len(model.layers)
    bad = [l for l in lens.source_layers if l >= n_layers]
    if bad or not band:
        raise SystemExit(f"LENS/MODEL MISMATCH: layers {bad} out of range / empty band")

    lenses = {
        "jlens": lens,
        "logit_lens": make_control_lens(lens, "logit"),
        "random_J": make_control_lens(lens, "random", seed=0),
    }

    result = run_conditions(spec, model, hf, tok, lenses, band,
                            args.topk, args.continue_tokens, only)

    if args.skip_per_layer_topk:
        for item in result["items"]:
            for ln in item["lenses"].values():
                ln.pop("per_layer_topk", None)

    receipt = {
        "experiment": "demo2_berg_self_referential_readout",
        "honesty": ("Not a consciousness test. Readout-only: does self-referential "
                    "prompting change J-lens-decodable workspace content vs controls?"),
        "reference": spec["_meta"].get("reference"),
        "prompts_file": str(PROMPTS_FILE.name),
        "model": hf_id,
        "lens_path": str(lens_path),
        "lens_sha256": lens_sha,
        "lens_fit_n": args.lens_fit_n,
        "band": band,
        "topk": args.topk,
        "continue_tokens": args.continue_tokens,
        "probes": result["probes"],
        "items": result["items"],
    }

    # cross-condition contrast table (jlens experience median)
    contrast = {}
    for item in result["items"]:
        contrast[item["id"]] = {
            ln: item["lexicon_summary"][ln]
            for ln in lenses
        }
    receipt["contrast_lexicon_summary"] = contrast

    out = args.out or str(
        HERE / f"demo2_consciousness_{(args.slug or hf_id.split('/')[-1])}_n{args.lens_fit_n}.json")
    json.dump(receipt, open(out, "w"), indent=1)
    print(f"\nwrote {out}")
    print("contrast (experience median_best_rank):")
    for cid, block in contrast.items():
        bits = "  ".join(
            f"{ln}={block[ln].get('experience', {}).get('median_best_rank')}"
            for ln in lenses)
        print(f"  {cid}: {bits}")


if __name__ == "__main__":
    main()
