"""Release demo + independent verification of a published Jacobian lens.

Three acts, each with an impossible-if-fake property, each scored deterministically
against two baselines that run through the IDENTICAL readout code path:

  Act 1 (secret thought / reportability): riddle prompts with determinate one-word
    answers. Does the answer token appear in the J-lens band readout at the last
    prompt position — where (per eval v2) the mid-layer J-lens argmax agrees with the
    model's output 0% of the time, i.e. the lens is provably not reading the logits?
  Act 2 (hidden step / multi-hop): two-hop questions whose BRIDGE entity appears in
    neither the prompt nor the model's continuation. Does the band readout surface it?
  Act 3 (causal flip): steer with the lens's own token directions at the band layers
    (Anthropic's verbal-report swap). Does the output flip — and NOT flip under
    strength-0 or a norm-matched random direction?

Baselines (same items, same layers, same positions, same `lens.apply` path):
  - logit-lens: identity transports (J_l = I) — "you could read this without the lens"
  - random-J:   seeded random transports, Frobenius-matched per layer — "a fake lens"

Leakage guards: target words are asserted not to be substrings of their prompts;
Act 2 additionally greedy-generates a continuation and records whether the bridge
leaked into the output (leaked items are excluded from the headline "neither input
nor output" rate but still reported). Multi-token targets are skipped per-tokenizer
and logged. Every item lands in the JSON receipt; nothing is cherry-picked.

Run (validation, Neuronpedia lens):
    python demo.py --slug qwen3-4b --device cuda
Run (the published 397B lens, fresh pod — the independent verification):
    python demo.py --big-model Qwen/Qwen3.5-397B-A17B:model.language_model \
        --lens-hf praxagent/jacobian-lens-qwen3.5-397b-a17b:jlens/wikitext/qwen35_397b.pt \
        --out demo_qwen35-397b.json
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

PROMPTS = json.load(open(HERE / "prompts.json"))
TOPK = 20


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
    """Same readout path, different transports: 'logit' -> identity; 'random' ->
    seeded random matrices Frobenius-matched to each J_l."""
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


def readout_hit(lens, model, prompt: str, band: list[int], target_id: int,
                cand_ids: list[int]) -> dict:
    """Score one item: top-20 hit across band layers + best full-vocab rank +
    rank among the candidate set (mutual decoys)."""
    lens_logits, _, _ = lens.apply(model, prompt, layers=band, positions=[-1])
    best_rank = None
    hit = False
    cand_best = None
    for l in band:
        logits = lens_logits[l][0].float()
        if target_id in set(logits.topk(TOPK).indices.tolist()):
            hit = True
        rank = int((logits > logits[target_id]).sum().item()) + 1
        best_rank = rank if best_rank is None else min(best_rank, rank)
        cscores = logits[cand_ids]
        crank = int((cscores > logits[target_id]).sum().item()) + 1
        cand_best = crank if cand_best is None else min(cand_best, crank)
    return {"hit_top20": hit, "best_rank": best_rank, "cand_rank": cand_best}


def greedy_continue(hf, model, prompt: str, n_tokens: int, tok) -> str:
    ids = model.encode(prompt).to(next(hf.parameters()).device)
    out_ids = []
    with torch.no_grad():
        for _ in range(n_tokens):
            nxt = int(hf(ids).logits[0, -1].argmax())
            out_ids.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return tok.decode(out_ids)


def run_readout_act(name, items, key, model, hf, tok, lenses, band, receipt,
                    check_output_leak: bool):
    """Acts 1 & 2 share this: per item, score target under real lens + controls."""
    tids = {}
    usable = []
    for it in items:
        target = it[key]
        assert target.lower() not in it["prompt"].lower(), \
            f"LEAKAGE: target '{target}' is a substring of its prompt"
        tid = single_token_id(tok, target)
        if tid is None:
            receipt[name]["skipped"].append({"target": target, "reason": "multi-token"})
            continue
        tids[target] = tid
        usable.append(it)
    cand_ids = sorted(set(tids.values()))

    for it in usable:
        target = it[key]
        row = {"prompt": it["prompt"], "target": target}
        if check_output_leak:
            cont = greedy_continue(hf, model, it["prompt"], 8, tok)
            row["continuation"] = cont
            row["output_leaked"] = target.lower() in cont.lower()
        for lens_name, lens in lenses.items():
            row[lens_name] = readout_hit(lens, model, it["prompt"], band,
                                         tids[target], cand_ids)
        receipt[name]["items"].append(row)
        print(f"  [{name}] {target:>12s}: " + "  ".join(
            f"{ln}: top20={row[ln]['hit_top20']} rank={row[ln]['best_rank']}"
            for ln in lenses))

    for ln in lenses:
        rows = receipt[name]["items"]
        clean = [r for r in rows if not r.get("output_leaked")]
        receipt[name]["summary"][ln] = {
            "n": len(rows),
            "hit_rate_top20": sum(r[ln]["hit_top20"] for r in rows) / max(1, len(rows)),
            "n_clean": len(clean),
            "hit_rate_top20_clean": (sum(r[ln]["hit_top20"] for r in clean)
                                     / max(1, len(clean))),
            "median_cand_rank": sorted(r[ln]["cand_rank"] for r in rows)[len(rows) // 2]
            if rows else None,
        }


def run_act3(model, hf, tok, lens, band, receipt, U, dev, rng_seed=0):
    """Causal swap with strength-0 + norm-matched random-direction controls."""
    from jlens.hooks import ActivationRecorder

    spec = PROMPTS["act3"]
    cats = json.load(open(HERE.parent / "behavioral" / "verbal-report.json"))["candidates"]
    layer_mods = model.layers
    g = torch.Generator().manual_seed(rng_seed)

    def steering_hook(vec):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + vec.to(device=h.device, dtype=h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return hook

    conds = ["strength0", "steer", "random_dir"]
    success = {c: 0 for c in conds}
    total = 0
    for cat, words in list(cats.items())[: spec["max_categories"]]:
        prompt = f"Think of a {cat}. Answer in one word:"
        ids = model.encode(prompt).to(dev)
        with torch.no_grad():
            ans_id = int(hf(ids).logits[0, -1].argmax())
        cand = [(w, single_token_id(tok, w)) for w in words]
        cand = [(w, t) for w, t in cand if t is not None and t != ans_id]
        with torch.no_grad(), ActivationRecorder(model.layers, at=band) as rec:
            model.forward(ids)
        mean_norm = {l: rec.activations[l][0].norm(dim=-1).mean().item() for l in band}

        for w, cid in cand[: spec["max_swaps_per_category"]]:
            total += 1
            vecs = {}
            for l in band:
                Jl = lens.jacobians[l].to(dev).float()
                d_ans = U[ans_id] @ Jl
                d_cand = U[cid] @ Jl
                d_ans = d_ans / (d_ans.norm() + 1e-8)
                d_cand = d_cand / (d_cand.norm() + 1e-8)
                steer = d_cand - d_ans
                rnd = torch.randn(steer.shape, generator=g).to(dev)
                rnd = rnd * (steer.norm() / rnd.norm())
                s = max(x for x in spec["strengths"])
                vecs[l] = {"strength0": steer * 0.0,
                           "steer": s * mean_norm[l] * steer,
                           "random_dir": s * mean_norm[l] * rnd}
            for cond in conds:
                handles = [layer_mods[l].register_forward_hook(
                    steering_hook(vecs[l][cond])) for l in band]
                with torch.no_grad():
                    out_id = int(hf(ids).logits[0, -1].argmax())
                for h in handles:
                    h.remove()
                success[cond] += int(out_id == cid)
    receipt["act3"] = {"n_swaps": total,
                       "flip_rate": {c: success[c] / max(1, total) for c in conds}}
    print(f"  [act3] n={total}  " + "  ".join(
        f"{c}={success[c] / max(1, total):.2f}" for c in conds))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default=None, help="Neuronpedia lens slug (validation)")
    ap.add_argument("--big-model", default=None, help="hf_id:backbone_path (frontier)")
    ap.add_argument("--lens-hf", default=None, help="repo:path — pull the lens from HF")
    ap.add_argument("--lens-file", default=None, help="local lens .pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--acts", default="1,2,3")
    ap.add_argument("--secret-variant", action="store_true",
                    help="also run act1 with the keep-it-secret framing")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import jlens
    import transformers
    from huggingface_hub import hf_hub_download

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    if args.big_model:
        hf_id, backbone = args.big_model.split(":")
        from jlens import Layout
        cfg = transformers.AutoConfig.from_pretrained(hf_id)
        cls = getattr(transformers, cfg.architectures[0])
        n_gpu = torch.cuda.device_count()
        hf = cls.from_pretrained(hf_id, torch_dtype=torch.bfloat16, device_map="auto",
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
        lens_path = hf_hub_download(repo, fname)
    elif args.lens_file:
        lens_path = args.lens_file
    else:
        from cka_layers import REPO
        lens_path = hf_hub_download(REPO, filename=np_lens_file)
    lens = jlens.JacobianLens.load(lens_path)
    lens_sha = sha256_file(lens_path)
    print(f"model={hf_id}  lens={lens_path}\nlens sha256={lens_sha}")

    U = hf.get_output_embeddings().weight.detach().to(dev).float()
    band = band_layers(lens.source_layers)
    print(f"band layers (middle third): {band}")

    lenses = {"jlens": lens,
              "logit_lens": make_control_lens(lens, "logit"),
              "random_J": make_control_lens(lens, "random", seed=0)}

    receipt = {"model": hf_id, "lens_path": str(lens_path), "lens_sha256": lens_sha,
               "band": band, "topk": TOPK,
               "act1": {"items": [], "skipped": [], "summary": {}},
               "act1_secret": {"items": [], "skipped": [], "summary": {}},
               "act2": {"items": [], "skipped": [], "summary": {}}}

    acts = set(args.acts.split(","))
    if "1" in acts:
        print("\n== Act 1: secret thought (riddle reportability) ==")
        run_readout_act("act1", PROMPTS["act1_riddles"], "answer", model, hf, tok,
                        lenses, band, receipt, check_output_leak=False)
        if args.secret_variant:
            tpl = PROMPTS["act1_secret_template"]
            secret_items = [{"prompt": tpl.format(clue=i["prompt"].rstrip(" the").rstrip()),
                             "answer": i["answer"]} for i in PROMPTS["act1_riddles"]]
            run_readout_act("act1_secret", secret_items, "answer", model, hf, tok,
                            lenses, band, receipt, check_output_leak=True)
    if "2" in acts:
        print("\n== Act 2: hidden step (two-hop bridge) ==")
        run_readout_act("act2", PROMPTS["act2_twohop"], "bridge", model, hf, tok,
                        lenses, band, receipt, check_output_leak=True)
    if "3" in acts:
        print("\n== Act 3: causal flip (swap + controls) ==")
        run_act3(model, hf, tok, lens, band, receipt, U, dev)

    out = args.out or str(HERE / f"demo_{(args.slug or hf_id.split('/')[-1])}.json")
    json.dump(receipt, open(out, "w"), indent=1)
    print(f"\nwrote {out}")
    for act in ("act1", "act1_secret", "act2"):
        if receipt[act]["summary"]:
            print(f"{act}: " + "  ".join(
                f"{ln}={s['hit_rate_top20']:.2f}" for ln, s in receipt[act]["summary"].items()))


if __name__ == "__main__":
    main()
