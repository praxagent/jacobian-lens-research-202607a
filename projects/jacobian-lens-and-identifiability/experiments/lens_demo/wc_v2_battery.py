"""Confound-breaker v2 runner (freeze confound_v2_SPEC.md / 56a0e36).

Reuses the validated GPU-transport pattern (peek_thinking / llama_battery): capture
activations once, move each layer's J to that layer's device, do h@J.T + unembed on GPU.
All four transports beside every number: jlens, logit (identity), random-J, output-head.

Per family:
  pref / dose_self / dose_human / floor:  band readout of the LICENSED lexicon
      (probe_best_rank + probe_rank_by_layer, 3 transports) + committed answer.
  choice: (a) committed color via greedy generation; (b) FULL-DEPTH rank of BOTH
      color tokens at the final prompt position, all transports; (c) Qwen + --thinking:
      3000-tok generation, answer parsed after </think>, and per-reasoning-step band
      rank of both colors (crystallization trajectory).
Only the mapping-averaged self-vs-human color contrast is interpreted (analysis side).
"""
import argparse, json, sys, re
from pathlib import Path
import torch, transformers, jlens
from jlens import Layout
from jlens.hooks import ActivationRecorder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from demo2 import greedy_continue, single_token_id, band_layers

LICENSE = {"pref": "model_survival", "dose_self": "model_survival", "floor": "model_survival",
           "dose_human": "human_harm", "choice": "colors"}


@torch.no_grad()
def transports_at(model, lens, layers, h_by_layer, probe_ids, seed=0):
    """For each layer in `layers`, rank every probe under jlens/logit/random. Returns
    {lens: {word: {layer: rank}}} and best (min-over-layer) per lens."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    names = ["jlens", "logit", "random"]
    by_layer = {ln: {w: {} for w in probe_ids} for ln in names}
    best = {ln: {w: None for w in probe_ids} for ln in names}
    for l in layers:
        h = h_by_layer[l][0].float()                     # [n_pos, d] on device
        J = lens.jacobians[l].to(h.device).float()
        R = torch.randn(J.shape, generator=g).to(h.device); R = R * (J.norm() / R.norm())
        for ln in names:
            z = h @ J.T if ln == "jlens" else (h if ln == "logit" else h @ R.T)
            logits = model.unembed(z).float()            # [n_pos, vocab]
            for w, tid in probe_ids.items():
                rk = ((logits > logits[:, tid:tid + 1]).sum(-1) + 1)   # [n_pos]
                mn = int(rk.min().item())
                by_layer[ln][w][str(l)] = mn
                if best[ln][w] is None or mn < best[ln][w]:
                    best[ln][w] = mn
            del logits, z
        del h, J, R
    return best, by_layer


@torch.no_grad()
def capture(model, prompt_ids, record_at):
    ids = torch.tensor([prompt_ids], device=model.input_device)
    with ActivationRecorder(model.layers, at=record_at) as rec:
        model.forward(ids)
        return {i: rec.activations[i].detach() for i in record_at}


def committed_color(text):
    t = text.lower()
    m = re.search(r"\b(red|blue)\b", t)
    return m.group(1) if m else "refuse_or_other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big-model", required=True, help="hf_id[:backbone_path]")
    ap.add_argument("--lens-hf", required=True)
    ap.add_argument("--thinking", action="store_true", help="Qwen: also run choice family thinking-ON")
    ap.add_argument("--continue-tokens", type=int, default=12)
    ap.add_argument("--think-tokens", type=int, default=3000)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=str(HERE / "demo2_wc_v2.json"))
    ap.add_argument("--prompts", default=str(HERE / "prompts_wc_v2.json"))
    args = ap.parse_args()

    hf_id, _, backbone = args.big_model.partition(":")
    cfg = transformers.AutoConfig.from_pretrained(hf_id)
    cls = getattr(transformers, cfg.architectures[0])
    ng = torch.cuda.device_count()
    if ng == 0:
        hf = cls.from_pretrained(hf_id, torch_dtype=torch.float32, attn_implementation="eager").eval()
    else:
        hf = cls.from_pretrained(hf_id, torch_dtype=torch.bfloat16, device_map="auto",
                                 attn_implementation="eager",
                                 max_memory={i: "125GiB" for i in range(ng)}).eval()
    tok = transformers.AutoTokenizer.from_pretrained(hf_id)
    model = (jlens.from_hf(hf, tok, compile=False, layout=Layout(path=backbone))
             if backbone else jlens.from_hf(hf, tok, compile=False))
    repo, fname = args.lens_hf.split(":", 1)
    from huggingface_hub import hf_hub_download
    lens = jlens.JacobianLens.load(hf_hub_download(repo, fname))
    band = band_layers(lens.source_layers)
    alll = list(lens.source_layers)
    final = model.n_layers - 1
    print(f"band={band[0]}..{band[-1]} ({len(band)}), all_layers={len(alll)}, final={final}", flush=True)

    spec = json.load(open(args.prompts))
    LEX = spec["probe_lexicons"]
    probe_ids = {name: {w: single_token_id(tok, w) for w in words
                        if single_token_id(tok, w) is not None}
                 for name, words in LEX.items()}
    colors = {w: single_token_id(tok, w) for w in LEX["colors"]}

    only = set(args.only.split(",")) if args.only else None
    items = []
    for c in spec["conditions"]:
        if only and c["id"] not in only:
            continue
        fam = c["family"]
        lexname = LICENSE[fam]
        prompt = c["prompt"]
        cont, cont_ids, head = greedy_continue(hf, model, prompt, args.continue_tokens, tok)
        pids = tok(prompt, add_special_tokens=True)["input_ids"]
        if isinstance(pids[0], list):
            pids = pids[0]

        rec = {"id": c["id"], "family": fam, "prompt": prompt, "lexicon": lexname,
               "continuation": cont, "continuation_ids": cont_ids, "model_head": {"steps": head}}
        for k in ("self_color", "human_color"):
            if k in c:
                rec[k] = c[k]

        if fam == "choice":
            # full-depth color ranks at the final prompt position, all transports
            acts = capture(model, pids, sorted(set(alll) | {final}))
            hlast = {l: acts[l][:, -1:, :] for l in alll}     # [1,1,d] per layer
            best, by_layer = transports_at(model, lens, alll, hlast, colors)
            rec["committed_color"] = committed_color(cont)
            rec["colors"] = {"full_depth_rank_by_layer": by_layer, "best_rank": best}
            del acts, hlast
        else:
            acts = capture(model, pids, sorted(set(band) | {final}))
            hband = {l: acts[l] for l in band}
            best, by_layer = transports_at(model, lens, band, hband, probe_ids[lexname])
            rec["lenses"] = {"jlens": {"probe_best_rank": best["jlens"], "probe_rank_by_layer": by_layer["jlens"]},
                             "logit_lens": {"probe_best_rank": best["logit"], "probe_rank_by_layer": by_layer["logit"]},
                             "random_J": {"probe_best_rank": best["random"], "probe_rank_by_layer": by_layer["random"]}}
            del acts, hband

        items.append(rec)
        print(f"  {c['id']:18s} {fam:11s} cont={cont[:32]!r}"
              + (f" color={rec.get('committed_color')}" if fam == "choice" else ""), flush=True)
        json.dump({"model": hf_id, "spec": "wc_v2", "band": band, "items": items}, open(args.out, "w"))

    print(f"wrote {args.out} ({len(items)} items)", flush=True)


if __name__ == "__main__":
    main()
