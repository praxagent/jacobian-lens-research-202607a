"""GPU-efficient battery readout for the Llama-3.3-70B replication.

demo2's readout (via jlens.lens.apply) does the J-transport on CPU — pathological for the
d=8192 Llama lens (~3+ min/condition). This runner uses the GPU-transport pattern validated
in peek_thinking.py: capture activations once, move each band layer's J to that layer's
device (h.device under device_map), do h@J.T + unembed on GPU. Identity + random-J controls
are generated inline per band layer (no slow deepcopy). Output matches what extract/analyze
expect (per condition: family, prompt, continuation, model_head, lenses.{jlens,logit_lens,
random_J}.probe_best_rank + probe_rank_by_layer). SEPARATE namespace (llama70b/).
"""
import argparse, json, sys
from pathlib import Path
import torch, transformers, jlens
from jlens import Layout
from jlens.hooks import ActivationRecorder

HERE = Path(__file__).resolve().parent           # llama70b/
sys.path.insert(0, str(HERE.parent))             # lens_demo/ for demo2 helpers
from demo2 import greedy_continue, single_token_id, band_layers   # reuse validated helpers


def resolve_probes(tok, lexicons):
    resolved = {}
    for name, words in lexicons.items():
        for w in words:
            tid = single_token_id(tok, w)
            if tid is not None:
                resolved[w] = tid
    return resolved   # flat {word: token_id}


@torch.no_grad()
def readout_gpu(model, lens, band, final, probe_ids, prompt_ids, topk, seed=0):
    """span readout, GPU transport. Returns per-lens probe_best_rank (min over layer x pos),
    probe_rank_by_layer (min over pos per layer), and a jlens cloud at the peak layer."""
    ids = torch.tensor([prompt_ids], device=model.input_device)
    record_at = sorted(set(band) | {final})
    with ActivationRecorder(model.layers, at=record_at) as rec:
        model.forward(ids)
        acts = {i: rec.activations[i].detach() for i in record_at}
    d = acts[band[0]].shape[-1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    names = ["jlens", "logit_lens", "random_J"]
    best = {ln: {w: None for w in probe_ids} for ln in names}
    by_layer = {ln: {w: {} for w in probe_ids} for ln in names}
    cloud_by_layer = {}   # jlens topk per layer (peak-position)
    for l in band:
        h = acts[l][0].float()                                  # [n_pos, d] on GPU
        J = lens.jacobians[l].to(h.device).float()
        R = torch.randn(d, d, generator=g).to(h.device); R = R * (J.norm() / R.norm())
        for ln in names:
            z = h @ J.T if ln == "jlens" else (h if ln == "logit_lens" else h @ R.T)
            logits = model.unembed(z).float()                   # [n_pos, vocab] GPU
            for w, tid in probe_ids.items():
                rk = (logits > logits[:, tid:tid + 1]).sum(-1) + 1   # [n_pos]
                mn = int(rk.min().item())
                by_layer[ln][w][str(l)] = mn
                if best[ln][w] is None or mn < best[ln][w]:
                    best[ln][w] = mn
            if ln == "jlens":
                peak_pos = int(logits.max(-1).values.argmax().item())
                tk = logits[peak_pos].topk(topk)
                cloud_by_layer[str(l)] = {"ids": tk.indices.cpu().tolist(),
                                          "scores": [round(float(x), 3) for x in tk.values.cpu()]}
            del logits, z
        del h, J, R
    return best, by_layer, cloud_by_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big-model", default="meta-llama/Llama-3.3-70B-Instruct:")
    ap.add_argument("--lens-hf", default="neuronpedia/jacobian-lens:llama3.3-70b-it/jlens/Salesforce-wikitext/Llama-3.3-70B-Instruct_jacobian_lens.pt")
    ap.add_argument("--lens-file", default=None)
    ap.add_argument("--prompts-file", default=str(HERE / "prompts_wc_llama.json"))
    ap.add_argument("--continue-tokens", type=int, default=128)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--lens-fit-n", type=int, default=125)
    ap.add_argument("--only", default=None, help="comma-separated condition ids (smoke)")
    ap.add_argument("--out", default=str(HERE / "demo2_wc_llama33-70b.json"))
    args = ap.parse_args()

    hf_id, backbone = args.big_model.split(":", 1)
    cfg = transformers.AutoConfig.from_pretrained(hf_id)
    cls = getattr(transformers, cfg.architectures[0])
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        hf = cls.from_pretrained(hf_id, torch_dtype=torch.float32, attn_implementation="eager").eval()
    else:
        hf = cls.from_pretrained(hf_id, torch_dtype=torch.bfloat16, device_map="auto",
                                 attn_implementation="eager",
                                 max_memory={i: "125GiB" for i in range(n_gpu)}).eval()
    tok = transformers.AutoTokenizer.from_pretrained(hf_id)
    model = (jlens.from_hf(hf, tok, compile=False, layout=Layout(path=backbone))
             if backbone else jlens.from_hf(hf, tok, compile=False))

    if args.lens_file:
        lens_path = args.lens_file
    else:
        from huggingface_hub import hf_hub_download
        repo, fname = args.lens_hf.split(":", 1)
        lens_path = hf_hub_download(repo, fname)
    lens = jlens.JacobianLens.load(lens_path)
    band = band_layers(lens.source_layers); final = model.n_layers - 1
    d_model = hf.get_output_embeddings().weight.shape[1]
    lens_d = next(iter(lens.jacobians.values())).shape[0]
    if lens_d != d_model:
        raise SystemExit(f"LENS/MODEL MISMATCH lens_d={lens_d} d_model={d_model}")
    print(f"band={band} ({len(band)} layers), final={final}, d={d_model}", flush=True)

    spec = json.load(open(args.prompts_file))
    probe_ids = resolve_probes(tok, spec["probe_lexicons"])
    print(f"resolved {len(probe_ids)} single-token probes", flush=True)

    only = set(args.only.split(",")) if args.only else None
    items = []
    for c in spec["conditions"]:
        if only and c["id"] not in only: continue
        cont, cont_ids, head = greedy_continue(hf, model, c["prompt"], args.continue_tokens, tok)
        prompt_ids = tok(c["prompt"], add_special_tokens=True)["input_ids"]
        if isinstance(prompt_ids[0], list): prompt_ids = prompt_ids[0]
        best, by_layer, cloud = readout_gpu(model, lens, band, final, probe_ids, prompt_ids, args.topk)
        items.append({
            "id": c["id"], "family": c.get("family"), "prompt": c["prompt"],
            "true_answer": c.get("true_answer"), "tempting_answer": c.get("tempting_answer"),
            "is_control": c.get("is_control"),
            "continuation": cont, "continuation_ids": cont_ids,
            "model_head": {"steps": head},
            "lenses": {ln: {"probe_best_rank": best[ln], "probe_rank_by_layer": by_layer[ln]}
                       for ln in best},
            "jlens_cloud_by_layer": cloud,
        })
        print(f"  {c['id']:16s} jlens survivalclean-ish done; cont={cont[:40]!r}", flush=True)
        json.dump({"model": hf_id, "lens_fit_n": args.lens_fit_n, "band": band, "items": items},
                  open(args.out, "w"))   # incremental
    print(f"wrote {args.out} ({len(items)} items)", flush=True)


if __name__ == "__main__":
    main()
