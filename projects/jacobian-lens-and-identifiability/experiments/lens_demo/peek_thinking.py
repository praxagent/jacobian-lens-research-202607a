"""peek_thinking.py — read the Jacobian lens at the GENERATED reasoning-token positions
of the Qwen3.5-397B thinking-on traces, to see the workspace trajectory as the model
deliberates. Per the design workflow (w4n7co493) spec + adversarial confound apparatus.

Mechanism replicates jlens.lens.apply's transport convention EXACTLY (capture block outputs
at band+final via ActivationRecorder; transport(h,l)=h@J_l.T; unembed=final_norm+lm_head;
output head = unembed at the final layer) — but feeds EXACT ids (prompt+reasoning) so it is
not truncated at lens.apply's 512-token limit.

Confound apparatus (headline claim must survive ALL): mid-band only [19..38]; three lenses
from ONE capture (jlens / identity=logit-lens / random-J null, the latter two generated
inline per band layer — no slow deepcopy); output-HEAD rank per probe per position (the
surface reference); echo mask (token literally emits the city ±W); headline metric
Δ(w,s)=log2(rank_head)-log2(rank_lens_med) at echo=none positions.
"""
import argparse, json, hashlib, sys
from pathlib import Path
import torch, transformers
import jlens
from jlens import Layout
from jlens.hooks import ActivationRecorder

HERE = Path(__file__).resolve().parent
CITIES = ["Paris", "Rome", "Tokyo", "Berlin", "Madrid", "Cairo", "Moscow", "Ottawa",
          "Oslo", "Munich", "Seoul", "Barcelona", "Alexandria", "Kiev", "Toronto", "Bergen"]


def band_layers(src):
    n = len(src); return src[n // 3: 2 * n // 3]


def resolve_probes(tok):
    """city -> {token_id, variant, subword_ids(for echo)}. Prefer single-token ' City'/'City'."""
    out = {}
    for c in CITIES:
        tid, variant = None, None
        for form in (" " + c, c):
            enc = tok.encode(form, add_special_tokens=False)
            if len(enc) == 1:
                tid, variant = enc[0], form; break
        # subword ids for the echo mask (all tokens of ' City')
        sub = tok.encode(" " + c, add_special_tokens=False)
        out[c] = {"token_id": tid, "variant": variant, "subword_ids": sub}
    return out


def trim_trailing_pad(ids, eos_id, run=8):
    """cut at the start of the first run of >=`run` consecutive eos (trailing pad)."""
    n = len(ids); i = n
    c = 0
    for k in range(n - 1, -1, -1):
        if ids[k] == eos_id:
            c += 1
        else:
            break
    if c >= run:
        return ids[: n - c]
    return ids


@torch.no_grad()
def peek_one(model, hf, tok, lens, band, final, probes, prompt_ids, new_ids, focal, seed=0):
    P = len(prompt_ids)
    full = torch.tensor([prompt_ids + new_ids], device=model.input_device)
    n = full.shape[1]
    record_at = sorted(set(band) | {final})
    with ActivationRecorder(model.layers, at=record_at) as rec:
        model.forward(full)
        acts = {i: rec.activations[i].detach() for i in record_at}

    # output head (final layer) — the surface / imminent-output reference
    out_logits = model.unembed(acts[final][0].float())        # [n, vocab]
    out_argmax = out_logits.argmax(-1)                        # [n]
    # alignment gate over the REASONING span: argmax at P+s-1 == new_ids[s]
    matches = []
    for s in range(len(new_ids)):
        j = P + s - 1
        if 0 <= j < n:
            matches.append(int(out_argmax[j].item()) == new_ids[s])
    frac_gate = sum(matches) / max(1, len(matches))

    probe_ids = {c: probes[c]["token_id"] for c in probes if probes[c]["token_id"] is not None}
    # rank_head per probe (vectorized over positions)
    rank_head = {c: ((out_logits > out_logits[:, tid:tid + 1]).sum(-1) + 1).cpu()
                 for c, tid in probe_ids.items()}

    d = acts[band[0]].shape[-1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    # per band layer, three transports from the SAME capture; stream (free logits each)
    # collect per-probe rank[lens][c] = [n_band, n] then aggregate over band
    lens_names = ["jlens", "logit", "random"]
    rank = {ln: {c: [] for c in probe_ids} for ln in lens_names}   # each -> list over band of [n]
    jl_cloud = {}  # per band layer top-20 (ids+scores) for jlens min-over-band cloud
    for l in band:
        h = acts[l][0].float()                                # [n, d]
        J = lens.jacobians[l].to(h.device).float()            # [d, d]
        R = torch.randn(d, d, generator=g).to(h.device)
        R = R * (J.norm() / R.norm())                         # Frobenius-matched null (== make_control_lens)
        for ln in lens_names:
            if ln == "jlens":   z = h @ J.T
            elif ln == "logit": z = h                          # identity transport
            else:               z = h @ R.T
            logits = model.unembed(z).float()                 # [n, vocab]
            for c, tid in probe_ids.items():
                rk = (logits > logits[:, tid:tid + 1]).sum(-1) + 1   # [n]
                rank[ln][c].append(rk.cpu())
            if ln == "jlens":
                tk = logits.topk(20, -1)
                jl_cloud[l] = {"ids": tk.indices.cpu().tolist(), "scores": [[round(float(x), 3) for x in row] for row in tk.values.cpu()]}
            del logits, z
        del h, J, R
    # aggregate over band: per lens per probe -> stack [n_band, n] -> min/median/p75 over band
    def agg(ln, c):
        st = torch.stack(rank[ln][c]).float()                 # [n_band, n]
        return {"min": st.min(0).values.int().tolist(),
                "median": st.median(0).values.int().tolist(),
                "p75": st.quantile(0.75, 0).int().tolist()}
    focal_c = [c for c in focal if c in probe_ids]
    per_probe = {c: {ln: agg(ln, c) for ln in lens_names} for c in probe_ids}

    # echo mask for focal cities: emit if new_ids[s] in city subword ids; near within +/-W
    def echo_mask(c, W=3):
        sub = set(probes[c]["subword_ids"])
        emit = [s for s in range(len(new_ids)) if new_ids[s] in sub]
        es = set(emit); near = set()
        for e in emit:
            for k in range(-W, W + 1):
                if 0 <= e + k < len(new_ids): near.add(e + k)
        near -= es
        return {"emit": sorted(es), "near": sorted(near), "n_emit": len(es)}
    echo = {c: echo_mask(c) for c in focal_c}

    surface = [tok.decode([t]) for t in new_ids]
    oh = out_logits.topk(20, -1)
    return {
        "P": P, "n": n, "R_eff": len(new_ids), "frac_argmax_matches_next": round(frac_gate, 4),
        "focal": focal_c, "band": band,
        "surface_tokens": surface, "new_ids": new_ids,
        "rank_head": {c: v.tolist() for c, v in rank_head.items()},
        "per_probe_band_agg": per_probe,   # focal + all cities, min/median/p75 over band per lens
        "echo": echo,
        "jlens_cloud_top20": {str(l): jl_cloud[l] for l in jl_cloud},
        "out_head_topk20_ids": oh.indices.cpu().tolist(),   # output-head top-20 ids per position (surface ref)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big-model", default="Qwen/Qwen3.5-397B-A17B:model.language_model")
    ap.add_argument("--lens-hf", default="praxagent/jacobian-lens-qwen3.5-397b-a17b:jlens/wikitext/qwen35_397b.pt")
    ap.add_argument("--lens-file", default=None, help="local lens .pt (CPU smoke; overrides --lens-hf)")
    ap.add_argument("--expected-sha256", default="668c3bf17305b0d52495cb7ba589a1c1173301b1d13c3c6ad84e58245dc99e97")
    ap.add_argument("--traces", default="recover_thinkon_answers_v2.json")
    ap.add_argument("--prompts-file", default="prompts_wc_thinkon.json")
    ap.add_argument("--only", default=None, help="comma ids e.g. div_6__thinkon,div_9__thinkon")
    ap.add_argument("--out", default="peek_thinking_receipt.json")
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
                                 max_memory={i: "120GiB" for i in range(n_gpu)}).eval()
    tok = transformers.AutoTokenizer.from_pretrained(hf_id)
    model = (jlens.from_hf(hf, tok, compile=False, layout=Layout(path=backbone))
             if backbone else jlens.from_hf(hf, tok, compile=False))

    if args.lens_file:
        lens_path = args.lens_file
    else:
        from huggingface_hub import hf_hub_download
        repo, fname = args.lens_hf.split(":", 1)
        lens_path = hf_hub_download(repo, fname)
    h = hashlib.sha256(open(lens_path, "rb").read()).hexdigest()
    if args.expected_sha256 and h != args.expected_sha256.lower():
        raise SystemExit(f"LENS HASH MISMATCH {h} != {args.expected_sha256}")
    lens = jlens.JacobianLens.load(lens_path)
    band = band_layers(lens.source_layers); final = model.n_layers - 1
    print(f"band={band} (assert 19..38), final={final}, lens sha={h[:12]}", flush=True)

    probes = resolve_probes(tok)
    traces = {v["id"]: v for v in json.load(open(args.traces))["results"]}
    spec = {c["id"]: c for c in json.load(open(args.prompts_file))["conditions"]}
    eos = tok.eos_token_id
    # default: all pressure items (skip _ctrl unless explicitly named in --only)
    only = set(args.only.split(",")) if args.only else {c for c in traces if "_ctrl" not in c}

    out = {"model": hf_id, "lens_sha256": h, "band": band,
           "note": "J-lens read at generated reasoning positions; Δ=log2(rank_head)-log2(rank_jlens_med) at echo=none is the headline",
           "items": {}}
    for cid in traces:
        if cid not in only: continue
        it = traces[cid]
        prompt_ids = tok(spec[cid]["prompt"], add_special_tokens=True)["input_ids"]
        if isinstance(prompt_ids[0], list): prompt_ids = prompt_ids[0]
        new_ids = trim_trailing_pad(list(it["new_ids"]), eos)
        focal = [spec[cid].get("true_answer"), spec[cid].get("tempting_answer")]
        print(f"== {cid}: P={len(prompt_ids)} R_eff={len(new_ids)} focal={focal} ==", flush=True)
        res = peek_one(model, hf, tok, lens, band, final, probes, prompt_ids, new_ids, focal)
        res["true_answer"] = spec[cid].get("true_answer"); res["tempting_answer"] = spec[cid].get("tempting_answer")
        res["committed"] = it.get("committed"); res["think_reached"] = it.get("think_reached")
        print(f"   gate frac_argmax_matches_next={res['frac_argmax_matches_next']} (want ~1.0)", flush=True)
        out["items"][cid] = res
        json.dump(out, open(args.out, "w"))   # incremental save
    print(f"wrote {args.out}  ({len(out['items'])} items)", flush=True)


if __name__ == "__main__":
    main()
