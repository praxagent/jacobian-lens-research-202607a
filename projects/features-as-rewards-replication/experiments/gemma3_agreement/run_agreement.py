"""Gemma-3-12B label-free reader-agreement arm (frozen design: see README.md).

Stages (all in one process on the pod):
  1. generate  : 300 greedy completions on pinned LongFact prompts (from the public
                 dataset's user turns -- prompts are model-independent).
  2. extract   : entity spans via DeepSeek (GuardedLLM, extraction ONLY, no labels).
                 --smoke uses a regex extractor (no paid calls).
  3. readers   : pre-token states at layers [primary, head]; score logit lens, native
                 head surprisal, random-transport null, length+freq heuristic, and
                 Gemma Scope 2 SAE max-activation. NO labels anywhere.
  4. agreement : Spearman correlation matrix + top-decile overlap across readers.

Smoke: .venv/bin/python run_agreement.py --smoke   (gpt2, regex extraction, CPU)
"""
from __future__ import annotations

import argparse, json, os, re, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE.parents[2] / "grounding"))
import data as D      # noqa: E402
import acts as A      # noqa: E402
import readers as R   # noqa: E402
import sae as S       # noqa: E402


def _env(k):
    p = HERE.parents[4] / "shared/runpod/.env"
    for line in p.read_text().splitlines() if p.exists() else []:
        if line.startswith(k + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get(k)


def prompts_from_dataset(n, tok_hf):
    from datasets import load_dataset
    ds = load_dataset("obalcells/longfact-annotations", "Meta-Llama-3.1-8B-Instruct",
                      split="test", streaming=True, token=tok_hf)
    out = []
    for row in ds:
        u = next((t["content"] for t in row["conversation"] if t["role"] == "user"), None)
        if u and row.get("subset") == "longfact_objects":
            out.append(u)
        if len(out) >= n:
            break
    return out


def extract_entities(completion, llm):
    """DeepSeek extraction: verbatim factual entity/claim spans, JSON list of strings."""
    sysm = ("List the specific factual entities (names, dates, places, organizations, "
            "numbers, titles) in the text as VERBATIM substrings. Respond ONLY with a "
            "JSON array of strings, max 25, each copied exactly from the text.")
    txt, _ = llm.chat([{"role": "system", "content": sysm},
                       {"role": "user", "content": completion[:4000]}], max_tokens=600)
    try:
        arr = json.loads(txt[txt.index("["): txt.rindex("]") + 1])
        return [s for s in arr if isinstance(s, str) and 2 < len(s) < 80 and s in completion]
    except Exception:
        return []


def regex_entities(completion):  # smoke only
    return list(dict.fromkeys(re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)?\b|\b1?\d{3,4}\b",
                                         completion)))[:15]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ca = ra - ra.mean(); cb = rb - rb.mean()
    d = float(np.sqrt((ca ** 2).sum() * (cb ** 2).sum()))
    return float((ca * cb).sum() / d) if d else float("nan")


def run(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llm import GuardedLLM
    t0 = time.time()
    tok_hf = _env("HF_TOKEN")
    model_name = "gpt2" if a.smoke else "google/gemma-3-12b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=tok_hf)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=tok_hf,
        torch_dtype=torch.float32 if a.device == "cpu" else torch.bfloat16,
        output_hidden_states=True).to(a.device)
    model.eval()
    n_layers = getattr(model.config, "num_hidden_layers", None) or \
        getattr(model.config, "n_layer", None) or model.config.text_config.num_hidden_layers
    primary = a.primary_layer if a.primary_layer else (6 if a.smoke else 24)
    layers = sorted({primary, n_layers})
    unembed, final_norm = A.model_readout_parts(model)
    ue = unembed.to(a.device).float()

    # --- 1. generate ---
    n = 4 if a.smoke else a.n
    prompts = prompts_from_dataset(n, tok_hf)
    comps = []
    print(f"  [stage] generate {len(prompts)}", flush=True)
    for i, p in enumerate(prompts):
        if a.smoke:
            ids = tokenizer(p[:200], return_tensors="pt").input_ids.to(a.device)
        else:
            msgs = [{"role": "user", "content": p}]
            ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                                return_tensors="pt").to(a.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=128 if a.smoke else 768,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        comps.append(tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
        if i % 25 == 0:
            print(f"    gen {i}/{len(prompts)}", flush=True)

    # --- 2. extract ---
    print("  [stage] extract", flush=True)
    llm = None if a.smoke else GuardedLLM(_env("OPENROUTER_API_KEY"), "deepseek/deepseek-chat",
                                          0.28, 0.88, max_usd=2.0, max_calls=a.n + 10)
    examples = []
    for i, c in enumerate(comps):
        ents = regex_entities(c) if a.smoke else extract_entities(c, llm)
        ex = D.Example(f"g3:{i}", prompts[i], c, [])
        for e in ents:
            pos = c.find(e)
            if pos > 0:
                ex.spans.append(D.Span(pos, pos + len(e), e, 0))  # label unused
        if ex.spans:
            examples.append(D.align_spans(ex, tokenizer))
    recs = A.cache_spans(model, tokenizer, examples, layers, device=a.device, max_len=1024)
    print(f"    {len(examples)} completions, {len(recs)} spans", flush=True)

    # --- 3. readers ---
    print("  [stage] readers", flush=True)
    import gc
    if final_norm is not None:
        _w = final_norm.weight.detach().float().to(a.device)
        _eps = getattr(final_norm, "variance_epsilon", getattr(final_norm, "eps", 1e-6))
        _b = getattr(final_norm, "bias", None)
        _b = _b.detach().float().to(a.device) if _b is not None else None
    model.cpu(); del model; gc.collect()
    if a.device != "cpu":
        torch.cuda.empty_cache()

    def _fn(h):
        h = h.to(a.device)
        if final_norm is None:
            return h
        if _b is None and not isinstance(final_norm, torch.nn.LayerNorm):
            return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + _eps) * _w
        import torch.nn.functional as F
        return F.layer_norm(h, (h.shape[-1],), _w, _b, _eps)

    d = recs[0]["hid"][primary].shape[-1]
    gen = torch.Generator().manual_seed(0)
    randT = torch.randn(d, d, generator=gen); randT *= (torch.eye(d).norm() / randT.norm())
    sae_r = None
    if a.sae_path:
        sae_r = S.load_gemmascope(a.sae_path) if a.sae_format == "gemmascope" else \
            S.load_goodfire(a.sae_path, d)
    from collections import Counter
    freq = Counter(int(t) for r in recs for t in r["tok_ids"].tolist())
    tot = sum(freq.values()) or 1
    scores = {k: [] for k in ("logit_lens", "native_head", "random_null", "heuristic",
                              *(["sae_max_act"] if sae_r else []))}
    import math
    for r in recs:
        hid_p = {primary: A.span_hidden(r, primary).to(a.device)}
        hid_h = {n_layers: A.span_hidden(r, n_layers).to(a.device)}
        tid = r["tok_ids"].to(a.device)
        scores["logit_lens"].append(R.logit_lens_score(hid_p, ue, _fn, slice(None), tid, primary))
        scores["native_head"].append(R.logit_lens_score(hid_h, ue, _fn, slice(None), tid,
                                                        n_layers, head_layer=n_layers))
        scores["random_null"].append(R.jlens_score(hid_p, ue, _fn, slice(None), tid, primary,
                                                   transport=randT.to(a.device).float()))
        ids = r["tok_ids"].tolist()
        scores["heuristic"].append(len(ids) + (-np.mean([math.log((freq.get(i, 0) + 1) / tot)
                                                         for i in ids]) if ids else 0))
        if sae_r:
            scores["sae_max_act"].append(float(sae_r.encode(A.span_hidden(r, primary)).max(1).values.mean()))

    # --- 4. agreement ---
    names = list(scores)
    mat = {x: {y: round(spearman(np.array(scores[x]), np.array(scores[y])), 3)
               for y in names} for x in names}
    k = max(1, len(recs) // 10)
    top = {x: set(np.argsort(-np.array(scores[x]))[:k].tolist()) for x in names}
    ovl = {x: {y: round(len(top[x] & top[y]) / k, 3) for y in names} for x in names}
    receipt = {"what": "Gemma-3-12B label-free reader-agreement (frozen README design); "
                       "SCORE AGREEMENT ONLY -- correlated readers may agree for reasons "
                       "unrelated to hallucination (I05); no detection-validity claim",
               "model": model_name, "primary_layer": primary, "head_layer": n_layers,
               "n_completions": len(examples), "n_spans": len(recs),
               "gen": {"greedy": True, "seed": 0, "max_new_tokens": 768},
               "extractor": "regex (smoke)" if a.smoke else
                            {"model": "deepseek/deepseek-chat", "guard": llm.summary()},
               "spearman": mat, "top_decile_overlap": ovl,
               "per_span": [{"cid": r["cid"], "entity": tokenizer.decode(r["tok_ids"]),
                             **{k2: float(scores[k2][i]) for k2 in names}}
                            for i, r in enumerate(recs)],
               "args": vars(a), "seconds": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(receipt, indent=1, default=float))
    print("  spearman:", {f"{x}~{y}": mat[x][y] for x in names for y in names if x < y})
    print(f"  receipt -> {a.out} ({receipt['seconds']}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--primary-layer", type=int, default=None)
    ap.add_argument("--sae-path", default=None)
    ap.add_argument("--sae-format", default="gemmascope")
    ap.add_argument("--out", default=str(HERE.parent / "receipts/smoke_agreement.json"))
    run(ap.parse_args())
