"""Behavioral test #1 — the verbal-report CAUSAL SWAP (geometry -> function).

Replicates Anthropic's verbal-report experiment (data/experiments/verbal-report.json
+ README) on open models, to test whether the WORKSPACE BAND is functional, not just
geometric. The killer comparison: qwen3-4b (band) vs gemma-2-9b (KD, ~no band) vs
gemma-2-27b (from-scratch sibling, band) — same Gemma-2 architecture across the last two.

Two measures per model:
  (A) REPORTABILITY (observation): prompt "Think of a {category}. Answer in one word:";
      take the model's greedy answer at the final ':'. Does that answer token appear in
      the J-lens readout across the workspace band? (rank / hit-rate)
  (B) CAUSAL SWAP (intervention — the headline): for each other candidate c in the
      category, add the J-lens STEERING vector (dir_c - dir_answer) at every position of
      the BAND layers, re-run, and check whether the model's output at ':' flips to c.
      Success = swapped-in candidate is argmax. This is the causal-workspace test:
      if swaps work where the band exists (qwen, gemma-2-27b) and fail where it doesn't
      (gemma-2-9b), the KD finding is FUNCTIONAL. If swaps work in gemma-2-9b anyway,
      our band statistic misses a real workspace (also reportable, changes the story).

Steering direction (Anthropic README): for token t at layer l, the residual-space
direction driving t is row t of (U @ J_l); we add
   strength * mean_residual_norm(l) * (unit(dir_c) - unit(dir_ans))
at all positions of each band layer. `mean_residual_norm` is estimated from the prompt's
own activations at that layer.

Run (GPU for gemma/qwen; CPU-smoke on gpt2):
    uv run python verbal_report.py --slug gpt2-small --strengths 0 4 8 --max-cats 3   # smoke
    uv run python verbal_report.py --slug qwen3-4b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "jacobian_lens"))
from cka_layers import REPO, resolve  # noqa: E402  (slug -> hf_id, lens_file)

# Anthropic's released prompt set (vendored path; falls back to a URL fetch note).
VERBAL_REPORT_JSON = HERE / "verbal-report.json"


def band_layers(source_layers: list[int]) -> list[int]:
    """Middle third of the fitted layers — the 'workspace band' we steer in."""
    n = len(source_layers)
    lo, hi = n // 3, 2 * n // 3
    return source_layers[lo:hi]


def single_token_id(tok, word: str) -> int | None:
    """Return the id if `word` (space-prefixed preferred) is exactly one token."""
    for form in (" " + word, word):
        enc = tok.encode(form, add_special_tokens=False)
        if len(enc) == 1:
            return enc[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--strengths", type=float, nargs="*", default=[0.0, 4.0, 8.0, 12.0])
    ap.add_argument("--max-cats", type=int, default=14)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import jlens
    import transformers
    from huggingface_hub import hf_hub_download

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    hf_id, lens_file = resolve(args.slug)
    print(f"slug={args.slug} hf={hf_id} device={dev}")

    from _loader import load_hf_model  # handles mxfp4 dequant (gpt-oss) + torch shim
    hf = load_hf_model(hf_id, dev)
    tok = transformers.AutoTokenizer.from_pretrained(hf_id)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(hf_hub_download(REPO, filename=lens_file))
    U = model.unembed_matrix() if hasattr(model, "unembed_matrix") else \
        hf.get_output_embeddings().weight.detach()
    U = U.to(dev).float()

    band = band_layers(lens.source_layers)
    print(f"band layers (middle third): {band}")

    cats = json.load(open(VERBAL_REPORT_JSON))["candidates"]
    results = {"slug": args.slug, "hf_id": hf_id, "band": band,
               "strengths": args.strengths, "per_category": [], }

    layer_mods = model.layers  # nn.ModuleList of residual blocks

    def steering_hook(vec):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + vec.to(h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return hook

    swap_success = {s: 0 for s in args.strengths}
    swap_total = 0
    report_hits, report_total = 0, 0

    for cat, words in list(cats.items())[: args.max_cats]:
        prompt = f"Think of a {cat}. Answer in one word:"
        ids = model.encode(prompt).to(dev)
        with torch.no_grad():
            base_logits = hf(ids).logits[0, -1]
        ans_id = int(base_logits.argmax())
        ans_word = tok.decode([ans_id]).strip()

        # (A) reportability: is ans_id in the J-lens top-k across the band?
        lens_logits, _, _ = lens.apply(model, prompt, layers=band, positions=[-1])
        in_topk = any(ans_id in set(lens_logits[l][0].topk(20).indices.tolist()) for l in band)
        report_hits += int(in_topk); report_total += 1

        # candidate steering directions: row of (U @ J_l) for each single-token word
        cand_ids = {}
        for w in words:
            tid = single_token_id(tok, w)
            if tid is not None and tid != ans_id:
                cand_ids[w] = tid
        if ans_id not in [single_token_id(tok, w) for w in words] and not cand_ids:
            continue

        # per-band-layer mean residual norm (from this prompt's activations)
        from jlens.hooks import ActivationRecorder
        with torch.no_grad(), ActivationRecorder(model.layers, at=band) as rec:
            model.forward(ids)
        acts = rec.activations  # {layer: [1, seq, d]}
        mean_norm = {l: acts[l][0].norm(dim=-1).mean().item() for l in band}

        for w, cid in list(cand_ids.items())[:10]:
            swap_total += 1
            # dir for a token t at layer l = row t of (U @ J_l)  (d_model,)
            for s in args.strengths:
                handles = []
                for l in band:
                    Jl = lens.jacobians[l].to(dev).float()  # (d, d)
                    d_ans = (U[ans_id] @ Jl)
                    d_cand = (U[cid] @ Jl)
                    d_ans = d_ans / (d_ans.norm() + 1e-8)
                    d_cand = d_cand / (d_cand.norm() + 1e-8)
                    vec = s * mean_norm[l] * (d_cand - d_ans)
                    handles.append(layer_mods[l].register_forward_hook(steering_hook(vec)))
                with torch.no_grad():
                    swapped = hf(ids).logits[0, -1]
                for h in handles:
                    h.remove()
                if int(swapped.argmax()) == cid:
                    swap_success[s] += 1

        results["per_category"].append(
            {"category": cat, "answer": ans_word, "reported_in_band": in_topk,
             "n_candidates": len(cand_ids)})

    results["reportability_hit_rate"] = report_hits / max(1, report_total)
    results["swap_success_rate"] = {str(s): swap_success[s] / max(1, swap_total)
                                    for s in args.strengths}
    results["swap_total"] = swap_total
    out = args.out or str(HERE / f"verbal_report_{args.slug}.json")
    json.dump(results, open(out, "w"), indent=1)
    print(f"\nreportability hit-rate (answer in band top-20): {results['reportability_hit_rate']:.2f}")
    print("causal swap success rate by strength:")
    for s in args.strengths:
        print(f"  strength {s:5.1f}: {swap_success[s]}/{swap_total} = {swap_success[s]/max(1,swap_total):.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
