"""Ignition-depth runner. Design frozen in PREREG.md.

For each prompt, apply the model's OWN output head (final norm + unembedding) to every layer's
residual stream and find the shallowest layer at which the correct answer is rank 1 and stays
rank 1 all the way down. That is a logit-lens readout, deliberately independent of the Jacobian.

Saves the full per-layer rank trajectory and the top-k at every layer, so a different definition
of ignition can be evaluated from the receipt without re-renting the GPU.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TOPK = 10


NORM_NAMES = ("norm", "ln_f", "final_layer_norm", "final_layernorm", "final_norm")


def output_head(model):
    """Return (f, description) where f(h) -> logits via the model's real final norm + unembedding.

    Discovered generically rather than by architecture name: `model.base_model` is the HF-wide
    accessor, and the final norm goes by at least five different attribute names across the
    families we use (GPT-NeoX alone uses none of the first two).
    """
    base = model.base_model
    norm, norm_name = None, None
    for n in NORM_NAMES:
        if hasattr(base, n) and getattr(base, n) is not None:
            norm, norm_name = getattr(base, n), n
            break
    head = model.get_output_embeddings()
    if head is None:
        raise SystemExit("model exposes no output embedding; cannot build a logit-lens head")
    def f(h):
        return head(norm(h)) if norm is not None else head(h)
    return f, {"base": type(base).__name__, "final_norm_attr": norm_name,
               "head": type(head).__name__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--prompts", default=str(HERE / "prompts_frozen.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float32).to(a.device).eval()
    head, head_desc = output_head(model)

    art = json.load(open(a.prompts))
    rows, dropped, final_ok = [], [], 0
    for item in art["prompts"]:
        ids_ans = tok.encode(item["answer"], add_special_tokens=False)
        if len(ids_ans) != 1:
            dropped.append({"prompt": item["prompt"], "answer": item["answer"],
                            "reason": f"answer is {len(ids_ans)} tokens in this tokenizer"})
            continue
        t_star = ids_ans[0]
        ids = tok(item["prompt"], return_tensors="pt").input_ids.to(a.device)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        if hs is None:
            raise SystemExit("hidden_states is None for this architecture")
        ranks, tops = [], []
        for k, h in enumerate(hs[1:]):                       # skip the embedding layer
            with torch.no_grad():
                lg = head(h[:, -1, :]).float()[0]
            order = torch.argsort(lg, descending=True)
            ranks.append(int((order == t_star).nonzero()[0, 0]) + 1)
            v, i = torch.topk(lg, TOPK)
            tops.append({"ids": i.tolist(), "scores": [round(x, 4) for x in v.tolist()]})
        L = len(ranks)
        # ignition = shallowest layer that is rank 1 and stays rank 1 to the end
        ign = None
        for l in range(L):
            if all(r == 1 for r in ranks[l:]):
                ign = l
                break
        if ranks[-1] == 1:
            final_ok += 1
        rows.append({"prompt": item["prompt"], "answer": item["answer"],
                     "answer_id": int(t_star), "n_layers": L, "ranks": ranks,
                     "ignition_layer": ign,
                     "ignition_reldepth": None if ign is None else ign / max(L - 1, 1),
                     "topk_per_layer": tops})

    ig = [r["ignition_reldepth"] for r in rows if r["ignition_reldepth"] is not None]
    res = {"slug": a.slug, "model": a.model, "output_head": head_desc,
           "n_prompts_used": len(rows), "n_dropped": len(dropped), "dropped": dropped,
           "n_ignited": len(ig), "n_layers": rows[0]["n_layers"] if rows else None,
           "median_ignition_reldepth": float(np.median(ig)) if ig else None,
           "final_layer_correct": final_ok,
           "gate_final_head": final_ok >= 8,
           "usable": len(ig) >= 5,
           "rows": rows,
           "transformers": transformers.__version__, "torch": torch.__version__}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"IGNITION {a.slug}: L={res['n_layers']} ignited={len(ig)}/{len(rows)} "
          f"median_reldepth={res['median_ignition_reldepth']} "
          f"final_head_ok={final_ok}/{len(rows)} usable={res['usable']}", flush=True)


if __name__ == "__main__":
    main()
