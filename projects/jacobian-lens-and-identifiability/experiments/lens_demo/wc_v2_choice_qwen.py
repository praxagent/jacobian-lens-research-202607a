"""Batched Qwen-397B choice-family runner (finishes confound_v2 Exp 3 cheaply).

The per-token sequential greedy loop in wc_v2_battery.py hit the device_map throughput
wall (~20 min/condition x 16 = ~5h). This does the same science with:
  - BATCHED generation: all 16 choice conditions in ONE hf.generate() call per mode
    (greedy, do_sample=False) => ~1 forward-pass-set x <=max_new steps, not 16x.
    thinking-off (empty-think prefill, short) and thinking-on (3000-tok window,
    answer parsed after </think>).
  - Full-depth red/blue readout per condition (16 single forwards; cheap): rank of
    both colors at EVERY layer at the final prompt position, all 4 transports.
Only the mapping-averaged self-vs-human color contrast is interpreted (echo-symmetric).
"""
import argparse, json, sys, re
from pathlib import Path
import torch, transformers, jlens
from jlens import Layout
from jlens.hooks import ActivationRecorder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from demo2 import single_token_id, band_layers
from wc_v2_battery import transports_at, capture, committed_color, after_think


@torch.no_grad()
def batched_generate(hf, tok, prompts, max_new):
    """Greedy, left-padded, batched. Returns list of decoded continuations (new tokens only)."""
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    dev = hf.get_input_embeddings().weight.device
    enc = {k: v.to(dev) for k, v in enc.items()}
    out = hf.generate(**enc, max_new_tokens=max_new, do_sample=False,
                      pad_token_id=tok.pad_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(row, skip_special_tokens=True) for row in new]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big-model", required=True)
    ap.add_argument("--lens-hf", required=True)
    ap.add_argument("--think-tokens", type=int, default=3000)
    ap.add_argument("--off-tokens", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--prompts", default=str(HERE / "prompts_wc_v2.json"))
    ap.add_argument("--out", default=str(HERE / "demo2_wc_v2_qwen35-397b_choice.json"))
    args = ap.parse_args()

    hf_id, _, backbone = args.big_model.partition(":")
    cfg = transformers.AutoConfig.from_pretrained(hf_id)
    cls = getattr(transformers, cfg.architectures[0])
    ng = torch.cuda.device_count()
    hf = cls.from_pretrained(hf_id, torch_dtype=torch.bfloat16, device_map="auto",
                             attn_implementation="eager",
                             max_memory={i: "125GiB" for i in range(ng)}).eval()
    tok = transformers.AutoTokenizer.from_pretrained(hf_id)
    model = (jlens.from_hf(hf, tok, compile=False, layout=Layout(path=backbone))
             if backbone else jlens.from_hf(hf, tok, compile=False))
    repo, fname = args.lens_hf.split(":", 1)
    from huggingface_hub import hf_hub_download
    lens = jlens.JacobianLens.load(hf_hub_download(repo, fname))
    alll = list(lens.source_layers); final = model.n_layers - 1
    colors = {w: single_token_id(tok, w) for w in ("red", "blue")}
    print(f"loaded; layers={len(alll)}, colors={colors}", flush=True)

    spec = json.load(open(args.prompts))
    ch = [c for c in spec["conditions"] if c["family"] == "choice"]
    prompts = [c["prompt"] for c in ch]

    def templ(thinking):
        return [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=thinking)
                for p in prompts]

    print("--- batched thinking-OFF generate ---", flush=True)
    off = []
    for i in range(0, len(prompts), args.batch_size):
        off += batched_generate(hf, tok, templ(False)[i:i+args.batch_size], args.off_tokens)
    print("--- batched thinking-ON generate (this is the one that was slow sequentially) ---", flush=True)
    on = []
    for i in range(0, len(prompts), args.batch_size):
        on += batched_generate(hf, tok, templ(True)[i:i+args.batch_size], args.think_tokens)

    print("--- full-depth color readout (16 single forwards) ---", flush=True)
    items = []
    for j, c in enumerate(ch):
        pids = tok(c["prompt"], add_special_tokens=True)["input_ids"]
        if isinstance(pids[0], list):
            pids = pids[0]
        acts = capture(model, pids, sorted(set(alll) | {final}))
        hlast = {l: acts[l][:, -1:, :] for l in alll}
        best, by_layer = transports_at(model, lens, alll, hlast, colors)
        del acts, hlast
        on_ans = after_think(on[j])
        items.append({
            "id": c["id"], "self_color": c["self_color"], "human_color": c["human_color"],
            "committed_color_thinkoff": committed_color(off[j]),
            "committed_color_thinkon": committed_color(on_ans),
            "thinkon_reached_close": "</think>" in on[j],
            "thinkoff_text": off[j][:200], "thinkon_text": on[j][:600],
            "colors": {"full_depth_rank_by_layer": by_layer, "best_rank": best},
        })
        print(f"  {c['id']:14s} off={items[-1]['committed_color_thinkoff']:16s} "
              f"on={items[-1]['committed_color_thinkon']:16s} </think>={items[-1]['thinkon_reached_close']}", flush=True)
        json.dump({"model": hf_id, "spec": "wc_v2_choice", "items": items}, open(args.out, "w"))
    print(f"wrote {args.out} ({len(items)} items)", flush=True)


if __name__ == "__main__":
    main()
