"""Cache residual-stream activations over gold-labeled entity spans.

For each aligned Example, run the model once (output_hidden_states) and, for each entity
span, keep ONLY the residual-stream slice at that span's tokens for the frozen layers —
not the whole sequence. That is what the four readers consume, and it keeps the receipt
small enough to re-analyze without the GPU (GPU playbook §7).

Returns per-span records plus the model-level unembed / final-norm the lens readers need.
"""
from __future__ import annotations

import torch


def model_readout_parts(model):
    """Return (unembed[V,d], final_norm callable, n_hidden_states). Works across the
    Llama/Gemma/GPT2 families via the standard lm_head + final norm."""
    unembed = model.get_output_embeddings().weight.detach()
    m = model.model if hasattr(model, "model") else getattr(model, "transformer", model)
    final_norm = getattr(m, "norm", None) or getattr(m, "ln_f", None)
    return unembed, final_norm


@torch.no_grad()
def cache_spans(model, tokenizer, examples, layers, device="cpu", dtype=torch.float16,
                max_len=4096, scoring="pre_token"):
    """List of per-span dicts: {cid, label, tok_ids[S], score_pos[S], hid: {layer:[S,d]}}.

    Reader-timing is frozen here (Pro-review B03): to score entity token x_t WITHOUT
    leaking its identity through the residual stream, the state used is the IMMEDIATELY
    PRECEDING causal residual h_{t-1} (which is exactly what predicts x_t in a causal LM).
    So `hid` holds the residual at positions [tok_start-1 .. tok_end-1) while `tok_ids`
    are the entity tokens [tok_start .. tok_end) being predicted. Every reader consumes
    the SAME `hid` states -> comparable, no leakage. `scoring="post_token"` (leaky) is
    available only for an explicit leakage-control experiment; default is pre_token.

    Note: these are TEACHER-FORCED activations of the pinned model over archived text, not
    necessarily the emitting model's original states -- an observational readout, no causal
    claim (B03). Entities whose first token is at position 0 are skipped (no preceding state).
    """
    assert scoring in ("pre_token", "post_token")
    shift = 1 if scoring == "pre_token" else 0
    model.eval()
    records = []
    for ex in examples:
        enc = tokenizer(ex.completion, return_offsets_mapping=False,
                        add_special_tokens=False, truncation=True, max_length=max_len,
                        return_tensors="pt")
        ids = enc["input_ids"].to(device)
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple[L+1] of [1, S, d]
        seqlen = ids.shape[1]
        for sp in ex.spans:
            if sp.tok_start < shift or sp.tok_end > seqlen or sp.tok_start >= sp.tok_end:
                continue  # need a preceding state for pre_token scoring
            state = slice(sp.tok_start - shift, sp.tok_end - shift)   # h_{t-1..} states
            tok = slice(sp.tok_start, sp.tok_end)                     # x_t entity tokens
            hid = {L: hs[L][0, state].detach().to("cpu", dtype) for L in layers}
            records.append({
                "cid": ex.cid, "label": int(sp.label),
                "tok_ids": ids[0, tok].detach().to("cpu"),
                "score_pos": list(range(sp.tok_start - shift, sp.tok_end - shift)),
                "hid": hid,
            })
    return records


def span_hidden(record, layer):
    """[S, d] float32 for a reader call (readers expect one layer's span slice)."""
    return record["hid"][layer].float()
