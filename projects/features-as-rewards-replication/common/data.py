"""Load LongFact++ gold-labeled entity spans (obalcells/longfact-annotations).

Each row is a (prompt, assistant completion) plus a list of entity annotations:
  {index: char offset into the completion, span: entity text,
   label: "Supported" | "Not Supported" | (other), verification_note: ...}

We keep only {Supported -> 0, Not Supported -> 1} and map each entity's character
span to TOKEN positions under a given tokenizer (return_offsets_mapping), so the
readers can pool residual-stream activations over the entity's tokens.

The dataset config name selects the model whose completions were annotated:
  Meta-Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct, gemma-2-9b-it, ...
"""
from __future__ import annotations

from dataclasses import dataclass, field

LABEL_MAP = {"Supported": 0, "Not Supported": 1}  # everything else excluded


@dataclass
class Span:
    char_start: int
    char_end: int
    text: str
    label: int          # 1 = hallucinated (Not Supported), 0 = supported
    tok_start: int = -1  # filled by align_spans
    tok_end: int = -1    # exclusive


@dataclass
class Example:
    cid: str            # stable completion id (subset + row index)
    prompt: str
    completion: str
    spans: list = field(default_factory=list)


def _locate(completion: str, ann: dict) -> tuple[int, int] | None:
    """Resolve an annotation to (char_start, char_end). `index` is a char offset; if the
    span text does not sit exactly there (whitespace/offset drift), search nearby, then
    fall back to the first occurrence. Returns None if the span text is not found."""
    span = ann.get("span") or ""
    if not span:
        return None
    idx = ann.get("index")
    if isinstance(idx, int) and completion[idx:idx + len(span)] == span:
        return idx, idx + len(span)
    if isinstance(idx, int):
        lo = max(0, idx - 40)
        near = completion.find(span, lo, idx + len(span) + 40)
        if near != -1:
            return near, near + len(span)
    pos = completion.find(span)
    return (pos, pos + len(span)) if pos != -1 else None


def load_examples(config: str, split: str, limit: int | None = None,
                  hf_token: str | None = None, streaming: bool = True) -> list:
    """Return a list[Example] with char-level spans (token alignment done later)."""
    from datasets import load_dataset
    ds = load_dataset("obalcells/longfact-annotations", config, split=split,
                      streaming=streaming, token=hf_token)
    out = []
    for i, row in enumerate(ds):
        if limit is not None and len(out) >= limit:
            break
        asst = next((t["content"] for t in row["conversation"] if t["role"] == "assistant"), None)
        prompt = next((t["content"] for t in row["conversation"] if t["role"] == "user"), "")
        if not asst:
            continue
        spans = []
        for ann in row.get("annotations", []):
            if ann.get("label") not in LABEL_MAP:
                continue
            loc = _locate(asst, ann)
            if loc is None:
                continue
            cs, ce = loc
            spans.append(Span(cs, ce, asst[cs:ce], LABEL_MAP[ann["label"]]))
        if spans:
            out.append(Example(f"{row.get('subset','?')}:{i}", prompt, asst, spans))
    return out


def align_spans(ex: Example, tokenizer) -> Example:
    """Fill tok_start/tok_end for each span using offset mapping of the completion.
    A span's tokens = every token whose char range overlaps the span's char range."""
    enc = tokenizer(ex.completion, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    for sp in ex.spans:
        toks = [j for j, (a, b) in enumerate(offsets)
                if a < sp.char_end and b > sp.char_start and b > a]
        if toks:
            sp.tok_start, sp.tok_end = toks[0], toks[-1] + 1
    return ex


def label_balance(examples: list) -> dict:
    pos = sum(sp.label for ex in examples for sp in ex.spans)
    n = sum(len(ex.spans) for ex in examples)
    return {"examples": len(examples), "spans": n, "hallucinated": pos,
            "supported": n - pos, "prevalence": (pos / n) if n else 0.0}
