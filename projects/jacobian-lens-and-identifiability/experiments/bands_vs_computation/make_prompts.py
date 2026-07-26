"""Freeze the two prompt sets Test C measures damage on, per PREREG.md.

Prose from WikiText-103, code from codeparrot, matched on token length by construction so the
prose/code contrast is not a length contrast. Written to a JSON artifact so the runner never
touches `datasets` (its pyarrow/numpy ABI mismatch segfaults on pods).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKENIZER = "google/gemma-3-270m"   # only used for length matching; frozen for reproducibility


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48, help="prompts per corpus")
    ap.add_argument("--target-tokens", type=int, default=64)
    ap.add_argument("--tolerance", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "prompts_frozen.json"))
    a = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    def take(stream, field, n):
        out = []
        for row in stream:
            t = row[field]
            if not t or len(t) < 200:
                continue
            ids = tok.encode(t, add_special_tokens=False)
            if len(ids) < a.target_tokens:
                continue
            # truncate to the frozen target so prose and code are length-matched exactly
            out.append(tok.decode(ids[:a.target_tokens]))
            if len(out) >= n:
                break
        return out

    prose = take(load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                              split="train", streaming=True), "text", a.n)
    code = take(load_dataset("codeparrot/codeparrot-clean-valid",
                             split="train", streaming=True), "content", a.n)

    lens = {"prose": [len(tok.encode(p, add_special_tokens=False)) for p in prose],
            "code": [len(tok.encode(p, add_special_tokens=False)) for p in code]}
    art = {"tokenizer": TOKENIZER, "target_tokens": a.target_tokens,
           "n_per_corpus": {"prose": len(prose), "code": len(code)},
           "token_lengths": lens, "prose": prose, "code": code}
    Path(a.out).write_text(json.dumps(art, indent=1))
    print(f"prose {len(prose)} (tok {min(lens['prose'])}-{max(lens['prose'])}), "
          f"code {len(code)} (tok {min(lens['code'])}-{max(lens['code'])}) -> {a.out}")


if __name__ == "__main__":
    main()
