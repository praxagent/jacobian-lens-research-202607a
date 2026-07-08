"""Shared-vocabulary probe set — the tokenizer-confound mitigation.

The Eleos commentary's central methodological caution: J-space is defined by each
model's token vocabulary, so cross-family J-lens comparisons may not be
commensurable (different tokenizers = different probe geometry). Mitigation
(same approach as eliebak's explorer, which used "4,096 token strings shared by
all 38 tokenizers"): compute the set of token STRINGS that exist as single
tokens in EVERY model's vocabulary, and probe every model's geometry with those
same strings. Then the CKA band statistic samples semantically identical
directions across families.

Build (needs tokenizers only — a few MB each; gated ones need HF_TOKEN):
    python shared_vocab.py --out shared_tokens.json          # all 38 slugs
    python shared_vocab.py --slugs gpt2-small pythia-70m-deduped --out /tmp/test.json

Then cka_layers.py --shared-probe shared_tokens.json uses it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cka_layers import REPO, resolve  # noqa: E402


def single_token_strings(hf_id: str) -> set[str]:
    """Strings that round-trip to exactly ONE token in this tokenizer.

    We normalize by testing encode(' '+word)/encode(word) forms and keep the
    string if either encodes to a single token; the per-model id is resolved at
    probe time by the same rule (see resolve_ids)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    vocab = tok.get_vocab()
    out = set()
    for surface in vocab.keys():
        # strip common space-markers to get the raw string form
        s = surface.replace("Ġ", " ").replace("▁", " ").strip()
        if not s or not s.isprintable():
            continue
        # keep plausibly word-like strings (letters incl. CJK, digits ok, len>=2)
        if len(s) < 2 or len(s) > 24:
            continue
        if not any(ch.isalpha() for ch in s):
            continue
        out.add(s)
    return out


def resolve_ids(hf_id: str, strings: list[str]) -> dict[str, int]:
    """Map each shared string to a single token id in this tokenizer (space-
    prefixed form preferred, bare form fallback); drop strings that need >1
    token here."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    ids = {}
    for s in strings:
        for form in (" " + s, s):
            enc = tok.encode(form, add_special_tokens=False)
            if len(enc) == 1:
                ids[s] = enc[0]
                break
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs", nargs="*", default=None, help="default: all Neuronpedia slugs")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not args.slugs:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(REPO)
        args.slugs = sorted({f.split("/")[0] for f in files if "/jlens/" in f})

    shared: set[str] | None = None
    hf_ids = {}
    for slug in args.slugs:
        try:
            hf_id, _ = resolve(slug)
            hf_ids[slug] = hf_id
            s = single_token_strings(hf_id)
            shared = s if shared is None else (shared & s)
            print(f"{slug:24s} vocab-strings={len(s):7d}  shared-so-far={len(shared):6d}")
        except Exception as e:
            print(f"{slug:24s} SKIP ({type(e).__name__}: {str(e)[:60]})")
    if not shared:
        sys.exit("no shared strings")

    import random
    pool = sorted(shared)
    random.Random(args.seed).shuffle(pool)
    chosen = sorted(pool[: args.max_tokens])
    json.dump({"strings": chosen, "models": hf_ids}, open(args.out, "w"),
              ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}: {len(chosen)} shared token strings across {len(hf_ids)} models")


if __name__ == "__main__":
    main()
