"""Fit our OWN Jacobian lens on an open model (validation + seed-stability).

Uses Anthropic's `jlens.fit` (Apache-2.0). Fitting is backward-pass-bound:
~ceil(d_model/dim_batch) backward passes per prompt, over N short prompts, as a
running mean. Needs the full model on-device with autograd — i.e. GPU for
anything past gpt2-size. Checkpointed + resumable.

Two tier-1 uses (see README):
  - VALIDATION: fit gpt2, compare to Neuronpedia's gpt2 lens (compare.py). If our
    lens matches theirs, our pipeline is faithful and bigger fits are trustworthy.
  - STABILITY: fit the same model with several --seed values (different corpus
    subsets); compare.py across them tests whether J-space is stable or
    fitting-dependent.

Run (GPU):
    uv run python fit_lens.py --model openai-community/gpt2 --n-prompts 100 --out lenses/gpt2_seed0.pt --seed 0
    uv run python fit_lens.py --model Qwen/Qwen3-4B --n-prompts 100 --out lenses/qwen4b_seed1.pt --seed 1
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def load_corpus(n_prompts: int, seed: int, corpus: str = "wikitext",
                min_chars: int = 200) -> list[str]:
    """N text passages for lens fitting.

    corpus:
      - "wikitext"      — English wikitext-103 (the paper's pretraining-like corpus)
      - "wikipedia-zh"  — Chinese Wikipedia (streaming; language-dependence test)
      - "wikipedia-fr"  — French Wikipedia (streaming)

    The seed shuffles which passages are used: different seeds = different
    corpus subsets (stability test); different corpora = different estimation
    distributions/languages (language-dependence test).
    """
    from datasets import load_dataset
    if corpus == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t) >= min_chars]
        rng = random.Random(seed)
        rng.shuffle(texts)
        return texts[:n_prompts]
    if corpus == "code":
        # Tier-1 corpus arm: code passages for the code-vs-prose lens A/B. min_chars is
        # raised so every passage fills the fit's 128-token window, matching the wikitext
        # prompts' effective length AT THE FIT CAP (both truncate to max_seq_len), which
        # is how we neutralize the prompt-length confound the community read left open.
        ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train",
                          streaming=True)   # ungated Python code, `content` field
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        texts = []
        for row in ds:
            t = row.get("content") or ""
            if len(t) >= max(min_chars, 600):   # ~>=128 code tokens at ~4-5 chars/token
                texts.append(t)
            if len(texts) >= n_prompts:
                break
        return texts
    if corpus.startswith("wikipedia-"):
        lang = corpus.split("-", 1)[1]
        # Streaming + shuffle-buffer: avoids downloading a multi-GB dump for 100 texts.
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}",
                          split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        texts = []
        for row in ds:
            t = row.get("text") or ""
            if len(t) >= min_chars:
                texts.append(t)
            if len(texts) >= n_prompts:
                break
        return texts
    raise ValueError(f"unknown corpus {corpus!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--dim-batch", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", default="wikitext",
                    help="wikitext | code | wikipedia-zh | wikipedia-fr")
    ap.add_argument("--min-chars", type=int, default=200,
                    help="min passage length; length-matching for the code-vs-prose A/B")
    ap.add_argument("--match-length", action="store_true",
                    help="keep only passages that fill max_seq_len tokens (code-vs-prose A/B)")
    ap.add_argument("--out", required=True, help="output lens .pt path")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import jlens
    import torch
    import transformers

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"fitting lens: model={args.model} n_prompts={args.n_prompts} "
          f"dim_batch={args.dim_batch} seed={args.seed} corpus={args.corpus} device={dev}")

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
    ).to(dev).eval()
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf, tok)

    # --match-length: keep only passages with >= max_seq_len tokens, so EVERY fitted
    # prompt truncates to exactly the cap. This is what makes the code-vs-prose A/B a
    # clean test: both corpora contribute identical-length windows, so a geometry
    # difference cannot be prompt length (the confound the community read left open).
    pool = args.n_prompts * (6 if args.match_length else 1)
    cands = load_corpus(pool, args.seed, corpus=args.corpus, min_chars=args.min_chars)
    if args.match_length:
        prompts = [p for p in cands if len(tok(p).input_ids) >= args.max_seq_len][:args.n_prompts]
        if len(prompts) < args.n_prompts:
            raise SystemExit(f"only {len(prompts)}/{args.n_prompts} passages reach "
                             f"{args.max_seq_len} tokens; raise pool or lower cap")
    else:
        prompts = cands[:args.n_prompts]
    import statistics as _st
    lens_tok = [min(len(tok(p).input_ids), args.max_seq_len) for p in prompts]
    print(f"corpus: {len(prompts)} passages | fitted token lengths "
          f"median={_st.median(lens_tok)} mean={_st.mean(lens_tok):.1f} "
          f"min={min(lens_tok)} (cap={args.max_seq_len}, match_length={args.match_length})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lens = jlens.fit(
        model, prompts,
        dim_batch=args.dim_batch, max_seq_len=args.max_seq_len,
        checkpoint_path=str(out.with_suffix(".ckpt")),
    )
    lens.save(str(out))
    print(f"saved lens -> {out}  ({lens})")


if __name__ == "__main__":
    main()
