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
    ap.add_argument("--corpus", default="wikitext", help="wikitext | wikipedia-zh | wikipedia-fr")
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

    prompts = load_corpus(args.n_prompts, args.seed, corpus=args.corpus)
    print(f"corpus: {len(prompts)} passages")

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
