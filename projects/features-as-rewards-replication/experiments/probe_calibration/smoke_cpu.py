"""CPU smoke (rung 0-1): prove the whole reader pipeline on cached gpt2 + synthetic
signal, before any GPU. A correctness check, not just no-crash:

  A. metric unit tests (auroc on known inputs)
  B. real gpt2 forward -> all 4 readers produce finite scalars over an entity span
     (exercises hidden-state extraction, unembed, J-transport matmul, SAE encode,
     attention-probe forward -- the exact ops the real runs use)
  C. synthetic-signal discrimination: spans where 'hallucinated' carry an injected
     residual direction. Trained probe must separate them on a HELD-OUT split
     (AUROC high); an aligned SAE latent reader must beat 0.5; a random reader ~0.5.

Run: .venv/bin/python experiments/probe_calibration/smoke_cpu.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import readers as R  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


# --------------------------------------------------------------------------- #
print("A. metric unit tests")
check("auroc perfect separation == 1.0", R.auroc([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0)
check("auroc inverted == 0.0", R.auroc([1, 1, 0, 0], [0, 0, 1, 1]) == 0.0)
check("auroc all-ties == 0.5", abs(R.auroc([1, 1, 1, 1], [0, 1, 0, 1]) - 0.5) < 1e-9)
check("directed_auroc(inverted) folds to 1.0",
      R.directed_auroc([1, 1, 0, 0], [0, 0, 1, 1]) == 1.0)

# --------------------------------------------------------------------------- #
print("\nB. real gpt2 forward -> four readers over an entity span")
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", output_hidden_states=True)
model.eval()
d_model = model.config.n_embd
unembed = model.lm_head.weight.detach()               # [V, d]
final_norm = model.transformer.ln_f                   # callable [.,d]->[.,d]
LAYER = 6

text = "The capital of Australia is Canberra, a planned city."
enc = tok(text, return_tensors="pt")
ids = enc["input_ids"][0]
with torch.no_grad():
    out = model(**enc)
hs = [h[0].detach() for h in out.hidden_states]        # list[L+1] of [S, d]
# entity span = the tokens of "Canberra"
span_toks = tok(" Canberra", add_special_tokens=False)["input_ids"]
S = ids.tolist()
start = next(i for i in range(len(S) - len(span_toks) + 1)
             if S[i:i + len(span_toks)] == span_toks)
span = slice(start, start + len(span_toks))
span_ids = ids[span]

J = torch.eye(d_model)                                 # smoke transport = identity
rng = torch.Generator().manual_seed(1)
sae_W = torch.randn(4096, d_model, generator=rng) * 0.02   # random SAE encoder (smoke)
sae_encode = lambda h: torch.relu(h @ sae_W.T)         # [S,d]->[S,L]

s_logit = R.logit_lens_score(hs, unembed, final_norm, span, span_ids, LAYER)
s_head = R.logit_lens_score(hs, unembed, final_norm, span, span_ids, len(hs) - 1)
s_jlens = R.jlens_score(hs, unembed, final_norm, span, span_ids, LAYER, transport=J)
s_sae = R.sae_latent_score(hs, sae_encode, latent_idx=17, span_slice=span, layer=LAYER)
probe = R.AttentionProbe(d_model, n_heads=4)
with torch.no_grad():
    s_probe = torch.sigmoid(probe(hs[LAYER][span].unsqueeze(0))).item()

for nm, v in [("logit_lens", s_logit), ("output_head", s_head), ("jlens(=id)", s_jlens),
              ("sae_latent", s_sae), ("attn_probe", s_probe)]:
    check(f"{nm} finite scalar", np.isfinite(v), f"= {v:.4f}")
check("jlens with identity transport == logit lens at same layer",
      abs(s_jlens - s_logit) < 1e-3, f"{s_jlens:.4f} vs {s_logit:.4f}")

# --------------------------------------------------------------------------- #
print("\nC. synthetic-signal discrimination (train probe, held-out AUROC)")
N, Sp, d = 240, 5, d_model
torch.manual_seed(2)
labels = (torch.arange(N) % 2).numpy()                 # balanced
direction = torch.randn(d)
direction = direction / direction.norm()
spans = torch.randn(N, Sp, d) * 0.5
for i in range(N):                                     # inject the hallucination dir
    if labels[i] == 1:
        spans[i] += 1.2 * direction
masks = torch.ones(N, Sp)
tr = np.arange(0, 160)                                  # train / held-out split
va = np.arange(160, N)
probe, _ = R.train_probe(spans[tr], masks[tr], labels[tr], d, n_heads=4,
                         epochs=250, lr=1e-2, seed=0)
with torch.no_grad():
    val_scores = torch.sigmoid(probe(spans[va], masks[va])).numpy()
probe_auroc = R.auroc(val_scores, labels[va])
check("trained attention probe separates on held-out (AUROC > 0.85)",
      probe_auroc > 0.85, f"AUROC = {probe_auroc:.3f}")

# an SAE-style reader aligned to the injected direction should beat chance;
# a random direction should not
aligned = (spans[va].mean(1) @ direction).numpy()
random_dir = torch.randn(d); random_dir /= random_dir.norm()
randscore = (spans[va].mean(1) @ random_dir).numpy()
check("aligned reader beats chance (directed AUROC > 0.7)",
      R.directed_auroc(aligned, labels[va]) > 0.7,
      f"= {R.directed_auroc(aligned, labels[va]):.3f}")
check("random reader is a null (directed AUROC < 0.65)",
      R.directed_auroc(randscore, labels[va]) < 0.65,
      f"= {R.directed_auroc(randscore, labels[va]):.3f}")

# --------------------------------------------------------------------------- #
print("\nD. reader timing (B03): score each token from the PRECEDING causal residual")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import data as Dmod   # noqa: E402
import acts as Amod   # noqa: E402

# one example: the entity 'Canberra' inside a sentence
comp = "The capital of Australia is Canberra, a planned city in the ACT."
cstart = comp.index("Canberra")
ex = Dmod.Example("smoke:0", "q", comp,
                  [Dmod.Span(cstart, cstart + len("Canberra"), "Canberra", 1)])
ex = Dmod.align_spans(ex, tok)
head = len(model.transformer.h)  # output-head hidden-state index for gpt2
pre = Amod.cache_spans(model, tok, [ex], [head], scoring="pre_token")
post = Amod.cache_spans(model, tok, [ex], [head], scoring="post_token")


def mean_surprisal(recs):
    r = recs[0]
    return R.logit_lens_score({head: Amod.span_hidden(r, head)}, unembed, final_norm,
                              slice(None), r["tok_ids"], head)


# Structural: pre-token scores each entity token from the PRECEDING causal residual
# (the state that predicts it) -- this is the B03 fix. Post-token uses the token's own
# state (h_t predicts x_{t+1}, not x_t), which is the causally-wrong position.
check("pre-token scoring states are the preceding positions",
      pre[0]["score_pos"] == [p - 1 for p in post[0]["score_pos"]],
      f"pre={pre[0]['score_pos']} post={post[0]['score_pos']}")
# Correctness: the pre-token readout is a genuine next-token prediction -- the model
# assigns more probability to the TRUE entity token than to a random wrong token id at
# the same preceding state (surprisal_true < surprisal_random).
r = pre[0]
true_surp = R.logit_lens_score({head: Amod.span_hidden(r, head)}, unembed, final_norm,
                               slice(None), r["tok_ids"], head)
rng2 = torch.Generator().manual_seed(7)
rand_ids = torch.randint(0, unembed.shape[0], r["tok_ids"].shape, generator=rng2)
rand_surp = R.logit_lens_score({head: Amod.span_hidden(r, head)}, unembed, final_norm,
                               slice(None), rand_ids, head)
check("pre-token readout predicts the true entity better than a random token",
      true_surp < rand_surp, f"true={true_surp:.2f} < random={rand_surp:.2f}")

# --------------------------------------------------------------------------- #
print()
if FAIL:
    print(f"SMOKE FAILED: {FAIL}")
    sys.exit(1)
print("SMOKE PASSED: metric, real-gpt2 four-reader path, and synthetic discrimination all green.")
