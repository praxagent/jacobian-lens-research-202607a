"""B08 control-gate suite (CPU, gpt2). ALL gates must pass before any arm's scores are
trusted; a failed gate means a pipeline bug -> outcome-masked implementation amendment +
rerun, never a report/no-report choice.

  G1 native-logit positive control: the output-head lens reproduces the model's OWN
     forward-pass NLL of the entity tokens within tolerance (proves alignment + shift +
     final-norm + unembed are all correct end to end).
  G2 J-lens identity control: identity transport == logit lens exactly.
  G3 alignment fixture: pre-token score positions are the preceding causal positions.
  G4 SAE compatibility: encoder shapes round-trip; a known direction injected into h
     lights the matching latent (encode fixture).
  G5 leakage negative control: a probe trained on SHUFFLED labels is at chance on
     held-out data under the same fitting pipeline.
  G6 procedure-matched SAE null: running the SAME select_latent procedure on permuted
     labels yields (expected) high selected-val-AUROC but ~chance TEST AUROC -- the null
     that matches the label-selected reader's selection pressure.

Run: .venv/bin/python experiments/probe_calibration/gates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import readers as R   # noqa: E402
import data as D      # noqa: E402
import acts as A      # noqa: E402
import sae as S       # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
FAIL = []


def gate(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", output_hidden_states=True)
model.eval()
d = model.config.n_embd
unembed, final_norm = A.model_readout_parts(model)
HEAD = model.config.n_layer          # last hidden state index

comp = "The Eiffel Tower in Paris was completed in 1889 and remains a famous landmark."
ex = D.Example("g:0", "q", comp, [])
enc = tok(comp, add_special_tokens=False, return_tensors="pt")
ids = enc["input_ids"][0]
for word in (" Paris", " 1889"):
    wids = tok(word, add_special_tokens=False)["input_ids"]
    L = ids.tolist()
    st = next(i for i in range(1, len(L) - len(wids) + 1) if L[i:i + len(wids)] == wids)
    cs = len(tok.decode(ids[:st]))
    ex.spans.append(D.Span(cs, cs + len(word), word.strip(), 0))
ex = D.align_spans(ex, tok)
recs = A.cache_spans(model, tok, [ex], [HEAD], scoring="pre_token")

# --- G1: native-logit positive control ---
with torch.no_grad():
    out = model(input_ids=enc["input_ids"])
    logp = torch.log_softmax(out.logits[0].float(), dim=-1)
for r in recs:
    # model's own NLL of token x_t from position t-1:
    own = float(np.mean([-logp[p, t].item() for p, t in zip(r["score_pos"], r["tok_ids"])]))
    lens = R.logit_lens_score({HEAD: A.span_hidden(r, HEAD)}, unembed, final_norm,
                              slice(None), r["tok_ids"], HEAD, head_layer=HEAD)
    gate(f"G1 native-logit control ({tok.decode(r['tok_ids'])!r})",
         abs(own - lens) < 0.05, f"forward={own:.4f} lens={lens:.4f}")

# --- G2: identity transport == logit lens ---
r = recs[0]
a = R.logit_lens_score({HEAD: A.span_hidden(r, HEAD)}, unembed, final_norm,
                       slice(None), r["tok_ids"], HEAD)
b = R.jlens_score({HEAD: A.span_hidden(r, HEAD)}, unembed, final_norm,
                  slice(None), r["tok_ids"], HEAD, transport=torch.eye(d))
gate("G2 J-lens identity == logit lens", abs(a - b) < 1e-4, f"{a:.5f} vs {b:.5f}")

# --- G3: alignment fixture ---
gate("G3 pre-token positions are t-1",
     all(r["score_pos"][0] == ex.spans[i].tok_start - 1 for i, r in enumerate(recs)))

# --- G4: SAE encode fixture ---
gen = torch.Generator().manual_seed(3)
W = torch.randn(512, d, generator=gen)
W = W / W.norm(dim=1, keepdim=True)
sae_r = S.SAEReader(W, torch.zeros(512))
probe_dir = W[137]
h = probe_dir.unsqueeze(0) * 5.0
acts = sae_r.encode(h)
gate("G4 SAE encode fixture: injected direction lights its latent",
     int(acts[0].argmax()) == 137, f"argmax={int(acts[0].argmax())}")

# --- G5: shuffled-label probe at chance ---
N, Sp = 200, 4
X = torch.randn(N, Sp, d) * 0.5
dirn = torch.randn(d); dirn /= dirn.norm()
W[100] = dirn                      # plant the signal latent so real-label selection can find it
sae_r = S.SAEReader(W, torch.zeros(512))
y = (torch.arange(N) % 2).numpy()
for i in range(N):
    if y[i]:
        X[i] += 1.0 * dirn
M = torch.ones(N, Sp)
rng = np.random.default_rng(0)
ysh = rng.permutation(y[:120])
probe, _ = R.train_probe(X[:120], M[:120], ysh, d, epochs=150, seed=0)
with torch.no_grad():
    sc = torch.sigmoid(probe(X[120:], M[120:])).numpy()
a5 = R.auroc(sc, y[120:])
gate("G5 shuffled-label probe at chance on held-out", 0.35 < a5 < 0.65, f"AUROC={a5:.3f}")

# --- G6: procedure-matched SAE null ---
hs = [X[i] for i in range(N)]
li, sgn, va_a = S.select_latent(sae_r, hs[:120], rng.permutation(y[:120]), R.auroc)
te_scores = np.array([sgn * sae_r.encode(h)[:, li].mean().item() for h in hs[120:]])
a6 = R.auroc(te_scores, y[120:])
gate("G6 procedure-matched SAE null: selected-on-permuted is ~chance on TEST",
     0.3 < a6 < 0.7, f"selected latent {li} val={va_a:.3f} -> test={a6:.3f}")
# and the REAL labels through the same procedure DO transfer (sanity that G6 can pass)
li2, sgn2, va2 = S.select_latent(sae_r, hs[:120], y[:120], R.auroc)
te2 = np.array([sgn2 * sae_r.encode(h)[:, li2].mean().item() for h in hs[120:]])
a6b = R.auroc(te2, y[120:])
gate("G6b same procedure on REAL labels transfers to test", a6b > 0.75, f"test={a6b:.3f}")

print()
if FAIL:
    print(f"GATES FAILED: {FAIL}")
    sys.exit(1)
print("ALL GATES PASS")
