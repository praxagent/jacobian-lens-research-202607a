"""Decode the top-4 invariant readout directions to tokens. Direction 3 is the one the
decomposition could not explain (model prior R^2 0.034), so this asks what it actually is."""
import json, numpy as np, torch
from pathlib import Path
from transformers import AutoTokenizer
HERE = Path(__file__).resolve().parent
DEC = HERE.parents[0].parent / "experiments/jspace_atlas/decompose_out"
A = np.load(DEC / "gemma2_9b_A.npy")
ev, evec = np.linalg.eigh(A); o = np.argsort(ev)[::-1]; ev, evec = ev[o], evec[:, o]
d = torch.load("/home/ubuntu/hf-lenses-tier1/gemma2_9b_wiki.pt", map_location="cpu", weights_only=False)
J = d["J"]; layers = sorted(J.keys())
U = np.load(DEC / "gemma2_9b_embed.npy", mmap_mode="r"); vocab = U.shape[0]
tok = AutoTokenizer.from_pretrained("google/gemma-2-9b")
out = {}
for k in range(4):
    w = evec[:, k] / np.linalg.norm(evec[:, k])
    V = np.stack([J[l].float().numpy() @ w for l in layers]).astype(np.float32)
    e = np.zeros(vocab)
    for i in range(0, vocab, 8192):
        e[i:i+8192] = (np.asarray(U[i:i+8192], dtype=np.float32) @ V.T).mean(1)
    if abs(e.min()) > abs(e.max()): e = -e
    pos = [tok.convert_ids_to_tokens(int(i)) for i in np.argsort(e)[::-1][:18]]
    neg = [tok.convert_ids_to_tokens(int(i)) for i in np.argsort(e)[:18]]
    out[k] = {"share": float(ev[k]/ev.sum()), "pos": pos, "neg": neg}
    print(f"--- direction {k} (share {ev[k]/ev.sum():.3f}) ---")
    print(f"  + : {' '.join(repr(t) for t in pos[:14])}")
    print(f"  - : {' '.join(repr(t) for t in neg[:14])}", flush=True)
Path(HERE/"direction_tokens.json").write_text(json.dumps(out, indent=1))
print("DECODE_DIRS_DONE")
