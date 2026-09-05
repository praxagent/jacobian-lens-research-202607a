"""python -m jlens_atlas --neuronpedia gpt2-small --out out/gpt2
   python -m jlens_atlas --lens my_lens.pt --model Qwen/Qwen3-4B --probe own --out out/q4b
   python -m jlens_atlas --hf praxagent-org/jacobian-lens-gemma-2-9b --file jlens/wikitext/gemma2_9b.pt --model google/gemma-2-9b --out out/g9b
"""
from __future__ import annotations
import argparse, sys
from .run import run_atlas


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jlens_atlas", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--lens", help="local jlens .pt checkpoint")
    src.add_argument("--neuronpedia", metavar="SLUG",
                     help="slug in the neuronpedia/jacobian-lens collection (model id read from its config)")
    src.add_argument("--hf", metavar="REPO_ID", help="any HF repo holding a jlens .pt (use with --file)")
    ap.add_argument("--file", help="path of the .pt inside --hf")
    ap.add_argument("--revision", help="HF revision for --hf")
    ap.add_argument("--model", help="HF model id whose unembedding the lens reads out through "
                                    "(required unless --neuronpedia)")
    ap.add_argument("--probe", choices=("shared", "own"), default="shared",
                    help="shared: 4,096 strings common to all zoo tokenizers (cross-model safe); "
                         "own: 4,096 rows sampled from this model's own vocabulary")
    ap.add_argument("--n-probe", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--geometry-dtype", choices=("fp32", "fp16"), default="fp32",
                    help="fp16 reproduces the cached atlas maps exactly (they stored D in fp16)")
    ap.add_argument("--null", choices=("frob", "std"), default="frob",
                    help="random-transport null: Frobenius-matched (note convention) or std-matched")
    ap.add_argument("--no-pr", action="store_true", help="skip per-layer participation ratio")
    ap.add_argument("--shared-tokens", help="alternative shared-token list (json with 'strings')")
    ap.add_argument("--title", help="heatmap title")
    ap.add_argument("--out", default="atlas_out", help="output directory")
    a = ap.parse_args(argv)
    if a.hf and not a.file:
        ap.error("--hf needs --file")
    s = run_atlas(lens_path=a.lens, neuronpedia=a.neuronpedia, hf_repo=a.hf, hf_file=a.file,
                  hf_revision=a.revision, model=a.model, probe=a.probe, n_probe=a.n_probe,
                  seed=a.seed, geometry_dtype=a.geometry_dtype, null=a.null, pr=not a.no_pr,
                  out_dir=a.out, shared_tokens=a.shared_tokens, title=a.title,
                  argv=["python", "-m", "jlens_atlas"] + (argv if argv is not None else sys.argv[1:]))
    seg = s["fitted_seg"]; ident = s["boundary_identifiability"]
    print(f"{s['model']}  L={s['n_layers']}  probe={s['probe']['probe']} (n={s['probe']['n_probe']})")
    print(f"  mid_sep {s['mid_sep']:+.4f}   fitted_sep {s['fitted_sep']:+.4f}   boundaries {seg}"
          if seg else f"  mid_sep {s['mid_sep']:+.4f}   (too few layers to fit a 3-segmentation)")
    if ident:
        print(f"  boundary spread b1 {ident['spread_b1']:.2f} b2 {ident['spread_b2']:.2f} of depth "
              f"-> {'identified' if ident['identified'] else 'NOT identified'}")
    print(f"  random-transport null mid_sep {s['null']['mid_sep']:+.4f}   "
          f"off-diagonal range [{s['offdiag_min']:.3f}, {s['offdiag_max']:.3f}]")
    print(f"  wrote {a.out}/cka.npz summary.json receipt.json heatmap.png")


if __name__ == "__main__":
    main()
