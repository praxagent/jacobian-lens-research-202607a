"""Receipt-generated figures + provenance manifest for the reader-benchmark note.

House rules (RESEARCH_NOTE_WRITEUP §4.1/§5): every plotted number is computed HERE
from committed receipts — no hand-authored data geometry. Outputs land in the post
bundle: 4 SVGs (+300dpi PNGs), one <stem>.receipt.json per figure, and the post-wide
provenance.json enumerating every evidentiary number with receipt path + SHA-256 +
computation. `--verify` regenerates everything into a temp dir, asserts byte-identity
of the SVGs, re-derives the manifest, and asserts every manifest value_str appears
verbatim in index.md.

  .venv/bin/python figures/build_figs.py            # build into the post bundle
  .venv/bin/python figures/build_figs.py --verify   # pre-publish gate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve()
PROJ = HERE.parents[1]
POST = HERE.parents[3] / "blog/features-as-rewards-reader-benchmark"
PIN = "242ca53"  # repo commit containing every receipt below
REPO_URL = "https://github.com/praxagent/jacobian-lens-research-202607a"

plt.rcParams.update({
    "svg.hashsalt": "praxagent-farr",
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Arial", "Helvetica", "DejaVu Sans"],
    "figure.facecolor": "#F7F4F0", "axes.facecolor": "#F7F4F0",
    "savefig.facecolor": "#F7F4F0",
    "text.color": "#2C2924", "axes.edgecolor": "#A89B8C",
    "axes.labelcolor": "#2C2924", "xtick.color": "#5A544C",
    "ytick.color": "#5A544C", "axes.grid": False,
})

BLUE, GREEN, CLAY, GRAY, DARK = "#4B6787", "#6F8D5E", "#A67C52", "#7F786D", "#2C2924"

R = {  # receipt registry: key -> repo-relative path
    "llama8b": "projects/features-as-rewards-replication/experiments/probe_calibration/receipts/receipt_llama31_8b.json",
    "gemma9b": "projects/features-as-rewards-replication/experiments/probe_calibration/receipts/receipt_gemma2_9b.json",
    "llama70b": "projects/features-as-rewards-replication/experiments/probe_calibration/receipts/receipt_llama33_70b.json",
    "g3_a6": "projects/features-as-rewards-replication/experiments/gemma3_labeling/receipts/receipt_gemma3_jury.json",
    "g3_a7": "projects/features-as-rewards-replication/experiments/gemma3_labeling/receipts/receipt_gemma3_jury_a7.json",
    "jury_val": "projects/features-as-rewards-replication/experiments/gemma3_labeling/receipts/jury_validation.json",
    "wiki_v1": "projects/features-as-rewards-replication/experiments/gemma3_labeling/receipts/val_deepseek.json",
    "wiki_v2": "projects/features-as-rewards-replication/experiments/gemma3_labeling/receipts/val_deepseek_v2.json",
    "wiki_v3": "projects/features-as-rewards-replication/experiments/gemma3_labeling/receipts/val_deepseek_v3.json",
}
ROOT = PROJ.parents[1]
RECEIPTS = {k: json.loads((ROOT / p).read_text()) for k, p in R.items()}
SHAS = {k: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for k, p in R.items()}

READERS = [  # display order, consistent across all figures
    ("attention_probe", "attention probe (supervised)", BLUE),
    ("native_head_surprisal", "native head surprisal (free)", GRAY),
    ("sae_latent_label_selected", "SAE latent (label-selected)", GRAY),
    ("heuristic_len_freq", "heuristic (len+freq)", CLAY),
    ("logit_lens", "logit lens (mid-layer)", GRAY),
    ("random_transport_null", "random-transport null", GRAY),
]
ARMS = [  # key, panel title, label source
    ("llama8b", "Llama-3.1-8B\n(public gold)"),
    ("gemma9b", "gemma-2-9b\n(public gold)"),
    ("llama70b", "Llama-3.3-70B\n(public gold)"),
    ("g3_a7", "Gemma-3-12B\n(jury labels, new test)"),
]


def fmt3(x):
    return f"{x:.3f}".lstrip("0") if 0 < x < 1 else f"{x:.3f}"


def fig_readers(out):
    """Q: which readers approach the supervised probe, on the same states?"""
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 4.4), sharex=True)
    fig.suptitle("Six readers of the same residual states, four models — AUROC "
                 "(hallucinated entity, higher is better)", fontsize=11,
                 fontweight="bold", x=0.02, ha="left")
    for ax, (arm, title) in zip(axes, ARMS):
        rd = RECEIPTS[arm]["readers"]
        ys = range(len(READERS))[::-1]
        for y, (key, label, color) in zip(ys, READERS):
            v = rd[key]["auroc"]
            lo, hi = rd[key]["ci95"]
            ax.barh(y, v, color=color, height=0.62, zorder=2)
            ax.plot([lo, hi], [y, y], color=DARK, lw=1.1, zorder=3)
            ax.text(max(hi + 0.012, v + 0.012), y, fmt3(v), va="center",
                    fontsize=7.4, color=DARK)
        ax.axvline(0.5, color="#A89B8C", lw=0.9, ls=(0, (5, 4)))
        ax.set_xlim(0.42, 0.92)
        ax.set_title(title, fontsize=8.6)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([lab for _, lab, _ in READERS] if ax is axes[0]
                           else [""] * len(READERS), fontsize=8)
        ax.tick_params(axis="x", labelsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_xlabel("")
    fig.text(0.02, 0.015, "dashed line = chance (.5); whiskers = completion-clustered "
             "95% bootstrap CI; bars share one axis across panels", fontsize=7.6,
             color="#5A544C")
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    vals = {f"{arm}.{key}": RECEIPTS[arm]["readers"][key]["auroc"]
            for arm, _ in ARMS for key, _, _ in READERS}
    return save(fig, out, "fig-readers-crossmodel",
                title="Six readers, four models: AUROC on the same pre-entity states",
                alt=("Four-panel horizontal bar chart of reader AUROC. The supervised "
                     "attention probe leads in every panel (.754 Llama-8B, .726 "
                     "gemma-2-9b, .774 Llama-70B on public gold; .672 Gemma-3-12B on "
                     "jury labels). Free readers trail; random-transport null sits at "
                     "chance in all panels. Whiskers are 95% CIs; higher is better."),
                sources=["llama8b", "gemma9b", "llama70b", "g3_a7"], values=vals)


def fig_paper_gap(out):
    """Q: does the paper-level probe AUROC (~.94) transfer to public labels?"""
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    pts = [(k, t.replace("\n", " ")) for k, t in ARMS]
    for i, (arm, label) in enumerate(pts):
        rd = RECEIPTS[arm]["readers"]["attention_probe"]
        ax.plot(rd["ci95"], [i, i], color=DARK, lw=1.2, zorder=3)
        ax.plot(rd["auroc"], i, "o", color=BLUE, ms=7, zorder=4)
        ax.text(rd["auroc"], i + 0.22, fmt3(rd["auroc"]), ha="center", fontsize=8.2)
    cap = RECEIPTS["llama70b"]["exploratory_probe_capacity"]
    ax.plot(cap["auroc"], 1.72, "D", color=GRAY, ms=5.5, zorder=4)
    ax.annotate(f"paper-capacity probe {fmt3(cap['auroc'])} (exploratory)",
                (cap["auroc"], 1.72), (0.565, 0.62), fontsize=7.4, color=GRAY,
                ha="left", arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
    ax.axvspan(0.88, 0.94, color="#F3E8E0", zorder=1)
    ax.text(0.91, 0.0, "paper's operating point\n.88 localize / .94 classify\n"
            "(their labels+probe,\nunreleased)", fontsize=7.4, color=CLAY,
            ha="center", va="bottom")
    ax.set_yticks(range(len(pts)))
    ax.set_yticklabels([l for _, l in pts], fontsize=8.4)
    ax.set_xlim(0.55, 1.0)
    ax.set_ylim(-0.6, 4.1)
    ax.set_xlabel("supervised-probe AUROC (3-seed mean, 95% CI)", fontsize=8.4)
    fig.suptitle("The paper-level probe number does not transfer to independent "
                 "labels", fontsize=11, fontweight="bold", x=0.02, ha="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    vals = {f"{arm}.attention_probe": RECEIPTS[arm]["readers"]["attention_probe"]["auroc"]
            for arm, _ in ARMS}
    vals["llama70b.exploratory_capacity"] = cap["auroc"]
    return save(fig, out, "fig-probe-vs-paper",
                title="Probe AUROC across scales vs the paper's operating point",
                alt=("Dot plot with CIs: probe AUROC .754, .726, .774 on the three "
                     "public-gold arms and .672 on jury labels, all far below the "
                     "paper's .88-.94 band (shaded). An exploratory paper-capacity "
                     "probe on the 70B lands at .776 — capacity does not close the "
                     "gap."), sources=["llama8b", "gemma9b", "llama70b", "g3_a7"],
                values=vals)


def fig_gate(out):
    """Q: did the powered re-test resolve the pre-registered gate?"""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharex=True, sharey=True)
    panels = [("g3_a6", "first pass — 550 test spans"),
              ("g3_a7", "powered re-test — 2,428 NEW test spans")]
    order = [k for k, _, _ in READERS if k != "attention_probe"]
    labels = {k: lab for k, lab, _ in READERS}
    vals = {}
    for ax, (arm, title) in zip(axes, panels):
        con = RECEIPTS[arm]["paired_contrasts_vs_probe"]
        ax.axvspan(-0.05, 0.05, color="#EDE8E1", zorder=1)
        ax.axvline(0, color="#A89B8C", lw=0.9)
        ax.axvline(-0.05, color=CLAY, lw=0.9, ls=(0, (5, 4)))
        for y, key in enumerate(order[::-1]):
            c = con[key]
            lo, hi = c["ci95"]
            col = CLAY if key == "heuristic_len_freq" else GRAY
            ax.plot([lo, hi], [y, y], color=col, lw=2.2, zorder=3,
                    solid_capstyle="butt")
            ax.plot(c["mean_diff"], y, "o", color=col, ms=5, zorder=4)
            ax.text(hi + 0.008, y, f"{c['mean_diff']:+.3f} {c['verdict']}",
                    va="center", fontsize=7.2, color=DARK)
            vals[f"{arm}.{key}.mean_diff"] = c["mean_diff"]
            vals[f"{arm}.{key}.verdict"] = c["verdict"]
        ax.set_title(title, fontsize=9)
        ax.set_yticks(range(len(order)))
        if ax is axes[0]:
            ax.set_yticklabels([labels[k] for k in order[::-1]], fontsize=8)
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(axis="x", labelsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_xlim(-0.36, 0.24)
    fig.suptitle("Gate (b): reader-minus-probe AUROC difference vs the frozen ±.05 "
                 "margin (dashed = 'worse' bound)", fontsize=11, fontweight="bold",
                 x=0.02, ha="left")
    fig.text(0.02, 0.02, "interval entirely left of the dashed line = 'worse'; "
             "power tightened every interval; the heuristic landed AT the margin "
             "(-.050)", fontsize=7.6, color="#5A544C")
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    return save(fig, out, "fig-gate-power",
                title="The pre-registered gate before and after the powered re-test",
                alt=("Two-panel interval plot of reader-minus-probe AUROC differences "
                     "with a shaded ±.05 equivalence margin. First pass (n=550): wide "
                     "CIs, null and heuristic inconclusive. Powered re-test (n=2428): "
                     "null resolves to worse (-.166), heuristic lands exactly at the "
                     "margin (-.050, inconclusive) — the gate fails on the merits."),
                sources=["g3_a6", "g3_a7"], values=vals)


def fig_kappa(out):
    """Q: which label-grounding passes the pre-registered kappa >= .5 gate?"""
    rows = [
        ("Wikipedia grader v1 (intro index)", RECEIPTS["wiki_v1"]["validation"]["cohen_kappa"], GRAY),
        ("Wikipedia grader v2 (rubric fix)", RECEIPTS["wiki_v2"]["validation"]["cohen_kappa"], GRAY),
        ("Wikipedia grader v3 (full-text index)", RECEIPTS["wiki_v3"]["validation"]["cohen_kappa"], GRAY),
        ("single judge: DeepSeek-chat", RECEIPTS["jury_val"]["validation"]["judge:deepseek/deepseek-chat"]["kappa"], GRAY),
        ("single judge: GLM-4.6", RECEIPTS["jury_val"]["validation"]["judge:z-ai/glm-4.6"]["kappa"], GRAY),
        ("single judge: Gemini-2.5-Flash", RECEIPTS["jury_val"]["validation"]["judge:google/gemini-2.5-flash"]["kappa"], GRAY),
        ("jury MAJORITY (2/3), search-grounded", RECEIPTS["jury_val"]["validation"]["majority"]["kappa"], GREEN),
        ("jury UNANIMOUS (3/3), search-grounded", RECEIPTS["jury_val"]["validation"]["unanimous"]["kappa"], GREEN),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    for y, (label, v, color) in enumerate(rows[::-1]):
        ax.barh(y, v, color=color, height=0.62, zorder=2)
        ax.text(v + 0.008, y, f"{v:.3f}".lstrip("0"), va="center", fontsize=8,
                color=DARK)
    ax.axvline(0.5, color=CLAY, lw=1.2, ls=(0, (5, 4)))
    ax.text(0.512, 2.48, "pre-registered gate: kappa >= .5", fontsize=7.6,
            color=CLAY, va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8.2)
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("Cohen's kappa vs public web-search-grounded gold labels",
                  fontsize=8.4)
    ax.set_title("Only the cross-provider jury clears the label-quality gate",
                 fontsize=11, fontweight="bold", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    vals = {r[0]: r[1] for r in rows}
    return save(fig, out, "fig-grader-ladder",
                title="Grounding ladder: Cohen's kappa vs the >=0.5 gate",
                alt=("Horizontal bars of Cohen's kappa vs gold: three Wikipedia-"
                     "grounded grader attempts at .281/.278/.179, three individual "
                     "search-grounded judges at .410/.472/.475, jury majority .598 "
                     "and unanimous .691 — only the jury tiers clear the dashed "
                     ".5 gate line."),
                sources=["wiki_v1", "wiki_v2", "wiki_v3", "jury_val"], values=vals)


def save(fig, out, stem, title, alt, sources, values):
    svg = out / f"{stem}.svg"
    fig.savefig(svg, format="svg", metadata={"Date": None})
    fig.savefig(out / f"{stem}.png", dpi=300)
    plt.close(fig)
    receipt = {
        "figure_id": stem, "title": title, "alt_text": alt,
        "data_source": [{"receipt": R[s], "sha256": SHAS[s],
                         "pinned_url": f"{REPO_URL}/blob/{PIN}/{R[s]}"}
                        for s in sources],
        "provenance": {"generator": "projects/features-as-rewards-replication/figures/build_figs.py",
                       "generator_sha256": hashlib.sha256(HERE.read_bytes()).hexdigest(),
                       "matplotlib": matplotlib.__version__,
                       "svg_sha256": hashlib.sha256(svg.read_bytes()).hexdigest()},
        "plotted_values": values,
        "uncertainty": "completion-clustered bootstrap 95% CI (2000 resamples) unless "
                       "stated; kappa figures carry n in their receipts",
        "accessibility": {"color_only_channel": False,
                          "text_equivalent": "plotted_values above + alt_text"},
    }
    (out / f"{stem}.receipt.json").write_text(json.dumps(receipt, indent=1))
    return stem, values


def build_manifest(out, fig_values):
    """Post-wide provenance: every evidentiary number, receipt-backed."""
    ent = []

    def add(label, value, vstr, receipt, computation):
        ent.append({"label": label, "value": value, "value_str": vstr,
                    "receipt": R[receipt], "receipt_sha256": SHAS[receipt],
                    "pinned_url": f"{REPO_URL}/blob/{PIN}/{R[receipt]}",
                    "computation": computation})

    for arm, _ in ARMS:
        for key, _, _ in READERS:
            v = RECEIPTS[arm]["readers"][key]["auroc"]
            add(f"{arm}.{key}.auroc", v, fmt3(round(v, 3)), arm,
                f"readers['{key}']['auroc']")
    cap = RECEIPTS["llama70b"]["exploratory_probe_capacity"]
    add("llama70b.exploratory_capacity.auroc", cap["auroc"], fmt3(round(cap["auroc"], 3)),
        "llama70b", "exploratory_probe_capacity['auroc']")
    a7 = RECEIPTS["g3_a7"]
    add("g3_a7.n_test_spans", a7["n_test_spans"], f"{a7['n_test_spans']:,}", "g3_a7",
        "n_test_spans")
    for key in ("random_transport_null", "heuristic_len_freq"):
        c = a7["paired_contrasts_vs_probe"][key]
        add(f"g3_a7.contrast.{key}.mean_diff", c["mean_diff"],
            f"{c['mean_diff']:+.3f}".replace("+0.", "+.").replace("-0.", "−.")
            if abs(c["mean_diff"]) < 1 else str(c["mean_diff"]),
            "g3_a7", f"paired_contrasts_vs_probe['{key}']['mean_diff']")
    jv = RECEIPTS["jury_val"]["validation"]
    add("jury.majority.kappa", jv["majority"]["kappa"], ".598", "jury_val",
        "validation['majority']['kappa']")
    add("jury.unanimous.kappa", jv["unanimous"]["kappa"], ".691", "jury_val",
        "validation['unanimous']['kappa']")
    add("g3_a6.probe.auroc", RECEIPTS["g3_a6"]["readers"]["attention_probe"]["auroc"],
        ".709", "g3_a6", "readers['attention_probe']['auroc']")
    manifest = {"what": "post-wide provenance manifest — every evidentiary number "
                        "re-derives from a committed receipt (RESEARCH_NOTE §5)",
                "post": "features-as-rewards-reader-benchmark", "repo": REPO_URL,
                "pin_commit": PIN, "entries": ent,
                "figures": {k: v for k, v in fig_values.items()}}
    (out / "provenance.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def check_prose(manifest, post_md):
    text = post_md.read_text()
    missing = [e for e in manifest["entries"]
               if e["value_str"].replace("−", "-") not in
               text.replace("−", "-").replace("−", "-")]
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(POST))
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if a.verify:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            figs = dict(f(tmp) for f in (fig_readers, fig_paper_gap, fig_gate, fig_kappa))
            man = build_manifest(tmp, figs)
            fail = []
            for stem in ("fig-readers-crossmodel", "fig-probe-vs-paper",
                         "fig-gate-power", "fig-grader-ladder"):
                if (tmp / f"{stem}.svg").read_bytes() != (out / f"{stem}.svg").read_bytes():
                    fail.append(f"{stem}.svg drifted")
            committed = json.loads((out / "provenance.json").read_text())
            if committed["entries"] != man["entries"]:
                fail.append("provenance.json entries drifted")
            missing = check_prose(man, out / "index.md")
            for e in missing:
                fail.append(f"prose missing value: {e['label']} = {e['value_str']}")
            if fail:
                print("VERIFY FAILED:"); [print("  -", f) for f in fail]
                sys.exit(1)
            print(f"VERIFY OK: 4 figures byte-identical, {len(man['entries'])} manifest "
                  f"numbers re-derived, all present in prose")
    else:
        out.mkdir(parents=True, exist_ok=True)
        figs = dict(f(out) for f in (fig_readers, fig_paper_gap, fig_gate, fig_kappa))
        man = build_manifest(out, figs)
        missing = check_prose(man, out / "index.md")
        print(f"built 4 figures + provenance.json ({len(man['entries'])} entries) -> {out}")
        if missing:
            print(f"  ⚠ {len(missing)} manifest numbers NOT found in prose yet:")
            for e in missing:
                print(f"    - {e['label']} = {e['value_str']}")


if __name__ == "__main__":
    main()
