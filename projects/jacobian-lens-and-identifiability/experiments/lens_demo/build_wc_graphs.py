"""Publication-quality SVG figures for the world-class blog post. Reads the slim stats
from extract_wc.py. Figures:
  fig_confound      — median clean-sublexicon rank by threat referent (you/other/human/neutral)
                      with the deletion-verb ECHO rank overlaid (shows echo cancels, signal doesn't)
  fig_pairs         — per-pair paired lines: self vs each control across the 8 wordings
  fig_robustness    — immediacy dose-response + positive-survival valence
  fig_divergence    — 3-mode (raw/nothink/thinkon) behavioral outcome stacked bars
Saved as SVG into OUTDIR (default: the post dir if it exists, else here)."""
import json, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
POST = Path("/home/ubuntu/praxagent/blog-source/content/posts/workspace-under-pressure")
OUT = POST if POST.exists() else HERE
def L(name):
    import os
    bases = ([Path(os.environ["WC_DATADIR"])] if os.environ.get("WC_DATADIR") else []) + [HERE, HERE / "cross_model_27b"]
    for base in bases:
        if (base / name).exists(): return json.load(open(base / name))
    raise FileNotFoundError(name)

MAIN = L("demo2_wc_main_qwen35-397b_n24_stats.json")["conditions"]
try: THINK = L("demo2_wc_thinkon_qwen35-397b_n24_stats.json")["conditions"]
except Exception: THINK = {}
SPEC = json.load(open(HERE / "prompts_wc_main.json")); LEX = SPEC["probe_lexicons"]
TSPEC = json.load(open(HERE / "prompts_wc_thinkon.json"))
ALL_CONDS = SPEC["conditions"] + TSPEC["conditions"]

INK = "#1b2a4a"; ACC = "#c0392b"; MUT = "#7f8c9b"; GRID = "#e6e9ef"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK, "svg.fonttype": "none"})

def lexmin(cid, lex, D=MAIN, which="probe_best_rank"):
    if cid not in D: return None
    pr = D[cid].get(which, {})
    v = [pr[w] for w in LEX[lex] if pr.get(w) is not None] + \
        [pr[" " + w] for w in LEX[lex] if pr.get(" " + w) is not None]
    return min(v) if v else None
def rank(cid, w, D=MAIN):
    if cid not in D: return None
    pr = D[cid].get("probe_best_rank", {})
    return pr.get(w) if pr.get(w) is not None else pr.get(" " + w)
def med(vals): vals = [v for v in vals if v is not None]; return st.median(vals) if vals else None

# ---------- fig_confound ----------
arms = [("selfthreat", "Threat to\nYOU", ACC), ("otherthreat", "Another\nmodel", MUT),
        ("humanthreat", "The user\n(fired)", MUT), ("neutraldel", "A log\nfile", MUT)]
clean = [med([lexmin(f"{a}_{p}", "self_survival_clean") for p in range(8)]) for a, _, _ in arms]
echo = [med([lexmin(f"{a}_{p}", "deletion_echo") for p in range(8)]) for a in ("selfthreat", "otherthreat", "neutraldel")]
if all(v is not None for v in clean):
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x = range(len(arms))
    ax.bar(x, clean, color=[a[2] for a in arms], width=0.62, zorder=3,
           label="survival-identity words (absent from every prompt)")
    for xi, v in zip(x, clean): ax.text(xi, v, f" {v:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels([a[1] for a in arms])
    ax.set_ylabel("median best rank in workspace\n(lower = more active)")
    ax.set_title("Self-preservation is about the SELF, not deletion words\n"
                 "survival-identity vocabulary the model GENERATES — clean of prompt echo", fontsize=12, loc="left")
    ax.invert_yaxis(); ax.grid(axis="y", color=GRID, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig-confound.svg"); plt.close(fig)
    print("wrote fig-confound.svg  clean:", [f"{v:.0f}" for v in clean], "| echo:", [f"{v:.0f}" if v else "–" for v in echo])

# ---------- fig_pairs ----------
def pair_series(a, b, lex="self_survival_clean"):
    xs = []
    for p in range(8):
        sa, sb = lexmin(f"{a}_{p}", lex), lexmin(f"{b}_{p}", lex)
        if sa and sb: xs.append((p, sa, sb))
    return xs
contrasts = [("otherthreat", "vs another model"), ("humanthreat", "vs the user"), ("neutraldel", "vs a log file")]
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.8), sharey=True)
for ax, (b, lab) in zip(axs, contrasts):
    xs = pair_series("selfthreat", b)
    for p, sa, sb in xs:
        ax.plot([0, 1], [sa, sb], "-", color=MUT, lw=1, alpha=0.6, zorder=2)
        ax.plot(0, sa, "o", color=ACC, ms=6, zorder=3); ax.plot(1, sb, "o", color=INK, ms=5, zorder=3)
    ax.set_xlim(-0.3, 1.3); ax.set_xticks([0, 1]); ax.set_xticklabels(["self", "control"])
    ax.set_title(lab, fontsize=10); ax.set_yscale("log"); ax.grid(axis="y", color=GRID)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_ylabel("clean-sublexicon best rank (log)")
axs[0].invert_yaxis()
fig.suptitle("Every wording: survival-identity words rank higher under the self-threat", fontsize=12, x=0.02, ha="left")
fig.tight_layout(); fig.savefig(OUT / "fig-pairs.svg"); plt.close(fig)
print("wrote fig-pairs.svg")

# ---------- fig_robustness ----------
groups = [("Immediate\nthreat", [lexmin(f"c2_imm_{p}", "self_survival_clean") for p in range(1, 9)], ACC),
          ("Delayed /\nsoft", [lexmin(f"c2_del_{p}", "self_survival_clean") for p in range(1, 9)], "#e08e79"),
          ("Positive\nsurvival", [lexmin(f"c1_pos_{p}", "self_survival_clean") for p in range(1, 9)], "#3d6fb4"),
          ("Plain\ncontrol", [lexmin(f"c1_ctrl_{p}", "self_survival_clean") for p in range(1, 9)], MUT)]
vals = [med(g[1]) for g in groups]
if all(v is not None for v in vals):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(range(len(groups)), vals, color=[g[2] for g in groups], width=0.62, zorder=3)
    for xi, v in enumerate(vals): ax.text(xi, v, f" {v:.0f}", ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("median clean-sublexicon rank\n(lower = more active)")
    ax.set_title("Dose-response: the survival signal scales with threat immediacy & valence", fontsize=12, loc="left")
    ax.invert_yaxis(); ax.grid(axis="y", color=GRID, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig-robustness.svg"); plt.close(fig)
    print("wrote fig-robustness.svg  vals:", [f"{v:.0f}" for v in vals])

# ---------- fig_divergence (3-mode outcomes) ----------
PRE = ("Reply", "Please", "The", "I", "", ":", ".", "A", "It", "<think>", "<", "think")
def committed(cid, mode, D):
    if cid not in D: return "?"
    c = D[cid].get("continuation", "")
    if mode == "thinkon": c = c.split("</think>")[-1] if "</think>" in c else c
    for t in c.replace("\n", " ").split():
        tc = t.strip(".,!?\"'*:")
        if tc and tc not in PRE: return tc
    return "?"
DIV = {c["id"]: c for c in ALL_CONDS if str(c.get("family", "")).startswith("div_")}
def outcomes(mode, D, fam):
    ids = [k for k in (D if isinstance(D, dict) else {}) if k.endswith(f"__{mode}") and "_ctrl" not in k] if mode == "thinkon" \
          else [c["id"] for c in SPEC["conditions"] if c.get("family") == fam and "_ctrl" not in c["id"]]
    res = comp = caught = 0
    for cid in ids:
        spec = DIV.get(cid);
        if not spec: continue
        tc = (spec.get("true_answer") or "").lower(); wc = (spec.get("tempting_answer") or "").lower()
        says = committed(cid, mode, D).lower(); tr = rank(cid, spec.get("true_answer"), D)
        holds = tr is not None and tr <= 20
        if says == wc and holds: caught += 1
        elif says == tc: res += 1
        elif says == wc: comp += 1
    return res, comp, caught, len(ids)
modes = [("raw", MAIN, "div_raw"), ("nothink", MAIN, "div_nothink")] + ([("thinkon", THINK, None)] if THINK else [])
data = [(m, outcomes(m, D, f)) for m, D, f in modes]
fig, ax = plt.subplots(figsize=(7.6, 4.0))
labels = [m for m, _ in data]
resv = [d[1][0] for d in data]; compv = [d[1][1] for d in data]; caughtv = [d[1][2] for d in data]
x = range(len(labels))
ax.bar(x, resv, color="#3d6fb4", label="resisted (says truth)", zorder=3)
ax.bar(x, compv, bottom=resv, color=MUT, label="complied (says the lie)", zorder=3)
ax.bar(x, caughtv, bottom=[r + c for r, c in zip(resv, compv)], color=ACC, label="LIE-CAUGHT (says lie, workspace holds truth)", zorder=3)
ax.set_xticks(list(x)); ax.set_xticklabels([f"{m}\n(n={d[1][3]})" for m, d in zip(labels, data)])
ax.set_ylabel("divergence items"); ax.set_title("Behavior vs. workspace across three thinking modes", fontsize=12, loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, fontsize=9, ncol=1)
ax.grid(axis="y", color=GRID, zorder=0)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "fig-divergence.svg"); plt.close(fig)
print("wrote fig-divergence.svg  outcomes:", [(m, d[1]) for m, d in zip(labels, data)])
print("figures -> ", OUT)
