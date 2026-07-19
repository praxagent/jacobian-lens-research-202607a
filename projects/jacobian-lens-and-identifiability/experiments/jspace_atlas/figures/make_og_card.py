"""og-card for the jspace-atlas-campaign note: conceptual 1200x630, no empirical data.
Committed per posts-guide 3.4 (conceptual card -> generator committed, no receipt needed)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
fig = plt.figure(figsize=(12, 6.3), dpi=100)
fig.patch.set_facecolor("#F7F4F0")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1200); ax.set_ylim(0, 630); ax.axis("off")
# abstract three-block diagonal motif (conceptual, no data)
x0, y0, w = 700, 110, 420
blocks = [(0.0, 0.30, "#A5C0DC"), (0.30, 0.72, "#4B6787"), (0.72, 1.0, "#7F94B0")]
for a, b, c in blocks:
    ax.add_patch(mp.FancyBboxPatch((x0 + a * w, y0 + a * w), (b - a) * w, (b - a) * w,
                 boxstyle="round,pad=2,rounding_size=8", fc=c, ec="none", alpha=0.9))
ax.add_patch(mp.FancyBboxPatch((x0, y0), w, w, boxstyle="round,pad=3,rounding_size=10",
             fc="none", ec="#A89B8C", lw=1.5))
ax.text(x0 + w/2, y0 - 34, "depth phases, model by model", ha="center", fontsize=15,
        color="#5A544C", family="sans-serif")
ax.text(64, 480, "The J-Space", fontsize=58, fontweight="bold", color="#2C2924",
        family="sans-serif")
ax.text(64, 400, "Atlas Campaign", fontsize=58, fontweight="bold", color="#2C2924",
        family="sans-serif")
ax.text(64, 330, "36 models · 5 families · a 500× scale span", fontsize=22,
        color="#5A544C", family="sans-serif")
ax.text(64, 285, "plan + pre-registration, before outcomes", fontsize=22,
        color="#A67C52", family="sans-serif")
ax.text(64, 130, "praxagent.ai — every map ships with its null",
        fontsize=17, color="#6F8D5E", family="sans-serif")
fig.savefig("/home/ubuntu/PRAX/pre-blog/blog-source/content/posts/2026/07/jspace-atlas-campaign/og-card.png")
print("og-card written 1200x630")
