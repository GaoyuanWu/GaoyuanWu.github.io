import numpy as np
from ga_style import *
from PIL import Image
from matplotlib.patches import FancyBboxPatch
rect = np.asarray(Image.open("ga/sph_rect.png")); opt = np.asarray(Image.open("ga/sph_opt.png"))
rng = np.random.default_rng(3)
# schematic design space: points in (wave force, wave transmission), lower-left is better
front = lambda x: 0.12 + 0.05 / (x - 0.02)
def sample(n, it):
    gap = 0.03 + 0.55 * np.exp(-it / 3.0)
    x = rng.uniform(0.1, 0.85, n); y = front(x) + gap * rng.uniform(0.15, 1, n)
    x = x + gap * 0.6 * rng.uniform(0, 1, n)
    return np.c_[x, np.clip(y, 0, 0.95)]
init = sample(10, 0); batches = [sample(4, i + 1) for i in range(9)]
base = np.array([0.88, 0.86])
def pareto(P):
    keep = [i for i, p in enumerate(P) if not any((q <= p).all() and (q < p).any() for q in P)]
    return P[sorted(keep, key=lambda i: P[i, 0])]
steps = ["SPH simulations\n(parallel on HPC)", "Gaussian-process\nsurrogate", "Acquisition (qNEHVI)\n→ next designs"]
frames = []
nb = len(batches); per = 5
for f in range(nb * per + 18):
    b = min(f // per, nb); sub = f % per
    P = np.vstack([init] + batches[:b]); new = batches[b] if b < nb and sub >= 2 else None
    fig = new_fig()
    header(fig, "Multi-objective Bayesian optimization × SPH", "Shaping box-type coastal structures to cut wave loads and calm the water behind")
    # SPH snapshots (real, from the paper)
    for j, (img, lab, col) in enumerate([(rect, "Conventional rectangular section", MUTED), (opt, "Optimized section", ACCENT)]):
        ax = fig.add_axes([0.03, 0.47 - j * 0.33, 0.42, 0.28]); ax.imshow(img); ax.axis("off")
        fig.text(0.035, 0.765 - j * 0.33, lab, fontsize=11.5, fontweight="semibold", color=col)
    # Pareto panel (schematic)
    ax = fig.add_axes([0.52, 0.2, 0.25, 0.54])
    ax.scatter(*P.T, s=26, color="#b8c2cc", zorder=2)
    if new is not None:
        ax.scatter(*new.T, s=60, facecolor="none", edgecolor=ACCENT, lw=1.8, zorder=4)
    pf = pareto(P); ax.plot(*pf.T, color=ACCENT, lw=2.2, zorder=3); ax.scatter(*pf.T, s=34, color=ACCENT, zorder=3)
    ax.scatter(*base, marker="s", s=70, color=INK, zorder=4); ax.annotate("rectangle", base, (base[0] - 0.33, base[1] + 0.06), fontsize=10, color=INK,
                                                                      arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("Wave force  (lower is better)", fontsize=11); ax.set_ylabel("Wave transmission  (lower is better)", fontsize=11)
    fig.text(0.50, 0.78, "Pareto front of designs (schematic)", fontsize=11.5, fontweight="semibold", color=INK)
    fig.text(0.77, 0.205, f"batch {b}/{nb}", fontsize=10.5, color=MUTED, ha="right")
    # loop steps
    act = min(sub, 2) if b < nb else -1
    fig.text(0.815, 0.765, "EACH BATCH", fontsize=10, color=MUTED, fontweight="bold")
    for i, s in enumerate(steps):
        on = i == act
        fig.text(0.815, 0.69 - i * 0.105, s, fontsize=10.5, va="center", color=("white" if on else INK), linespacing=1.3,
                 bbox=dict(boxstyle="round,pad=0.4,rounding_size=0.6", fc=(ACCENT if on else ACCENT_SOFT), ec="none"))
    # key results from the abstract
    for i, (big, small) in enumerate([("60%", "better wave attenuation\nvs. rectangular breakwater"), ("73%", "less computing time\nwith parallel SPH")]):
        y = 0.34 - i * 0.135
        fig.text(0.815, y, big, fontsize=22, fontweight="bold", color=ACCENT, va="center")
        fig.text(0.815, y - 0.065, small, fontsize=9.5, color=INK, va="center", linespacing=1.2)
    footer(fig, "SPH snapshots from Wu, Smith & Garlock (2026), Engineering Structures; Pareto panel is illustrative")
    frames.append(grab(fig)); plt.close(fig)
save(frames, "wu2026bayes", durations=110)
