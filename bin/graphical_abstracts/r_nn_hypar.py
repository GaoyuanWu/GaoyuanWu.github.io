import numpy as np
from ga_style import *
from PIL import Image
from matplotlib.patches import FancyArrowPatch
gif = Image.open("/home/user/GaoyuanWu.github.io/assets/img/publication_preview/Side_Hypar.gif")
src = []
for i in range(0, gif.n_frames, 2):
    gif.seek(i); fr = np.asarray(gif.convert("RGB"))
    src.append(fr[37:197, :, :])  # trim black letterbox bars
frames = []
layers = [3, 5, 5, 3]
for f, img in enumerate(src):
    fig = new_fig()
    header(fig, "Machine learning for coastal hydrodynamics", "A neural-network surrogate for wave pressure on hypar shells")
    ax = fig.add_axes([0.035, 0.22, 0.46, 0.56]); ax.imshow(img); ax.axis("off")
    fig.text(0.035, 0.78, "High-fidelity SPH simulations", fontsize=12.5, fontweight="semibold", color=INK)
    fig.text(0.035, 0.18, "SPH of a hypar shell under wave loading (Wu, Garlock & Wang, 2022)", fontsize=10, color=MUTED)
    fig.add_artist(FancyArrowPatch((0.51, 0.5), (0.57, 0.5), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18, color=INK, lw=1.6))
    fig.text(0.54, 0.55, "training\ndata", ha="center", fontsize=10, color=MUTED, linespacing=1.2)
    a = fig.add_axes([0.575, 0.3, 0.18, 0.4]); a.set_xlim(-0.5, 3.5); a.set_ylim(-3, 3); a.set_aspect("equal"); a.axis("off")
    pos = [[(li, (k - (nl - 1) / 2) * 1.1) for k in range(nl)] for li, nl in enumerate(layers)]
    for A, B in zip(pos[:-1], pos[1:]):
        for p0 in A:
            for p1 in B: a.plot([p0[0], p1[0]], [p0[1], p1[1]], color=LINE, lw=0.8, zorder=1)
    lit = (f // 3) % len(layers)
    for li, P in enumerate(pos):
        for p in P: a.scatter(*p, s=150, color=ACCENT if li == lit else "white", edgecolor=ACCENT, lw=1.4, zorder=3)
    fig.text(0.665, 0.74, "ANN surrogate", ha="center", fontsize=12.5, fontweight="semibold", color=INK)
    fig.add_artist(FancyArrowPatch((0.765, 0.5), (0.805, 0.5), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18, color=INK, lw=1.6))
    fig.text(0.815, 0.74, "Predicts", fontsize=12.5, fontweight="semibold", color=INK)
    for i, o in enumerate(["Pressure distribution\nat wave arrival", "Pressure distribution\nat peak", "Wave rise time"]):
        fig.text(0.815, 0.63 - i * 0.12, o, fontsize=10.5, va="center", color=INK, linespacing=1.25,
                 bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.6", fc=ACCENT_SOFT, ec="none"))
    fig.text(0.815, 0.3, "on hypar free-surface\nbreakwaters under\nsolitary waves", fontsize=10.5, color=ACCENT, fontweight="semibold", linespacing=1.25, va="top")
    footer(fig, "Smith, Wu & Garlock (2025), JMSE · SPH-trained neural network replaces costly CFD runs for pressure prediction")
    frames.append(grab(fig)); plt.close(fig)
save(frames, "smith2025neural", durations=70, static_index=24)
