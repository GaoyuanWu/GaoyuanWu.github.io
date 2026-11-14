import numpy as np
from ga_style import *
from PIL import Image
Fz = np.load("ga/oe_force.npy", allow_pickle=True).item()     # digitized Fig. 25b (elevated, H = 0.275 m, H/λ = 0.05)
snaps = [[np.asarray(Image.open(f"ga/oe/p_{i}_{j}.png")) for j in range(3)] for i in range(3)]  # pressure fields (Fig. 26)
times = [8.30, 8.44, 8.55]
names = ["ST90", "CC60", "CX60"]; labels = ["ST-90  (discontinuous)", "CC-60  (concave)", "CX-60  (convex)"]
cols = {"ST90": INK, "CC60": TEAL, "CX60": BLUE}
def series(k):
    d = Fz[k]; T = d[:, 0]; hi = d[:, 2]
    tt = np.linspace(8.0, 9.0, 900); y = np.interp(tt, T, hi, left=0)
    y[tt < 8.39] = 0; return tt, y
S_ = {k: series(k) for k in names}
peaks = {k: S_[k][1].max() for k in names}
ts = list(np.linspace(8.22, 8.75, 66)) + [8.75] * 16
frames = []
for f, tc in enumerate(ts):
    fig = new_fig()
    header(fig, "Wave–structure interaction · SPH", "A smooth cross-section avoids the corner impact on elevated coastal structures")
    k = max([i for i, t in enumerate(times) if t <= tc] or [0])
    for j in range(3):
        ax = fig.add_axes([0.035 + j * 0.31, 0.47, 0.295, 0.3]); ax.imshow(snaps[k][j]); ax.axis("off")
        fig.text(0.035 + j * 0.31, 0.785, labels[j], fontsize=11.5, fontweight="semibold", color=cols[names[j]] if j else ORANGE)
    fig.text(0.965, 0.785, f"t = {times[k]:.2f} s", fontsize=11, color=MUTED, ha="right")
    ax = fig.add_axes([0.07, 0.165, 0.52, 0.235])
    for n in names:
        tt, y = S_[n]; m = tt <= tc
        ax.plot(tt, y, color=cols[n] if n != "ST90" else ORANGE, lw=1, alpha=0.15)
        ax.plot(tt[m], y[m], color=cols[n] if n != "ST90" else ORANGE, lw=1.6, label=n)
    for t in times: ax.axvline(t, color=LINE, lw=1, ls=":")
    ax.axvline(tc, color=ACCENT, lw=1.2)
    ax.set_xlim(8.2, 9.0); ax.set_ylim(-60, 520); ax.set_yticks([0, 250, 500])
    ax.set_xlabel("Time (s)", fontsize=10); ax.set_ylabel("Horizontal force (N/m)", fontsize=10)
    ax.legend(frameon=False, fontsize=9.5, ncol=3, loc="upper right")
    # peak bars
    ab = fig.add_axes([0.69, 0.165, 0.27, 0.2])
    cur = {n: (S_[n][1][S_[n][0] <= tc].max() if (S_[n][0] <= tc).any() else 0) for n in names}
    ab.barh([2, 1, 0], [cur[n] for n in names], color=[ORANGE, TEAL, BLUE], height=0.6)
    for yv, n in zip([2, 1, 0], names): ab.text(cur[n] + 8, yv, f"{cur[n]:.0f}", va="center", fontsize=10, color=INK)
    ab.set_yticks([2, 1, 0], names); ab.set_xlim(0, 560); ab.set_xticks([])
    ab.spines["bottom"].set_visible(False)
    fig.text(0.69, 0.39, "Peak horizontal force so far (N/m)", fontsize=11, fontweight="semibold", color=INK)
    footer(fig, "Elevated case, H = 0.275 m, H/λ = 0.05 · force traces digitized and pressure fields from Wu et al. (2025), Ocean Engineering")
    frames.append(grab(fig)); plt.close(fig)
save(frames, "wu2025sph", durations=90)
