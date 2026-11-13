import numpy as np
from ga_style import *
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
water = LinearSegmentedColormap.from_list("w", ["#9cc9ee", "#2f6db5", "#173b65"])
def arc(p0, p1, bulge, n=24):
    p0, p1 = np.array(p0, float), np.array(p1, float); t = np.linspace(0, 1, n)
    mid = (p0 + p1) / 2; nrm = np.array([-(p1 - p0)[1], (p1 - p0)[0]]); nrm /= np.linalg.norm(nrm)
    return (1 - t)[:, None] ** 2 * p0 + 2 * ((1 - t) * t)[:, None] * (mid + bulge * nrm) + t[:, None] ** 2 * p1
fl = 0.22  # deck flange thickness
def deck():
    return [(-1.5, 1.5), (1.5, 1.5), (1.5, 1.5 - fl)]
shapes = {
    "Base": np.array(deck() + [(1.0, 1.5 - fl), (0.7, 0.0), (-0.7, 0.0), (-1.0, 1.5 - fl), (-1.5, 1.5 - fl)]),
    "Straight web": np.array(deck() + [(0.7, 1.5 - fl), (0.7, 0.0), (-0.7, 0.0), (-0.7, 1.5 - fl), (-1.5, 1.5 - fl)]),
    "Concave": np.vstack([[(-1.5, 1.5), (1.5, 1.5)], arc((1.5, 1.5 - fl), (0.7, 0.0), -0.45), arc((-0.7, 0.0), (-1.5, 1.5 - fl), -0.45)]),
    "Convex": np.vstack([[(-1.5, 1.5), (1.5, 1.5)], arc((1.5, 1.5), (-1.5, 1.5), -1.25)[1:-1][::-1][::-1]]),
}
# convex: bowl under the deck
t = np.linspace(0, np.pi, 40); shapes["Convex"] = np.vstack([[(-1.5, 1.5), (1.5, 1.5)], np.c_[1.5 * np.cos(t), 1.5 - 1.5 * np.sin(t)][::-1][::-1]])
shapes["Convex"] = np.vstack([np.c_[1.5 * np.cos(t), 1.5 - 1.5 * np.sin(t) * 0.95]])
X = np.linspace(-9, 9, 400); swl = 0.35; Hw = 0.9; Lw = 7.5
nF = 36; frames = []
for f in range(nF):
    ph = 2 * np.pi * f / nF
    fig = new_fig()
    header(fig, "Wave–structure interaction · SPH", "How cross-sectional shape changes regular-wave forces on coastal structures")
    for i, (name, P) in enumerate(shapes.items()):
        r, c = divmod(i, 2)
        ax = fig.add_axes([0.035 + c * 0.475, 0.47 - r * 0.32, 0.45, 0.3])
        eta = swl + Hw / 2 * np.cos(2 * np.pi * X / Lw - ph)
        bottom = -2.2
        ax.fill_between(X, bottom, eta, color="#2f6db5", alpha=0.85, lw=0)
        ax.fill_between(X, bottom, eta - 0.9, color="#173b65", alpha=0.35, lw=0)
        ax.plot(X, eta, color="#173b65", lw=1.2)
        ax.add_patch(Polygon(P, closed=True, fc="#d5dbe2", ec=INK, lw=1.5, zorder=3))
        ax.axhline(swl, color="white", lw=0.8, ls=(0, (4, 3)), alpha=0.7, zorder=2)
        ax.annotate("", (-4.4, 1.95), (-6.2, 1.95), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
        ax.set_xlim(-6.5, 6.5); ax.set_ylim(bottom, 2.4); ax.set_aspect("equal"); ax.axis("off")
        ax.text(6.3, 1.95, name, ha="right", fontsize=12, fontweight="semibold", color=ACCENT if i else INK)
    fig.text(0.035, 0.095, "Regular waves  ·  same deck width and height · only the cross-section shape changes · forces from SPH simulations",
             fontsize=11, color=MUTED)
    footer(fig, "Cross-sections redrawn from Wu, Smith, Pawitan & Garlock (2025), Ocean Engineering · wave motion schematic")
    frames.append(grab(fig)); plt.close(fig)
save(frames, "wu2025sph", durations=70, static_index=9)
