import numpy as np
from ga_style import *
from PIL import Image
from matplotlib.patches import Polygon
R = np.load("ga/pawitan_series.npy", allow_pickle=True).item()
photos = {n: [np.asarray(Image.open(f"ga/pawitan/ins_{n}_{i}.png"))[8:-10, 8:-8] for i in range(4)] for n in R}
def arcpts(c, r, a0, a1, n=30):
    t = np.linspace(a0, a1, n); return np.c_[c[0] + r * np.cos(t), c[1] + r * np.sin(t)]
# sections (mm, 1:10 model): top 300 wide, bottom 140, depth 150
cc = np.vstack([[(-150, 150), (150, 150)], arcpts((150 + 0, 0), 80, np.pi / 2 + 0.55, np.pi, 1)[:0],
                [(150, 150)], arcpts((150, 0), 150, np.pi * 0.5, np.pi * 0.5, 1)[:0]])
# concave: circular arcs bowing inward from top corners to bottom corners
def concave(side):
    p0 = np.array([side * 150, 150]); p1 = np.array([side * 70, 0]); t = np.linspace(0, 1, 30)
    mid = np.array([side * 75, 125]); return (1 - t)[:, None] ** 2 * p0 + 2 * ((1 - t) * t)[:, None] * mid + t[:, None] ** 2 * p1
CC = np.vstack([[(-150, 150), (150, 150)], concave(1), concave(-1)[::-1]])
def convex(side):
    p0 = np.array([side * 150, 150]); p1 = np.array([side * 70, 0]); t = np.linspace(0, 1, 30)
    mid = np.array([side * 150, 20]); return (1 - t)[:, None] ** 2 * p0 + 2 * ((1 - t) * t)[:, None] * mid + t[:, None] ** 2 * p1
CX = np.vstack([[(-150, 150), (150, 150)], convex(1), convex(-1)[::-1]])
ST = np.array([(-150, 150), (150, 150), (150, 125), (70, 125), (70, 0), (-70, 0), (-70, 125), (-150, 125)])
shapes = {"CC60": (CC, "Concave"), "CX60": (CX, "Convex"), "ST90": (ST, "Box girder")}
cols = {"CC60": TEAL, "CX60": BLUE, "ST90": ORANGE}
t0, t1 = -0.1, 0.42; nF = 64; ts = list(np.linspace(t0, t1, nF)) + [t1] * 18
frames = []
for f, tc in enumerate(ts):
    fig = new_fig()
    header(fig, "1:10-scale wave-flume experiments", "Smooth, continuous shapes tame wave impacts on elevated coastal structures")
    for r, (n, (P, lab)) in enumerate(shapes.items()):
        y = 0.61 - r * 0.225; col = cols[n]; d = R[n]
        a0 = fig.add_axes([0.025, y + 0.02, 0.085, 0.15]); a0.add_patch(Polygon(P, closed=True, fc="#d5dbe2", ec=INK, lw=1.4))
        a0.set_xlim(-170, 170); a0.set_ylim(-20, 175); a0.set_aspect("equal"); a0.axis("off")
        fig.text(0.0675, y + 0.005, f"{n}\n{lab}", ha="center", va="top", fontsize=10, color=INK, linespacing=1.15, fontweight="semibold")
        ax = fig.add_axes([0.14, y, 0.33, 0.19])
        m = d["T"] <= tc
        ax.fill_between(d["T"], d["Flo"], d["Fhi"], color=LINE, lw=0)
        ax.fill_between(d["T"][m], d["Flo"][m], d["Fhi"][m], color=col, lw=0)
        ax.plot(d["T"][m], (d["Flo"][m] + d["Fhi"][m]) / 2, color=col, lw=1.4)
        for k, (et, ef) in enumerate(zip(d["eT"], d["eF"])):
            if et <= tc: ax.scatter([et], [ef], s=26, color=ACCENT, zorder=4); ax.text(et + 0.006, ef + 6, f"E{k + 1}", fontsize=8.5, color=ACCENT)
        ax.set_xlim(t0, t1); ax.set_ylim(-50, 300); ax.set_yticks([0, 150, 300]); ax.tick_params(labelsize=9)
        ax.axhline(0, color=LINE, lw=0.8)
        if r == 2: ax.set_xlabel("time / wave period", fontsize=10)
        else: ax.set_xticklabels([])
        k = int(np.sum(d["eT"] <= tc)) - 1
        ap = fig.add_axes([0.49, y - 0.005, 0.2, 0.2]); ap.axis("off")
        if k >= 0:
            ap.imshow(photos[n][k]); ap.text(0.03, 0.9, f"E{k + 1}", transform=ap.transAxes, fontsize=10, color="white", fontweight="bold",
                                              bbox=dict(boxstyle="round,pad=0.25", fc=ACCENT, ec="none"))
        else:
            ap.imshow(np.full_like(photos[n][0], 245)); ap.text(0.5, 0.5, "wave approaching…", transform=ap.transAxes, ha="center", va="center", fontsize=10, color=MUTED)
    fig.text(0.14, 0.815, "Measured uplift force (N/m)", fontsize=10.5, color=MUTED)
    fig.text(0.49, 0.815, "Flume photos at events E1–E4", fontsize=10.5, color=MUTED)
    fig.text(0.72, 0.79, "KEY FINDINGS", fontsize=10, color=MUTED, fontweight="bold")
    finds = [(BLUE, "Convex (CX60)", "lowest horizontal forces"), (TEAL, "Concave (CC60)", "lowest uplift forces"),
             (ORANGE, "Box girder (ST90)", "highest impulsive forces —\nup to 3× its quasi-static force,\ndriven by the overhang corner")]
    for (c, h, s), yy in zip(finds, [0.72, 0.6, 0.48]):
        fig.text(0.72, yy, h, fontsize=12, fontweight="bold", color=c); fig.text(0.72, yy - 0.025, s, fontsize=10.5, color=INK, va="top", linespacing=1.3)
    fig.text(0.72, 0.2, "Regular & irregular waves · structure\nsemi-submerged, at SWL, and elevated", fontsize=10, color=MUTED, linespacing=1.3)
    footer(fig, "Uplift force (elevated case ELE04) and photos digitized from Pawitan et al. (2026), Applied Ocean Research, Fig. 13")
    frames.append(grab(fig)); plt.close(fig)
save(frames, "pawitan2026experimental", durations=90)
