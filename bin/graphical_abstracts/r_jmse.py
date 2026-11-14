import numpy as np
from ga_style import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource
L, Wd, D, dr = 10.0, 5.0, 5.0, 2.5   # length b, width w_BW, depth d_BW, draft (paper: b=10, d=w=5 m)
def face(y0, sign, Rn, n=10):
    """Two 5x5 hypar units across the length; each = 4 bilinear tympans sharing a centre vertex
    displaced inward by Re = Rn*sqrt(A), A = 25 m^2 (paper Sec. 2.4)."""
    Re = Rn * 5.0; quads = []
    for u0 in (0.0, 5.0):
        cx, cz = u0 + 2.5, -dr + 2.5
        for (xa, xb) in ((u0, cx), (cx, u0 + 5)):
            for (za, zb) in ((-dr, cz), (cz, -dr + 5)):
                s = np.linspace(0, 1, n + 1)
                X = xa + (xb - xa) * s[None, :] + 0 * s[:, None]; Z = za + (zb - za) * s[:, None] + 0 * s[None, :]
                # bilinear: displacement only at the corner that is the centre vertex
                wx = np.where(np.isclose(xb, cx), s[None, :], 1 - s[None, :]); wz = np.where(np.isclose(zb, cz), s[:, None], 1 - s[:, None])
                Y = y0 + sign * Re * wx * wz
                for i in range(n):
                    for j in range(n):
                        quads.append([(X[i, j], Y[i, j], Z[i, j]), (X[i, j + 1], Y[i, j + 1], Z[i, j + 1]),
                                      (X[i + 1, j + 1], Y[i + 1, j + 1], Z[i + 1, j + 1]), (X[i + 1, j], Y[i + 1, j], Z[i + 1, j])])
    return np.array(quads)
def shade(q, base):
    v1 = q[:, 1] - q[:, 0]; v2 = q[:, 3] - q[:, 0]; nrm = np.cross(v1, v2); nrm /= np.linalg.norm(nrm, axis=1)[:, None] + 1e-12
    lt = np.array([-0.5, -0.8, 0.6]); lt /= np.linalg.norm(lt); k = 0.55 + 0.45 * np.abs(nrm @ lt)
    c = np.array(matplotlib.colors.to_rgb(base)); return np.clip(c[None, :] * k[:, None], 0, 1)
tbl = {"kt": [0.22, 0.18, 0.12], "kr": [0.76, 0.60, 0.40], "kd": [0.61, 0.78, 0.91]}
Rns = [0.0, 0.25, 0.5]
# timeline: morph 0 -> .25 -> .5, hold each, then back
seq = []
for a, b in [(0, 0.25), (0.25, 0.5)]:
    seq += [a] * 10 + list(np.linspace(a, b, 14))
seq += [0.5] * 16 + list(np.linspace(0.5, 0, 16))
frames = []
for f, Rn in enumerate(seq):
    fig = new_fig()
    header(fig, "Hypar free-surface breakwaters · SPH + FEM", "Warping the face of a breakwater dissipates more wave energy")
    ax = fig.add_axes([-0.06, 0.02, 0.62, 0.86], projection="3d")
    sea = face(0.0, 1, Rn); land = face(Wd, -1, Rn)
    ax.add_collection3d(Poly3DCollection(sea, facecolors=shade(sea, "#c8ced6"), edgecolors=(1, 1, 1, 0.25), linewidths=0.3))
    ax.add_collection3d(Poly3DCollection(land, facecolors=shade(land, "#c8ced6"), edgecolors="none", alpha=0.6))
    top = [[(0, 0, D - dr), (L, 0, D - dr), (L, Wd, D - dr), (0, Wd, D - dr)]]
    ends = [[(L, 0, -dr), (L, Wd, -dr), (L, Wd, D - dr), (L, 0, D - dr)]]
    ax.add_collection3d(Poly3DCollection(ends, facecolors=["#d3d9df"], edgecolors=INK, linewidths=0.6))
    ax.add_collection3d(Poly3DCollection(top, facecolors="#e7ebef", edgecolors=INK, linewidths=0.8))
    # water surface with gentle waves (seaside)
    gx = np.linspace(-1, L + 1, 40); gy = np.linspace(-9, -0.05, 30); GX, GY = np.meshgrid(gx, gy)
    ph = 2 * np.pi * f / 24; GZ = 0.35 * np.cos(2 * np.pi * GY / 4.5 - ph) * np.clip((GY + 9) / 3, 0, 1)
    ax.plot_surface(GX, GY, GZ, color="#5d93cf", alpha=0.45, linewidth=0, shade=True)
    ax.set_xlim(-1, L + 1); ax.set_ylim(-9, Wd + 1); ax.set_zlim(-dr - 0.5, D - dr + 0.5)
    ax.set_box_aspect((L + 2, Wd + 10, D + 1), zoom=1.25); ax.view_init(20, -32); ax.set_axis_off()
    fig.text(0.04, 0.2, f"Rn = {Rn:.2f}", fontsize=22, fontweight="bold", color=ACCENT)
    fig.text(0.17, 0.215, "flat face" if Rn < 0.01 else "hypar-faced", fontsize=11.5, color=INK)
    fig.text(0.04, 0.12, "10 × 5 × 5 m box breakwater at 2.5 m draft; waves approach the seaside face", fontsize=10.5, color=MUTED)
    # Table 4 bars
    ab = fig.add_axes([0.6, 0.37, 0.36, 0.3]); w = 0.24; xs = np.arange(3)
    near = int(np.argmin([abs(Rn - r) for r in Rns])); exact = abs(Rn - Rns[near]) < 1e-6
    for j, (key, lab, c) in enumerate([("kt", r"transmission $k_t$", ACCENT), ("kr", r"reflection $k_r$", "#8a96a3"), ("kd", r"dissipation $k_d$", BLUE)]):
        vals = tbl[key]
        alphas = [1.0 if (exact and i == near) else 0.35 for i in range(3)]
        for i in range(3):
            ab.bar(xs[i] + (j - 1) * w, vals[i], w * 0.92, color=c, alpha=alphas[i], label=lab if i == 0 else None)
            if exact and i == near: ab.text(xs[i] + (j - 1) * w, vals[i] + 0.02, f"{vals[i]:.2f}", ha="center", fontsize=9, color=INK)
    ab.set_xticks(xs, [f"Rn = {r:.2f}" for r in Rns]); ab.set_ylim(0, 1.05); ab.set_yticks([0, 0.5, 1])
    ab.legend(frameon=False, fontsize=9.5, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.13))
    fig.text(0.6, 0.77, "SPH results, T = 4 s, draft 2.5 m (Table 4)", fontsize=11.5, fontweight="semibold", color=INK)
    for i, (big, small) in enumerate([("5×", "lower wave\ntransmission\n(T = 3 s, 2.5 m draft)"), ("3×", "lower principal\nstresses than\na flat face"),
                                      ("<6%", "static vs.\ndynamic stress\ndifference")]):
        x = 0.6 + i * 0.13
        fig.text(x, 0.235, big, fontsize=20, fontweight="bold", color=ACCENT)
        fig.text(x, 0.215, small, fontsize=9.5, color=INK, va="top", linespacing=1.25)
    footer(fig, "Geometry and results from Smith, Wu, Pawitan & Garlock (2025), J. Mar. Sci. Eng. 13(2), 245")
    frames.append(grab(fig)); plt.close(fig)
save(frames, "jmse13020245", durations=90, static_index=60)
