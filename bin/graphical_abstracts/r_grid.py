import numpy as np
from ga_style import *
from mpl_toolkits.mplot3d.art3d import Line3DCollection
d = np.load("ga/grid_hist.npz")
xs, ys, edges, free = d["xs"], d["ys"], d["edges"], d["free"]
runs = [("Constrained force density method", d["fdm_Z"], d["fdm_C"], BLUE, "geometric stiffness · optimizes force densities"),
        ("FEA-based shape optimization", d["fso_Z"], d["fso_C"], ACCENT, "elastic stiffness · optimizes nodal heights")]
n = len(d["fso_C"]); zmax = max(d["fdm_Z"].max(), d["fso_Z"].max())
sched = list(range(0, n, 2)) + [n - 1] * 14
frames = []
for k in sched:
    fig = new_fig()
    header(fig, "Form-finding comparison", "Geometric-stiffness vs elastic-stiffness shape optimization of gridshells")
    for j, (name, Z, C, col, sub) in enumerate(runs):
        x0 = 0.02 + j * 0.49
        ax = fig.add_axes([x0 - 0.03, 0.16, 0.52, 0.66], projection="3d")
        z = np.zeros(xs.size); z[free] = Z[k]
        segs = np.stack([np.c_[xs[edges[:, 0]], ys[edges[:, 0]], z[edges[:, 0]]],
                         np.c_[xs[edges[:, 1]], ys[edges[:, 1]], z[edges[:, 1]]]], 1)
        ax.add_collection3d(Line3DCollection(segs, colors=col, linewidths=1.4, alpha=0.9))
        ax.plot([0, 20, 20, 0, 0], [0, 0, 20, 20, 0], [0] * 5, color=INK, lw=1.2)
        ax.set_xlim(0, 20); ax.set_ylim(0, 20); ax.set_zlim(0, zmax * 1.05)
        ax.set_box_aspect((20, 20, zmax * 2.2), zoom=0.95); ax.view_init(22, -55); ax.set_axis_off()
        fig.text(x0 + 0.235, 0.80, name, ha="center", fontsize=13, fontweight="semibold", color=col)
        fig.text(x0 + 0.235, 0.765, sub, ha="center", fontsize=10.5, color=MUTED)
        # sparkline of strain energy
        a2 = fig.add_axes([x0 + 0.1, 0.1, 0.28, 0.07])
        a2.plot(C, color=LINE, lw=1.6); a2.plot(np.arange(k + 1), C[: k + 1], color=col, lw=2.2)
        a2.scatter([k], [C[k]], color=col, s=22, zorder=3)
        a2.set_xticks([]); a2.set_yticks([]); a2.spines[:].set_visible(False)
        fig.text(x0 + 0.1, 0.185, "strain energy ↓", fontsize=10.5, color=MUTED)
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.15, 0.78], color=LINE, lw=1))
    fig.text(0.965, 0.03, f"iteration {k + 1}", ha="right", fontsize=10.5, color=MUTED)
    footer(fig, "Illustrative 20 m × 20 m gridshell (edges pinned, uniform load) computed with JaxSSO + automatic differentiation")
    frames.append(grab(fig)); plt.close(fig)
save(frames, "wu2023gridshells", durations=90)
