import numpy as np
from ga_style import *
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
d = np.load("ga/mm_hist.npz")
C = d["C"]; Z = d["Z"]; xs = d["xs"]; ys = d["ys"]; design = d["design"]; net = d["cnct_network"]
zbase = d["zs0"].copy()
nd = np.setdiff1d(np.arange(xs.size), design); zbase[nd] = 0.0  # supports on ground
def full(zd):
    z = zbase.copy(); z[design] = zd; return z
n_it = len(C)
# frame schedule: ease-out over iterations, then hold
its = np.unique(np.round(np.linspace(0, 1, 56) ** 1.15 * (n_it - 1)).astype(int))
sched = list(its) + [n_it - 1] * 16
zmax_all = max(Z.max(), 1)
steps = ["Geometry  z", "Differentiable FEA (JAX)", "Strain energy  C(z)", "∂C/∂z  via AD + adjoint", "Gradient step on z"]
frames = []
for f, k in enumerate(sched):
    fig = new_fig()
    header(fig, "Gradient-based shape optimization", "Form-finding a free-form gridshell with exact gradients")
    ax = fig.add_axes([-0.07, -0.08, 0.76, 1.0], projection="3d")
    z = full(Z[k])
    segs = np.stack([np.c_[xs[net[:, 0]], ys[net[:, 0]], z[net[:, 0]]], np.c_[xs[net[:, 1]], ys[net[:, 1]], z[net[:, 1]]]], 1)
    zc = (z[net[:, 0]] + z[net[:, 1]]) / 2
    cmap = plt.get_cmap("viridis")
    lc = Line3DCollection(segs, colors=cmap(np.clip(zc / zmax_all, 0, 1)), linewidths=1.3)
    ax.add_collection3d(lc)
    ax.scatter(xs[nd], ys[nd], 0 * xs[nd], s=6, color=INK)
    ax.set_xlim(xs.min(), xs.max()); ax.set_ylim(ys.min(), ys.max()); ax.set_zlim(0, zmax_all * 1.05)
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), zmax_all * 2.0), zoom=1.25); ax.view_init(26, -62)
    ax.set_axis_off()
    fig.text(0.05, 0.12, f"Mannheim Multihalle–inspired gridshell  ·  iteration {k + 1}  ·  vertical scale ×2", color=MUTED, fontsize=11)
    # history curve
    ax2 = fig.add_axes([0.66, 0.47, 0.3, 0.3])
    ax2.plot(np.arange(n_it), C / C[0], color=LINE, lw=2)
    ax2.plot(np.arange(k + 1), C[: k + 1] / C[0], color=ACCENT, lw=2.5)
    ax2.scatter([k], [C[k] / C[0]], color=ACCENT, s=40, zorder=3)
    ax2.set_yscale("log"); ax2.set_xlabel("Iteration"); ax2.set_ylabel("Strain energy  C / C₀")
    ax2.set_title(f"C reduced to {C[k] / C[0] * 100:.1f}% of initial", fontsize=12, color=INK, loc="left")
    # pipeline with active step cycling
    fig.text(0.67, 0.405, "EACH ITERATION", fontsize=10, color=MUTED, fontweight="bold")
    act = f % len(steps) if f < 56 else -1
    for i, s in enumerate(steps):
        y = 0.35 - i * 0.062
        on = i == act
        fig.text(0.67, y, s, fontsize=11.5, color=("white" if on else INK), va="center",
                 bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.6", fc=(ACCENT if on else ACCENT_SOFT), ec="none"))
    footer(fig, "Computed with JaxSSO (Wu, 2023): automatic differentiation · adjoint method · XLA")
    frames.append(grab(fig)); plt.close(fig)
save(frames, "wu2023framework", durations=90)
