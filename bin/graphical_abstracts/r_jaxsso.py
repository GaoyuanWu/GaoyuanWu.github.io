import numpy as np
from ga_style import *
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
d = np.load("ga/nn_hist.npz")
xs, ys, cnct, sup, design, C, Z = d["xs"], d["ys"], d["cnct"], d["sup"], d["design"], d["C"], d["Z"]
n = len(C); zmax = Z.max()
sched = list(range(0, 40, 1)) + list(range(40, n, 6)) + [n - 1] * 14
cmap = plt.get_cmap("plasma")
def nn_diagram(ax, phase):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    layers = [3, 5, 5, 5, 1]; xsl = np.linspace(0.08, 0.92, len(layers)); pos = []
    for x, nl in zip(xsl, layers):
        yy = np.linspace(0.5 - 0.09 * (nl - 1), 0.5 + 0.09 * (nl - 1), nl); pos.append([(x, y) for y in yy])
    for a, b in zip(pos[:-1], pos[1:]):
        for p in a:
            for q in b: ax.plot([p[0], q[0]], [p[1], q[1]], color=LINE, lw=0.8, zorder=1)
    # forward pulse (ink) then backward pulse (accent)
    fwd = phase < 0.5; s = (phase * 2) % 1.0; li = s * (len(layers) - 1)
    for L, pts in enumerate(pos):
        lit = (fwd and abs(L - li) < 0.6) or ((not fwd) and abs((len(layers) - 1 - L) - li) < 0.6)
        for p in pts:
            ax.scatter(*p, s=120, color=(INK if fwd else ACCENT) if lit else "white", edgecolor=INK if fwd else ACCENT, lw=1.4, zorder=3)
    for (x, y), t in zip(pos[0], ["x", "y", "d"]): ax.text(x - 0.06, y, t, ha="right", va="center", fontsize=12, color=INK, style="italic")
    ax.text(pos[-1][0][0] + 0.06, 0.5, "z", va="center", fontsize=12, color=INK, style="italic")
    ax.text(0.5, 0.06, "forward: shape" if fwd else "backward: ∂C/∂θ", ha="center", fontsize=11.5,
            color=INK if fwd else ACCENT, fontweight="semibold")
frames = []
for f, k in enumerate(sched):
    fig = new_fig()
    header(fig, "Differentiable physics × neural networks", "Training a neural network with FEA in the loop")
    a1 = fig.add_axes([0.03, 0.2, 0.3, 0.55]); nn_diagram(a1, (f % 12) / 12)
    fig.text(0.18, 0.77, "Neural network  θ", ha="center", fontsize=12.5, fontweight="semibold", color=INK)
    fig.text(0.18, 0.15, "node features → shell height", ha="center", fontsize=10.5, color=MUTED)
    # arrows between panels
    fig.add_artist(FancyArrowPatch((0.335, 0.52), (0.39, 0.52), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=16, color=INK, lw=1.5))
    fig.add_artist(FancyArrowPatch((0.39, 0.42), (0.335, 0.42), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=16, color=ACCENT, lw=1.5, linestyle="--"))
    # shell
    ax = fig.add_axes([0.33, 0.1, 0.42, 0.72], projection="3d")
    z = np.zeros(xs.size); z[design] = Z[k]
    polys = np.stack([np.c_[xs[c], ys[c], z[c]] for c in cnct])
    zc = z[cnct].mean(1)
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cmap(0.15 + 0.75 * zc / zmax), edgecolors=(1, 1, 1, 0.35), linewidths=0.4))
    ax.scatter(xs[sup], ys[sup], 0 * xs[sup] - 0.05, marker="^", s=40, color=INK, depthshade=False)
    ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.set_zlim(0, zmax); ax.set_box_aspect((6, 6, zmax), zoom=1.0)
    ax.view_init(24, -50); ax.set_axis_off()
    fig.text(0.545, 0.80, "Differentiable FEA (JAX-SSO)", ha="center", fontsize=12.5, fontweight="semibold", color=INK)
    fig.text(0.545, 0.15, "MITC4 shell · corner supports", ha="center", fontsize=10.5, color=MUTED)
    # loss
    a3 = fig.add_axes([0.79, 0.44, 0.18, 0.29])
    a3.plot(C / C[0], color=LINE, lw=2); a3.plot(np.arange(k + 1), C[: k + 1] / C[0], color=ACCENT, lw=2.5)
    a3.scatter([k], [C[k] / C[0]], color=ACCENT, s=36, zorder=3)
    a3.set_xlabel("Training step"); a3.set_ylim(0, 1.05); a3.set_yticks([0, 0.5, 1])
    fig.text(0.79, 0.77, "Loss = strain energy", fontsize=12.5, fontweight="semibold", color=INK)
    fig.text(0.79, 0.2, f"{(1 - C[k] / C[0]) * 100:.0f}%", fontsize=30, fontweight="bold", color=ACCENT)
    fig.text(0.79, 0.15, "lower strain energy", fontsize=11, color=MUTED)
    footer(fig, "Computed with JaxSSO: gradients flow from the FEA solver back to the network weights via automatic differentiation")
    frames.append(grab(fig)); plt.close(fig)
save(frames, "wu2024jaxsso", durations=90)
