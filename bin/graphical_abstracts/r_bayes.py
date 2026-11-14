import numpy as np
from ga_style import *
from PIL import Image
d = np.load("ga/bo_samples.npz"); P = d["p"]          # digitized Fig. 15c: force (kN/m), kt, iteration
eta = np.load("ga/bo_eta.npy", allow_pickle=True).item()  # digitized Fig. 17
rect = np.asarray(Image.open("ga/sph_rect.png")); opt = np.asarray(Image.open("ga/sph_opt.png"))
vir = plt.get_cmap("viridis")
def front(Q):
    keep = [i for i, p in enumerate(Q) if not any((q[:2] <= p[:2]).all() and (q[:2] < p[:2]).any() for q in Q)]
    return Q[sorted(keep, key=lambda i: Q[i, 0])]
nF = 70; hold = 16
frames = []
for f in range(nF + hold):
    s = min(f, nF - 1) / (nF - 1); it = s * 100; tcut = 15 + 25 * s
    fig = new_fig()
    header(fig, "Multi-objective Bayesian optimization × SPH", "Shaping box-type structures to cut wave loads and calm the water behind")
    ax = fig.add_axes([0.07, 0.17, 0.36, 0.58])
    Q = P[P[:, 2] <= it]
    ax.scatter(Q[:, 0], Q[:, 1], c=vir(Q[:, 2] / 100), s=26, alpha=0.9, lw=0, zorder=2)
    if len(Q) > 1:
        pf = front(Q); ax.plot(pf[:, 0], pf[:, 1], color=ACCENT, lw=2.4, zorder=3)
    ax.scatter([299.8], [0.61], marker="s", s=70, color=INK, zorder=4)
    ax.annotate("rectangular box", (299.8, 0.61), (300, 0.8), fontsize=10, color=INK, ha="center", arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.set_xlim(0, 400); ax.set_ylim(0, 1.0); ax.set_xticks([0, 100, 200, 300, 400]); ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("Total resultant wave force (kN/m)", fontsize=11); ax.set_ylabel(r"Wave transmission $k_t$", fontsize=11)
    fig.text(0.07, 0.78, "Design samples (qNEHVI) and Pareto front", fontsize=12, fontweight="semibold", color=INK)
    ax.text(395, 0.96, f"sample {int(round(it))}/100", fontsize=10.5, color=MUTED, ha="right", va="top")
    # free-surface elevation behind the structure
    a2 = fig.add_axes([0.53, 0.5, 0.43, 0.25])
    for k, (col, ls, lab) in {"grey": ("#9aa4ae", "-", "No structure"), "red": ("#d1495b", "--", "Rectangular box"),
                              "blue": (ACCENT, "-", "Optimized (Design-3)")}.items():
        T, E = eta[k][:, 0], eta[k][:, 1]; m = T <= tcut
        a2.plot(T, E, color=col, ls=ls, lw=1.2, alpha=0.18)
        a2.plot(T[m], E[m], color=col, ls=ls, lw=2.0 if k == "blue" else 1.6, label=lab)
    a2.set_xlim(15, 40); a2.set_ylim(8.9, 11.2); a2.set_yticks([9, 10, 11]); a2.set_xlabel("Time (s)", fontsize=10); a2.set_ylabel("η (m)", fontsize=10)
    a2.legend(frameon=False, fontsize=9, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.2))
    fig.text(0.53, 0.8, "Water level behind the structure  (H = 1.8 m, T = 6 s)", fontsize=12, fontweight="semibold", color=INK)
    for j, (img, lab, c) in enumerate([(rect, "Rectangular box", MUTED), (opt, "Optimized shape", ACCENT)]):
        ai = fig.add_axes([0.53 + j * 0.225, 0.2, 0.205, 0.18]); ai.imshow(img); ai.axis("off")
        fig.text(0.53 + j * 0.225, 0.385, lab, fontsize=10.5, fontweight="semibold", color=c)
    for i, (big, small) in enumerate([("60%", "better wave attenuation (single objective)"), ("73.1%", "time saved with parallel SPH")]):
        x = 0.53 + i * 0.225
        fig.text(x, 0.11, big, fontsize=18, fontweight="bold", color=ACCENT); fig.text(x, 0.085, small, fontsize=9, color=INK, va="top")
    footer(fig, "Samples and water levels digitized from Wu, Smith & Garlock (2026), Engineering Structures, Figs. 15c, 17 & 19")
    frames.append(grab(fig)); plt.close(fig)
save(frames, "wu2026bayes", durations=90)
