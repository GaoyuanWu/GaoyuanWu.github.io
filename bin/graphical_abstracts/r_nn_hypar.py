import numpy as np
from ga_style import *
from PIL import Image
from matplotlib.patches import FancyArrowPatch
L = lambda n: np.asarray(Image.open(f"ga/nn/{n}.png"))
sph = [L("sph_t0"), L("sph_pk")]
models = [("fnn", "FNN"), ("cnn", "CNN"), ("don", "DeepONet")]
dts = {"gt": 5.55, "fnn": 5.55, "cnn": 5.58, "don": 5.56}
per = 26; frames = []
for f in range(per * 3):
    mi = f // per; key, name = models[mi]
    fig = new_fig()
    header(fig, "Machine learning for coastal hydrodynamics", "Neural networks predict wave pressure on hypar breakwaters in milliseconds")
    fig.text(0.035, 0.78, "SPH simulations (training data)", fontsize=12, fontweight="semibold", color=INK)
    row = (f // 13) % 2
    for j, (img, lab) in enumerate(zip(sph, ["wave arrival  t₀", "peak loading  t_peak"])):
        ax = fig.add_axes([0.03, 0.5 - j * 0.25, 0.3, 0.22]); ax.imshow(img); ax.axis("off"); ax.set_alpha(1)
        if j != row: ax.imshow(np.full_like(img, 255), alpha=0.55)
        fig.text(0.035, 0.725 - j * 0.25, lab, fontsize=10.5, color=ACCENT if j == row else MUTED, fontweight="semibold" if j == row else "normal")
    fig.text(0.035, 0.17, "Inputs: hypar warping Rn · draft dr · wave height H", fontsize=10.5, color=INK)
    fig.add_artist(FancyArrowPatch((0.335, 0.5), (0.385, 0.5), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18, color=INK, lw=1.6))
    for i, (_, nm) in enumerate(models):
        on = i == mi
        fig.text(0.44, 0.62 - i * 0.09, nm, fontsize=11.5, ha="center", va="center", color="white" if on else INK, fontweight="semibold",
                 bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.6", fc=ACCENT if on else ACCENT_SOFT, ec="none"))
    fig.text(0.44, 0.73, "ANN surrogate", fontsize=12, fontweight="semibold", color=INK, ha="center")
    fig.add_artist(FancyArrowPatch((0.495, 0.5), (0.53, 0.5), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18, color=INK, lw=1.6))
    heads = ["SPH (ground truth)", f"{name} prediction", "|error|"]
    for c, h in enumerate(heads):
        fig.text(0.6 + c * 0.13, 0.78, h, fontsize=10.5, ha="center", fontweight="semibold", color=ACCENT if c == 1 else INK)
    for r, (tag, gtk, pk, ek) in enumerate([("P(t₀)", "gt_t0", f"{key}_t0", f"{key}_e0"), ("P(t_peak)", "gt_pk", f"{key}_pk", f"{key}_epk")]):
        for c, k in enumerate([gtk, pk, ek]):
            ax = fig.add_axes([0.545 + c * 0.13, 0.5 - r * 0.25, 0.11, 0.22]); ax.imshow(L(k)); ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values(): s_.set_visible(True); s_.set_color(ACCENT if r == row else LINE); s_.set_linewidth(2.2 if r == row else 1)
        fig.text(0.54, 0.61 - r * 0.25, tag, fontsize=10.5, color=MUTED, ha="right", rotation=90, va="center")
    ac = fig.add_axes([0.545, 0.175, 0.24, 0.05]); ac.imshow(L("cb_p")); ac.axis("off")
    ae = fig.add_axes([0.805, 0.175, 0.12, 0.05]); ae.imshow(L("cb_e")); ae.axis("off")
    fig.text(0.6, 0.125, f"rise time Δt:  SPH {dts['gt']:.2f} s   ·   {name} {dts[key]:.2f} s", fontsize=11, color=INK)
    fig.text(0.6, 0.085, "milliseconds per prediction: orders of magnitude faster than SPH", fontsize=10, color=ACCENT, fontweight="semibold")
    footer(fig, "Representative training sample, ~50k-parameter models · figures from Smith, Wu & Garlock (2025), J. Mar. Sci. Eng. 13(12), 2277")
    frames.append(grab(fig)); plt.close(fig)
save(frames, "smith2025neural", durations=90, static_index=per * 2 + 5)
