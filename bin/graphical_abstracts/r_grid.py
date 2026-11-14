import numpy as np
from ga_style import *
from PIL import Image
# Table 1 of Wu, Pastrana, Adriaenssens & Garlock (IASS 2023), constant cross-section regime
cases = [("Barrel vault", "barrel", dict(se=(34.91, 34.05), h=(6.52, 8.25), t=(36, 8.5))),
         ("Circular dome", "dome", dict(se=(58.03, 42.03), h=(7.99, 11.97), t=(25.85, 65.76))),
         ("Free-form gridshell", "free", dict(se=(46.58, 33.57), h=(19.54, 28.47), t=(666.28, 102.63)))]
imgs = {k: [np.asarray(Image.open(f"ga/iass/{k}_{i}.png")) for i in range(3)] for _, k, _ in cases}
per = 34; frames = []
for f in range(per * 3):
    ci = f // per; s = min(1, (f % per) / 14); s = 1 - (1 - s) ** 3
    name, key, m = cases[ci]
    fig = new_fig()
    header(fig, "Form-finding comparison · gridshells", "Force density method vs FEA-based shape optimization for minimum strain energy")
    for i, (cap, col) in enumerate([("Initial", MUTED), ("FDM (geometric stiffness)", BLUE), ("FSO (elastic stiffness)", ACCENT)]):
        ax = fig.add_axes([0.03 + i * 0.205, 0.3, 0.195, 0.45]); ax.imshow(imgs[key][i]); ax.axis("off")
        fig.text(0.03 + i * 0.205, 0.77, cap, fontsize=11, fontweight="semibold", color=col)
    for i, (nm, _, _) in enumerate(cases):
        on = i == ci
        fig.text(0.035 + i * 0.13, 0.2, nm, fontsize=10.5, va="center", color="white" if on else INK,
                 bbox=dict(boxstyle="round,pad=0.4,rounding_size=0.6", fc=ACCENT if on else ACCENT_SOFT, ec="none"))
    fig.text(0.035, 0.13, "Optimized for minimal strain energy with fixed plan view · constant cross-sections", fontsize=10, color=MUTED)
    # metrics
    mets = [("Strain energy (% of initial)", m["se"], "{:.1f}%"), ("Height (m)", m["h"], "{:.2f}"), ("Optimization time (s)", m["t"], "{:.0f}")]
    for r, (lab, (a, b), fmt) in enumerate(mets):
        y0 = 0.62 - r * 0.2
        fig.text(0.665, y0 + 0.11, lab, fontsize=11, fontweight="semibold", color=INK)
        ab = fig.add_axes([0.665, y0, 0.3, 0.1]); mx = max(a, b) * 1.25
        ab.barh([1, 0], [a * s, b * s], color=[BLUE, ACCENT], height=0.62)
        for yv, v in zip([1, 0], [a, b]): ab.text(v * s + mx * 0.02, yv, fmt.format(v), va="center", fontsize=10, color=INK)
        ab.set_xlim(0, mx); ab.set_yticks([1, 0], ["FDM", "FSO"]); ab.set_xticks([]); ab.spines["bottom"].set_visible(False)
        ab.tick_params(axis="y", labelsize=10)
    footer(fig, "Shapes and Table 1 from Wu, Pastrana, Adriaenssens & Garlock (2023), Proc. IASS Annual Symposium")
    frames.append(grab(fig)); plt.close(fig)
save(frames, "wu2023gridshells", durations=90, static_index=per * 3 - 1)
