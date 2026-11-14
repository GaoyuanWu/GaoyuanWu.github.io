import numpy as np
from ga_style import *
from PIL import Image
from matplotlib.patches import FancyArrowPatch
gif = Image.open("/home/user/GaoyuanWu.github.io/assets/img/publication_preview/Side_Hypar.gif")
src = []
for i in range(gif.n_frames):
    gif.seek(i); src.append(np.asarray(gif.convert("RGB"))[37:197])
maps = [np.asarray(Image.open(f"ga/es/map_{i}.png")) for i in range(3)]
umb = np.asarray(Image.open("ga/es/umbrella.png"))
labs = ["H = 1 m (non-breaking)", "H = 2 m (non-breaking)", "H = 3 m (breaking)"]
finds = ["Wave pressure peaks near the mean water level\nand along the hypar's longitudinal spine",
         "Breaking waves intensify the spine pressure,\nmore so as hypar warping (r/A) grows",
         "Static analysis underestimates shell demands:\ndynamic SPH–FEM analysis is needed"]
frames = []
n = len(src)
for f in range(n):
    k = min(2, f * 3 // n)
    fig = new_fig()
    header(fig, "Hypar thin-shell coastal armor · SPH → FEM", "From wave pressure to structural demand on kinetic-umbrella seawalls")
    ax = fig.add_axes([0.03, 0.36, 0.45, 0.42]); ax.imshow(src[f]); ax.axis("off")
    fig.text(0.035, 0.79, "SPH: wave striking a hypar thin shell", fontsize=12, fontweight="semibold", color=INK)
    fig.text(0.035, 0.29, "Kinetic Umbrellas: deployable hypar thin shells for coastal defense", fontsize=10.5, color=MUTED)
    for i, s in enumerate(["SPH wave pressure", "mapped to shell elements", "FEM: shell & column demands"]):
        fig.text(0.17 + i * 0.105 if i else 0.17, 0.13 if i == 0 else 0.13, "", fontsize=1)
    fig.text(0.035, 0.22, "SPH pressure  →  mapped to shells  →  FEM demands", fontsize=10.5, color=ACCENT, fontweight="semibold")
    fig.text(0.53, 0.79, "Normalized wave pressure on the shell (SPH)", fontsize=12, fontweight="semibold", color=INK)
    fig.text(0.53, 0.755, f"{labs[k]} · flat → increasingly warped hypar", fontsize=10.5, color=ACCENT if k == 2 else MUTED)
    am = fig.add_axes([0.52, 0.4, 0.46, 0.34]); am.imshow(maps[k]); am.axis("off")
    for i, t in enumerate(finds):
        fig.text(0.53, 0.36 - i * 0.095, "•  " + t.replace("\n", "\n    "), fontsize=10.5, color=INK, va="top", linespacing=1.3)
    footer(fig, "SPH animation and pressure maps from Wu, Garlock & Wang (2022), Engineering Structures")
    frames.append(grab(fig)); plt.close(fig)
save(frames, "wu2022decoupled", durations=100, static_index=int(n * 0.85))
