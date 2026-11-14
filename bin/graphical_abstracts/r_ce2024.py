import numpy as np
from ga_style import *
from PIL import Image
gif = Image.open("/home/user/GaoyuanWu.github.io/assets/img/publication_preview/cait.gif")
src = []
for i in range(gif.n_frames):
    gif.seek(i); src.append(np.asarray(gif.convert("RGB")))
g0 = [np.asarray(Image.open(f"ga/ce/g0_{i}.png")) for i in range(4)]
g4 = [np.asarray(Image.open(f"ga/ce/g4_{i}.png")) for i in range(4)]
geos = np.asarray(Image.open("ga/ce/geos.png"))
finds = ["Elevated bridges: the sharper the web–flange\nangle, the larger the maximum wave force",
         "The effect grows with taller waves\nand shallower water",
         "Parapets without openings raise the risk;\nrubber bearings help against horizontal force"]
n = len(src); frames = []
for f in range(n * 3):
    k = min(3, (f * 4) // (n * 3))
    fig = new_fig()
    header(fig, "Coastal bridges · solitary waves · SPH", "How box-girder geometry changes tsunami-like wave forces")
    ax = fig.add_axes([0.03, 0.36, 0.45, 0.42]); ax.imshow(src[f % n]); ax.axis("off")
    fig.text(0.035, 0.79, "SPH: solitary wave striking a box girder (Geo-4)", fontsize=12, fontweight="semibold", color=INK)
    fig.text(0.035, 0.33, "Five geometries studied, from smooth web–flange transition (Geo-0)\nto a sharp 90° corner (Geo-4); deck elevation, wave height and depth varied", fontsize=10, color=MUTED, va="top", linespacing=1.3)
    fig.text(0.53, 0.79, "Pressure field over time (elevated girder)", fontsize=12, fontweight="semibold", color=INK)
    fig.text(0.53, 0.755, "colour: pressure, 0 (blue) to 10 kPa (red)", fontsize=10, color=MUTED)
    fig.text(0.965, 0.79, f"step {k + 1}/4", fontsize=10.5, color=MUTED, ha="right")
    for j, (img, lab, c) in enumerate([(g0[k], "Geo-0 (smooth)", TEAL), (g4[k], "Geo-4 (sharp corner)", ORANGE)]):
        a = fig.add_axes([0.53 + j * 0.225, 0.5, 0.215, 0.24]); a.imshow(img); a.axis("off")
        fig.text(0.53 + j * 0.225, 0.49, lab, fontsize=10.5, fontweight="semibold", color=c, va="top")
    for i, t in enumerate(finds):
        fig.text(0.53, 0.39 - i * 0.095, "•  " + t.replace("\n", "\n    "), fontsize=10.5, color=INK, va="top", linespacing=1.3)
    footer(fig, "SPH animation, geometries and pressure fields from Wu & Garlock (2024), Coastal Engineering")
    frames.append(grab(fig)); plt.close(fig)
save(frames, "wu2024coastal", durations=110, static_index=n * 3 - 2)
