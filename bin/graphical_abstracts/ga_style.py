"""Shared look for all graphical abstracts: 1200x620 px, white, Inter, site purple accent."""
import io, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

plt.rcParams.update({
    "font.family": "Inter", "font.size": 13, "axes.edgecolor": "#c9d2dc",
    "axes.labelcolor": "#33414f", "xtick.color": "#5b6875", "ytick.color": "#5b6875",
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
})
INK = "#1f2d3d"; MUTED = "#5b6875"; ACCENT = "#b509ac"; ACCENT_SOFT = "#f6e6f5"
BLUE = "#2f6db5"; TEAL = "#1f9e89"; ORANGE = "#e07b39"; LINE = "#d9dee5"
W, H = 12.0, 6.2

def new_fig():
    fig = plt.figure(figsize=(W, H), facecolor="white")
    return fig

def header(fig, kicker, title):
    fig.text(0.035, 0.925, kicker.upper(), color=ACCENT, fontsize=11.5, fontweight="bold")
    fig.text(0.035, 0.865, title, color=INK, fontsize=19, fontweight="semibold")

def footer(fig, text):
    fig.text(0.035, 0.03, text, color=MUTED, fontsize=10.5, style="italic")

def pill(fig, x, y, text, color=ACCENT, fs=11.5):
    fig.text(x, y, text, color="white", fontsize=fs, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.8", fc=color, ec="none"))

def grab(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100, facecolor="white"); buf.seek(0)
    return Image.open(buf).convert("RGB")

def save(frames, name, durations=80, static_index=-1, out="ga/out"):
    import os; os.makedirs(out, exist_ok=True)
    frames[0].save(f"{out}/{name}.webp", save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, quality=70, method=6)
    frames[static_index].save(f"{out}/{name}.png", optimize=True)
    sz = os.path.getsize(f"{out}/{name}.webp") / 1e6
    print(name, len(frames), "frames", f"{sz:.2f} MB webp", f"{os.path.getsize(f'{out}/{name}.png')/1e6:.2f} MB png")
