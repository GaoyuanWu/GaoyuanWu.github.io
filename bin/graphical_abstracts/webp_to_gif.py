"""Convert the animated graphical abstracts (.webp) into .gif copies.

Usage (from the site root, needs Pillow and NumPy):

    python bin/graphical_abstracts/webp_to_gif.py                      # all *.webp in the folder
    python bin/graphical_abstracts/webp_to_gif.py wu2024jaxsso          # selected ids
    python bin/graphical_abstracts/webp_to_gif.py --width 800 wu2023framework   # force a downscale
    python bin/graphical_abstracts/webp_to_gif.py --step 2 wu2023framework      # keep every 2nd frame

All frames share one adaptive palette (160 median-cut + 94 max-coverage colours from the non-white
pixels of every frame, plus exact white and black) and are Floyd-Steinberg dithered against it; a
shared palette keeps the inter-frame deltas small, so the GIF stays far smaller than with a
per-frame palette. Per-frame durations and loop=0 are kept.
If the full-size (1200 px) GIF is larger than --max-mb (default 3.5), it is redone at --fallback-width
(default 800 px). With --step N the dropped frames' durations are added to the kept frame, so the
total running time is unchanged.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

WHITE, BLACK = 254, 255  # palette indices pinned in shared_palette()

DIR = Path(__file__).resolve().parents[2] / "assets/img/graphical_abstracts"


def read_frames(path, step=1):
    im = Image.open(path)
    frames, durations = [], []
    for i in range(im.n_frames):
        im.seek(i)
        im.load()
        dur = im.info.get("duration", 100)
        if i % step == 0:
            frames.append(im.convert("RGB"))
            durations.append(dur)
        else:
            durations[-1] += dur
    return frames, durations


def shared_palette(frames, every=7):
    # Pool every 7th pixel of every frame, leaving out pure white: the white page background would
    # otherwise dominate and starve rare colours (titles, labels, curves) of palette entries.
    px = np.concatenate([np.asarray(f).reshape(-1, 3)[i % every :: every] for i, f in enumerate(frames)])
    px = px[~(px == 255).all(1)]
    strip = Image.fromarray(np.ascontiguousarray(px).reshape(1, -1, 3))
    # 160 median-cut colours cover the bulk; 94 max-coverage colours catch the outliers.
    bulk = strip.quantize(colors=160, method=Image.Quantize.MEDIANCUT).getpalette()[: 160 * 3]
    rare = strip.quantize(colors=94, method=Image.Quantize.MAXCOVERAGE).getpalette()[: 94 * 3]
    colours = bulk + [0] * (160 * 3 - len(bulk)) + rare + [0] * (94 * 3 - len(rare))
    # Pin exact white and black (indices WHITE, BLACK) so backgrounds and text stay clean.
    pal = Image.new("P", (1, 1))
    pal.putpalette(colours + [255, 255, 255, 0, 0, 0])
    return pal


def quantize(frame, pal):
    q = frame.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG)
    # Pillow's palette lookup goes through a reduced-precision colour cache, so pure white can land on
    # a near-white entry; force exact white/black source pixels onto the pinned entries.
    src, idx = np.asarray(frame), np.array(q)
    idx[(src == 255).all(-1)] = WHITE
    idx[(src == 0).all(-1)] = BLACK
    out = Image.fromarray(idx, "P")
    out.putpalette(pal.getpalette())
    return out


def write_gif(frames, durations, out, width=None):
    if width and frames[0].width > width:
        h = round(frames[0].height * width / frames[0].width)
        frames = [f.resize((width, h), Image.LANCZOS) for f in frames]
    pal = shared_palette(frames)
    q = [quantize(f, pal) for f in frames]
    q[0].save(out, save_all=True, append_images=q[1:], duration=durations, loop=0)


def convert(path, width=None, step=1, max_mb=3.5, fallback_width=800):
    frames, durations = read_frames(path, step)
    out = path.with_suffix(".gif")
    write_gif(frames, durations, out, width)
    if not width and out.stat().st_size > max_mb * 1e6:
        write_gif(frames, durations, out, fallback_width)
    return out


def describe(path):
    im = Image.open(path)
    durs = []
    for i in range(im.n_frames):
        im.seek(i)
        im.load()
        durs.append(im.info.get("duration", 0))
    return im.size, im.n_frames, sum(durs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", help="paper ids (default: every .webp in the folder)")
    ap.add_argument("--width", type=int, help="downscale to this width (keeps aspect ratio)")
    ap.add_argument("--step", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--max-mb", type=float, default=3.5, help="size above which to downscale (MB)")
    ap.add_argument("--fallback-width", type=int, default=800, help="width used when over --max-mb")
    args = ap.parse_args()
    paths = [DIR / f"{i}.webp" for i in args.ids] or sorted(DIR.glob("*.webp"))
    for p in paths:
        out = convert(p, args.width, args.step, args.max_mb, args.fallback_width)
        (w, h), n, total = describe(out)
        _, wn, wtotal = describe(p)
        print(f"{out.name}: {w}x{h}, {n} frames ({wn} in webp), {total} ms ({wtotal} in webp), {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
