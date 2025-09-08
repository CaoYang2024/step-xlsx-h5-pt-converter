#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import sys
from pathlib import Path
from typing import List
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import numpy as np


def natural_key(s: str):
    """Natural sort: split into numeric and non-numeric parts; numbers sorted as integers, others as lowercase strings."""
    parts = re.split(r"(\d+)", s)
    key = []
    for t in parts:
        if t.isdigit():
            key.append(int(t))
        else:
            key.append(t.lower())
    return key


def extract_timeline_text(p: Path) -> str:
    """Extract the first numeric segment from the filename as annotation; if no number, use the filename stem."""
    m = re.search(r"(\d+)", p.stem)
    return m.group(1) if m else p.stem


def load_font(size: int = 60) -> ImageFont.FreeTypeFont:
    """Try common system fonts first; if all fail, fall back to Pillow’s default font."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Unicode.ttf",                      # macOS
        "C:/Windows/Fonts/arial.ttf",                            # Windows
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    # Fallback
    return ImageFont.load_default()


def collect_inputs(inputs: List[str]) -> List[Path]:
    """
    Collect PNG files from inputs:
      - If inputs contains a single directory: pick files matching `Tiefgezogenes Bauteil_<number>.png`,
        sort by <number> in descending order.
      - If inputs is a list of files: filter out files matching the same naming pattern and sort descending.
    """
    pattern = re.compile(r"^Tiefgezogenes Bauteil_(\d+)\.png$")

    def key_num(p: Path) -> int:
        m = pattern.match(p.name)
        return int(m.group(1)) if m else -1

    if len(inputs) == 1 and Path(inputs[0]).is_dir():
        d = Path(inputs[0])
        candidates = [p for p in d.iterdir() if p.is_file() and pattern.match(p.name)]
        candidates.sort(key=key_num, reverse=True)
        return candidates
    else:
        files = [Path(x) for x in inputs]
        pngs = [p for p in files if p.exists() and pattern.match(p.name)]
        pngs.sort(key=key_num, reverse=True)
        return pngs


def ensure_non_empty(pngs: List[Path], where: str):
    if not pngs:
        print(f"[Error] No PNG inputs found: {where}", file=sys.stderr)
        sys.exit(1)


def make_gif(inputs: List[str], out_gif: str, fps: int = 5, loop: int = 0, annotate: bool = True):
    # Collect input PNGs
    pngs = collect_inputs(inputs)
    where = inputs[0] if len(inputs) == 1 else f"{len(inputs)} files"
    ensure_non_empty(pngs, where)

    # Print the order
    print("[Info] PNG order:")
    for p in pngs:
        print("   ", p.name)

    # Unify canvas size (use max width/height across images)
    sizes = []
    valid_pngs = []
    for p in pngs:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
                valid_pngs.append(p)
        except (UnidentifiedImageError, OSError) as e:
            print(f"[Warn] Could not read image, skipped: {p} ({e})", file=sys.stderr)

    ensure_non_empty(valid_pngs, "valid PNG list (some files corrupted or unreadable)")

    max_w = max(w for w, _ in sizes)
    max_h = max(h for _, h in sizes)
    print(f"[Info] Found {len(valid_pngs)} usable PNGs; unified canvas size -> {max_w}x{max_h}")

    # Font and annotation settings
    font = load_font(size=60)
    margin = 20

    frames = []
    for p in valid_pngs:
        with Image.open(p).convert("RGBA") as im:
            w, h = im.size
            canvas = Image.new("RGBA", (max_w, max_h), "white")
            # Center paste; if alpha exists, use as mask
            canvas.paste(im, ((max_w - w) // 2, (max_h - h) // 2), im)

            if annotate:
                text = extract_timeline_text(p)
                draw = ImageDraw.Draw(canvas)
                l, t, r, b = draw.textbbox((0, 0), text, font=font)
                tw, th = r - l, b - t
                x, y = max_w - tw - margin, margin
                # White text with black stroke for readability
                draw.text((x, y), text, font=font, fill="white",
                          stroke_width=3, stroke_fill="black")

            frames.append(canvas.convert("RGB"))

    images = [np.array(f) for f in frames]

    # Ensure output directory exists
    out_path = Path(out_gif)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as GIF
    try:
        imageio.mimsave(out_path, images, fps=fps, loop=loop)
    except Exception as e:
        print(f"[Error] Failed to generate GIF: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] GIF saved: {out_path} (frames={len(images)}, fps={fps}, loop={'∞' if loop == 0 else loop})")


def main():
    parser = argparse.ArgumentParser(description="Combine PNG images into a GIF (support directory or file list)")
    parser.add_argument("inputs", nargs="+", help="A directory, or one or more PNG file paths")
    parser.add_argument("--gif", required=True, help="Output GIF path")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second (default: 5)")
    parser.add_argument("--loop", type=int, default=0, help="Loop count (0 = infinite)")
    parser.add_argument("--no-annotate", action="store_true", help="Disable timeline annotation overlay")
    args = parser.parse_args()

    make_gif(
        inputs=args.inputs,
        out_gif=args.gif,
        fps=args.fps,
        loop=args.loop,
        annotate=not args.no_annotate,
    )


if __name__ == "__main__":
    main()
