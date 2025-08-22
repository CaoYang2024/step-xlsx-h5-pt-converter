#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def natural_key(s: str):
    """自然排序：支持文件名里的数字"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def extract_timeline_text(p: Path):
    """从文件名提取数字作为时间线文本"""
    m = re.search(r"(\d+)", p.stem)
    return m.group(1) if m else p.stem

def make_gif(input_dir: str, out_gif: str):
    d = Path(input_dir)
    if not d.exists():
        print(f"[Error] 目录不存在: {d}", file=sys.stderr); sys.exit(1)

    # 找 PNG 文件并排序
    pngs = sorted(d.glob("*.png"), key=lambda p: int(re.search(r"\d+", p.stem).group()))
    if not pngs:
        print(f"[Error] 目录下没有 PNG 文件: {d}", file=sys.stderr); sys.exit(1)
         # 打印顺序
    print("[Info] PNG 顺序：")
    for p in pngs:
        print("   ", p.name)

    # 统一画布尺寸
    sizes = []
    for p in pngs:
        with Image.open(p) as im:
            sizes.append(im.size)
    max_w = max(w for w, h in sizes)
    max_h = max(h for w, h in sizes)

    print(f"[Info] 发现 {len(pngs)} 张 PNG; 统一画布尺寸 -> {max_w}x{max_h}")

    # 字体固定字号=60
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
    margin = 20

    frames = []
    for p in pngs:
        im = Image.open(p).convert("RGBA")
        w, h = im.size

        # 居中 pad 到统一画布
        canvas = Image.new("RGBA", (max_w, max_h), "white")
        canvas.paste(im, ((max_w - w)//2, (max_h - h)//2), im)

        # 时间线文字
        text = extract_timeline_text(p)
        draw = ImageDraw.Draw(canvas)
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        x, y = max_w - tw - margin, margin
        draw.text((x, y), text, font=font, fill="white",
                  stroke_width=3, stroke_fill="black")

        frames.append(canvas.convert("RGB"))

    # 合成 GIF（无限循环）
    images = [np.array(f) for f in frames]
    imageio.mimsave(out_gif, images, fps=5, loop=0)
    print(f"[OK] GIF 已保存：{out_gif}（frames={len(images)}, fps=5, loop=∞）")

if __name__ == "__main__":
    # 固定目录和输出文件名，可按需改
    make_gif("/home/RUS_CIP/st186635/format_transformate", "all_frames.gif")
