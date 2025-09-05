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
    """自然排序：按数字块排序，其他按不区分大小写的字符串排序"""
    parts = re.split(r"(\d+)", s)
    key = []
    for t in parts:
        if t.isdigit():
            key.append(int(t))
        else:
            key.append(t.lower())
    return key


def extract_timeline_text(p: Path) -> str:
    """从文件名提取第一段数字作为标注；没有数字就用原文件名（不含后缀）"""
    m = re.search(r"(\d+)", p.stem)
    return m.group(1) if m else p.stem


def load_font(size: int = 60) -> ImageFont.FreeTypeFont:
    """优先尝试常见系统字体，失败则回退到 Pillow 自带字体"""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # 常见 Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Unicode.ttf",                      # macOS 可能存在
        "C:/Windows/Fonts/arial.ttf",                            # Windows
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    # 回退
    return ImageFont.load_default()


def collect_inputs(inputs: List[str]) -> List[Path]:
    """
    将输入解析成 PNG 文件列表：
    - 若 inputs 只有一个目录，只挑选形如 `Tiefgezogenes Bauteil_<number>.png` 的文件，
      并按 <number> 倒序排列。
    - 若 inputs 是多个具体文件，则仍会过滤出符合该命名格式的文件并倒序。
    """
    # 仅匹配：Tiefgezogenes Bauteil_<数字>.png
    pattern = re.compile(r"^Tiefgezogenes Bauteil_(\d+)\.png$")

    def key_num(p: Path) -> int:
        m = pattern.match(p.name)
        return int(m.group(1)) if m else -1  # 不匹配的会被过滤，不会用到

    if len(inputs) == 1 and Path(inputs[0]).is_dir():
        d = Path(inputs[0])
        # 只取符合命名格式的 PNG
        candidates = [p for p in d.iterdir() if p.is_file() and pattern.match(p.name)]
        # 按数字倒序
        candidates.sort(key=key_num, reverse=True)
        return candidates
    else:
        files = [Path(x) for x in inputs]
        pngs = [p for p in files if p.exists() and pattern.match(p.name)]
        pngs.sort(key=key_num, reverse=True)
        return pngs


def ensure_non_empty(pngs: List[Path], where: str):
    if not pngs:
        print(f"[Error] 未找到 PNG 输入：{where}", file=sys.stderr)
        sys.exit(1)


def make_gif(inputs: List[str], out_gif: str, fps: int = 5, loop: int = 0, annotate: bool = True):
    # 收集输入
    pngs = collect_inputs(inputs)
    where = inputs[0] if len(inputs) == 1 else f"{len(inputs)} files"
    ensure_non_empty(pngs, where)

    # 打印顺序
    print("[Info] PNG 顺序：")
    for p in pngs:
        print("   ", p.name)

    # 统一画布尺寸（取最大宽高）
    sizes = []
    valid_pngs = []
    for p in pngs:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
                valid_pngs.append(p)
        except (UnidentifiedImageError, OSError) as e:
            print(f"[Warn] 无法读取图像，已跳过：{p} ({e})", file=sys.stderr)

    ensure_non_empty(valid_pngs, "有效 PNG 列表（有损坏或不可读文件）")

    max_w = max(w for w, _ in sizes)
    max_h = max(h for _, h in sizes)
    print(f"[Info] 发现 {len(valid_pngs)} 张可用 PNG；统一画布尺寸 -> {max_w}x{max_h}")

    # 字体与标注
    font = load_font(size=60)
    margin = 20

    frames = []
    for p in valid_pngs:
        with Image.open(p).convert("RGBA") as im:
            w, h = im.size
            canvas = Image.new("RGBA", (max_w, max_h), "white")
            # 居中粘贴；带 alpha 时使用自身为 mask
            canvas.paste(im, ((max_w - w) // 2, (max_h - h) // 2), im)

            if annotate:
                text = extract_timeline_text(p)
                draw = ImageDraw.Draw(canvas)
                # textbbox 返回 (l, t, r, b)
                l, t, r, b = draw.textbbox((0, 0), text, font=font)
                tw, th = r - l, b - t
                x, y = max_w - tw - margin, margin
                # 黑色描边，白色字，保证可读性
                draw.text((x, y), text, font=font, fill="white", stroke_width=3, stroke_fill="black")

            frames.append(canvas.convert("RGB"))

    images = [np.array(f) for f in frames]

    # 确保输出目录存在
    out_path = Path(out_gif)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 合成 GIF
    try:
        imageio.mimsave(out_path, images, fps=fps, loop=loop)
    except Exception as e:
        print(f"[Error] GIF 合成失败：{e}", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] GIF 已保存：{out_path}（frames={len(images)}, fps={fps}, loop={'∞' if loop == 0 else loop}）")


def main():
    parser = argparse.ArgumentParser(description="将若干 PNG 合成为 GIF（支持传目录或文件列表）")
    parser.add_argument("inputs", nargs="+",
                        help="一个目录，或若干 PNG 文件路径")
    parser.add_argument("--gif", required=True, help="输出 GIF 路径")
    parser.add_argument("--fps", type=int, default=5, help="帧率（默认 5）")
    parser.add_argument("--loop", type=int, default=0, help="循环次数（0=无限）")
    parser.add_argument("--no-annotate", action="store_true", help="不在角上叠加时间标注")
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

