#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enable 3D)

def load_points_and_faces(obj):
    """
    兼容多种 .pt 结构：
      - dict: 包含 "points"(M,3)、"faces"(N,4)、"thickness"(N,)
               可能还有 "faces_t_float"(N,5)
      - tensor (N,3) 或具备 .pos 的对象
    返回: points(np.float32, M,3), faces(np.int64, N,4) or None
    """
    points = None
    faces = None

    if isinstance(obj, dict):
        # points / pos
        if "points" in obj and torch.is_tensor(obj["points"]):
            points = obj["points"].cpu().numpy().astype(np.float32)
        elif "pos" in obj and torch.is_tensor(obj["pos"]):
            points = obj["pos"].cpu().numpy().astype(np.float32)

        # faces
        if "faces" in obj and torch.is_tensor(obj["faces"]):
            faces = obj["faces"].cpu().numpy().astype(np.int64)
        elif "faces_t_float" in obj and torch.is_tensor(obj["faces_t_float"]):
            tmp = obj["faces_t_float"].cpu().numpy()
            if tmp.shape[1] >= 4:
                faces = tmp[:, :4].astype(np.int64)
    elif torch.is_tensor(obj):
        if obj.ndim == 2 and obj.shape[1] == 3:
            points = obj.cpu().numpy().astype(np.float32)
    elif hasattr(obj, "pos") and torch.is_tensor(obj.pos):
        points = obj.pos.cpu().numpy().astype(np.float32)

    return points, faces

def maybe_fix_index_base(faces: np.ndarray, num_points: int) -> np.ndarray:
    """
    自动判断 faces 是否是 1-based。如果是，则转换为 0-based。
    规则：
      - 若 min==1 且 max==num_points 则判定为 1-based，减 1
      - 若 min>=0 且 max<=num_points-1 则认为已经是 0-based
      - 其他情况：保持不变，但给出提醒
    """
    if faces is None:
        return None
    fmin = faces.min()
    fmax = faces.max()
    if fmin == 1 and fmax == num_points:
        return faces - 1
    if fmin >= 0 and fmax <= num_points - 1:
        return faces
    # 模糊情况：尽量不改动，仅提示
    print(f"[warn] faces index range looks unusual: min={fmin}, max={fmax}, num_points={num_points}. "
          f"Assuming 0-based; override manually if needed.", file=sys.stderr)
    return faces

def main():
    ap = argparse.ArgumentParser(description="Load a .pt (points/faces) and plot + print shapes.")
    ap.add_argument("pt_path", type=Path, help=".pt file path")
    ap.add_argument("--max-faces", type=int, default=None, help="最多绘制的面数（加速）")
    ap.add_argument("--no-faces", action="store_true", help="只画点，不画网格边线")
    ap.add_argument("--save", type=Path, default=None, help="保存为 PNG（不弹窗）")
    ap.add_argument("--elev", type=float, default=20.0, help="视角：仰角")
    ap.add_argument("--azim", type=float, default=-60.0, help="视角：方位角")
    ap.add_argument("--point-size", type=float, default=2.0, help="散点大小")
    args = ap.parse_args()

    obj = torch.load(args.pt_path, map_location="cpu", weights_only=False)
    points, faces = load_points_and_faces(obj)

    # 打印 shape
    if isinstance(obj, dict):
        for k, v in obj.items():
            if torch.is_tensor(v):
                try:
                    print(f"{k:14s}: shape={tuple(v.shape)}, dtype={v.dtype}")
                except Exception:
                    print(f"{k:14s}: (tensor)")
            else:
                print(f"{k:14s}: type={type(v)}")
    else:
        print(f"loaded object type: {type(obj)}")

    if points is None:
        print("[error] 未在 .pt 中找到点 (points/pos)。无法绘图。", file=sys.stderr)
        return

    print(f"points array: {points.shape} (dtype={points.dtype})")
    if faces is not None:
        faces = maybe_fix_index_base(faces, points.shape[0])
        print(f"faces array : {faces.shape} (dtype={faces.dtype})")
    else:
        print("faces array : None (将只绘制散点)")

    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=args.elev, azim=args.azim)

    # 画点
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=args.point_size)

    # 画面（四边形边框）
    if (faces is not None) and (not args.no_faces):
        F = faces
        if args.max_faces is not None and F.shape[0] > args.max_faces:
            F = F[:args.max_faces]
            print(f"[info] 绘制前 {args.max_faces} 个面（共 {faces.shape[0]} 个）以加速。")
        # 逐面画闭合折线
        # 每画一面一次 ax.plot：xs, ys, zs 长度=5（首尾闭合）
        for f in F:
            try:
                pts = points[f]  # (4,3)
            except Exception:
                # 防止越界
                continue
            loop = np.vstack([pts, pts[0:1, :]])  # 闭合
            ax.plot(loop[:, 0], loop[:, 1], loop[:, 2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=300, bbox_inches="tight")
        print(f"[ok] Saved figure to: {args.save}")
    else:
        try:
            plt.show()
        except Exception as e:
            # 无显示环境时兜底保存
            out = Path("preview.png")
            plt.savefig(out, dpi=300, bbox_inches="tight")
            print(f"[warn] 无法显示窗口（{e}）。已保存到 {out}")

if __name__ == "__main__":
    main()
