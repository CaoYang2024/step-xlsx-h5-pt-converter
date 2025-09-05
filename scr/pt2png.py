#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def to_numpy(arr):
    """兼容 torch.Tensor / numpy.ndarray"""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        raise TypeError(f"不支持的类型: {type(arr)}")


def plot_mesh(pt_path: str, out_png: str = "preview.png",
              xlim=(0, 200), ylim=(0, 200), zlim=(-100, 100),
              thick_range=(0.8, 1.2), reverse_cmap=False):
    """
    从 .pt 读取 pos(N,3) 和 faces(Nf,4) 画网格，固定坐标范围与颜色范围：
      - xlim = (0, 200) mm
      - ylim = (0, 200) mm
      - zlim = (0, 100) mm
      - 厚度颜色范围 fixed 为 thick_range=(0.8, 1.2) mm
    """
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)

    if "pos" not in obj or "faces" not in obj:
        raise ValueError(f"{pt_path} 中缺少 pos 或 faces")

    pos = to_numpy(obj["pos"]).astype(np.float32, copy=False)
    faces = to_numpy(obj["faces"]).astype(np.int64, copy=False)

    # 可选厚度
    faces_t = None
    if "faces_t_float" in obj:
        ft = to_numpy(obj["faces_t_float"])
        if isinstance(ft, np.ndarray) and ft.ndim == 2 and ft.shape[1] >= 5:
            faces_t = ft[:, 4].astype(np.float32, copy=False)

    print("pos shape:", pos.shape)
    print("faces shape:", faces.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 组装每个面的四个点
    quads = [[pos[idx] for idx in f] for f in faces]

    # 颜色：固定厚度范围 0.8~1.2 mm
    cmap = plt.get_cmap("viridis_r" if reverse_cmap else "viridis")
    if faces_t is not None:
        vmin, vmax = thick_range
        # 避免 vmin==vmax
        if np.isclose(vmin, vmax):
            vmin, vmax = vmin - 1e-6, vmax + 1e-6
        norm = plt.Normalize(vmin=float(vmin), vmax=float(vmax))
        facecolors = [cmap(norm(float(t))) for t in faces_t]
    else:
        norm = None
        facecolors = "#2a9d8f"  # 无厚度时的统一颜色

    mesh = Poly3DCollection(
        quads,
        facecolors=facecolors,
        alpha=0.95,
        edgecolor="#2f2f2f",
        linewidths=0.4
    )
    ax.add_collection3d(mesh)

    # 固定坐标范围（统一所有帧的视窗）
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    # 视角接近示例
    ax.view_init(elev=28, azim=-60)

    # 轴标签（单位）
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")

    # 网格 + 面板
    ax.grid(True, which="both", linewidth=0.6, alpha=0.35)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((0.9, 0.9, 0.9, 0.25))
        pane.set_edgecolor("#b0b0b0")
    ax.tick_params(axis='both', which='major', labelsize=9)

    # 厚度 colorbar（固定范围 0.8~1.2 mm）
    if faces_t is not None:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(faces_t)
        cb = fig.colorbar(mappable, ax=ax, shrink=0.8, aspect=18, pad=0.02)
        cb.set_label("Thickness [mm]")
        cb.set_ticks([thick_range[0], (thick_range[0]+thick_range[1])/2, thick_range[1]])

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"PNG 已保存: {out_png}")



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Plot mesh from .pt file and save as PNG (with thickness coloring)")
    ap.add_argument("pt_file", type=str, help="输入 .pt 文件路径")
    ap.add_argument("--png", type=str, default="preview.png", help="输出 PNG 文件路径")
    args = ap.parse_args()

    plot_mesh(args.pt_file, args.png)
