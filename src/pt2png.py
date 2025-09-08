#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def to_numpy(arr):
    """Convert torch.Tensor / numpy.ndarray to numpy, compatible with both."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        raise TypeError(f"Unsupported type: {type(arr)}")


def plot_mesh(pt_path: str, out_png: str = "preview.png",
              xlim=(0, 200), ylim=(0, 200), zlim=(-100, 100),
              thick_range=(0.8, 1.2), reverse_cmap=False):
    """
    Load pos(N,3) and faces(Nf,4) from a .pt file and plot the mesh.
    Fixed axis ranges and thickness color range:
      - xlim = (0, 200) mm
      - ylim = (0, 200) mm
      - zlim = (-100, 100) mm
      - thickness color range fixed as thick_range=(0.8, 1.2) mm
    """
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)

    if "pos" not in obj or "faces" not in obj:
        raise ValueError(f"Missing 'pos' or 'faces' in {pt_path}")

    pos = to_numpy(obj["pos"]).astype(np.float32, copy=False)
    faces = to_numpy(obj["faces"]).astype(np.int64, copy=False)

    # Optional thickness data
    faces_t = None
    if "faces_t_float" in obj:
        ft = to_numpy(obj["faces_t_float"])
        if isinstance(ft, np.ndarray) and ft.ndim == 2 and ft.shape[1] >= 5:
            faces_t = ft[:, 4].astype(np.float32, copy=False)

    print("pos shape:", pos.shape)
    print("faces shape:", faces.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Assemble each quad face from vertex indices
    quads = [[pos[idx] for idx in f] for f in faces]

    # Color mapping: fixed thickness range (0.8~1.2 mm)
    cmap = plt.get_cmap("viridis_r" if reverse_cmap else "viridis")
    if faces_t is not None:
        vmin, vmax = thick_range
        # Avoid vmin == vmax
        if np.isclose(vmin, vmax):
            vmin, vmax = vmin - 1e-6, vmax + 1e-6
        norm = plt.Normalize(vmin=float(vmin), vmax=float(vmax))
        facecolors = [cmap(norm(float(t))) for t in faces_t]
    else:
        norm = None
        facecolors = "#2a9d8f"  # Single color if no thickness data

    mesh = Poly3DCollection(
        quads,
        facecolors=facecolors,
        alpha=0.95,
        edgecolor="#2f2f2f",
        linewidths=0.4
    )
    ax.add_collection3d(mesh)

    # Fix axis limits (consistent viewport across renders)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    # View angle similar to example
    ax.view_init(elev=28, azim=-60)

    # Axis labels with units
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")

    # Grid and pane backgrounds
    ax.grid(True, which="both", linewidth=0.6, alpha=0.35)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((0.9, 0.9, 0.9, 0.25))
        pane.set_edgecolor("#b0b0b0")
    ax.tick_params(axis='both', which='major', labelsize=9)

    # Thickness colorbar (fixed range 0.8~1.2 mm)
    if faces_t is not None:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(faces_t)
        cb = fig.colorbar(mappable, ax=ax, shrink=0.8, aspect=18, pad=0.02)
        cb.set_label("Thickness [mm]")
        cb.set_ticks([thick_range[0], (thick_range[0] + thick_range[1]) / 2, thick_range[1]])

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"PNG saved: {out_png}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Plot mesh from .pt file and save as PNG (with thickness coloring)")
    ap.add_argument("pt_file", type=str, help="Input .pt file path")
    ap.add_argument("--png", type=str, default="preview.png", help="Output PNG file path")
    args = ap.parse_args()

    plot_mesh(args.pt_file, args.png)
