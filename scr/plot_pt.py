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
    Support multiple .pt formats:
      - dict: containing "points"(M,3), "faces"(N,4), "thickness"(N,)
              possibly also "faces_t_float"(N,5)
      - tensor: (N,3) or an object with attribute .pos
    Returns: points (np.float32, M,3), faces (np.int64, N,4) or None
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
    Automatically detect if faces are 1-based. If so, convert to 0-based.
    Rules:
      - If min==1 and max==num_points → considered 1-based → subtract 1
      - If min>=0 and max<=num_points-1 → considered already 0-based
      - Otherwise: leave unchanged but print a warning
    """
    if faces is None:
        return None
    fmin = faces.min()
    fmax = faces.max()
    if fmin == 1 and fmax == num_points:
        return faces - 1
    if fmin >= 0 and fmax <= num_points - 1:
        return faces
    # Ambiguous case: do not modify, just warn
    print(f"[warn] Faces index range looks unusual: min={fmin}, max={fmax}, num_points={num_points}. "
          f"Assuming 0-based; override manually if needed.", file=sys.stderr)
    return faces


def main():
    ap = argparse.ArgumentParser(description="Load a .pt (points/faces) and plot + print shapes.")
    ap.add_argument("pt_path", type=Path, help=".pt file path")
    ap.add_argument("--max-faces", type=int, default=None, help="Maximum number of faces to plot (speedup)")
    ap.add_argument("--no-faces", action="store_true", help="Plot only points, skip mesh edges")
    ap.add_argument("--save", type=Path, default=None, help="Save as PNG (no interactive window)")
    ap.add_argument("--elev", type=float, default=20.0, help="View elevation angle")
    ap.add_argument("--azim", type=float, default=-60.0, help="View azimuth angle")
    ap.add_argument("--point-size", type=float, default=2.0, help="Point size in scatter plot")
    args = ap.parse_args()

    obj = torch.load(args.pt_path, map_location="cpu", weights_only=False)
    points, faces = load_points_and_faces(obj)

    # Print shape info
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
        print("[error] Could not find points (points/pos) in .pt. Cannot plot.", file=sys.stderr)
        return

    print(f"points array: {points.shape} (dtype={points.dtype})")
    if faces is not None:
        faces = maybe_fix_index_base(faces, points.shape[0])
        print(f"faces array : {faces.shape} (dtype={faces.dtype})")
    else:
        print("faces array : None (will plot only scatter points)")

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=args.elev, azim=args.azim)

    # Scatter points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=args.point_size)

    # Draw face edges
    if (faces is not None) and (not args.no_faces):
        F = faces
        if args.max_faces is not None and F.shape[0] > args.max_faces:
            F = F[:args.max_faces]
            print(f"[info] Drawing first {args.max_faces} faces (out of {faces.shape[0]}) for speed.")
        # Plot each face as a closed loop
        for f in F:
            try:
                pts = points[f]  # (4,3)
            except Exception:
                # Prevent out-of-range index errors
                continue
            loop = np.vstack([pts, pts[0:1, :]])  # Close loop
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
            # Fallback: save to file when no display available
            out = Path("preview.png")
            plt.savefig(out, dpi=300, bbox_inches="tight")
            print(f"[warn] Cannot open interactive window ({e}). Saved to {out}")


if __name__ == "__main__":
    main()
