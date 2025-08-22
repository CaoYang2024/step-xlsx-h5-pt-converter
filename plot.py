import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required to enable 3D plotting

def load_points_from_pt(file_path):
    """Load a .pt file and return a (N, 3) numpy array"""
    obj = torch.load(file_path, map_location="cpu", weights_only=False)

    points = None
    if isinstance(obj, dict) and "pos" in obj and torch.is_tensor(obj["pos"]):
        points = obj["pos"].cpu().numpy()
    elif torch.is_tensor(obj) and obj.ndim == 2 and obj.shape[1] == 3:
        points = obj.cpu().numpy()
    elif hasattr(obj, "pos") and torch.is_tensor(obj.pos):
        points = obj.pos.cpu().numpy()
    else:
        raise ValueError(f"Could not find (N,3) shaped 'pos' point cloud data in {file_path}")

    if points.shape[1] != 3:
        raise ValueError(f"Point cloud does not have 3 dimensions: shape={points.shape}")
    return points

def plot_points(points, out_path=None, show=True, s=1):
    """Visualize a point cloud"""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=s, c=points[:, 2], cmap="jet")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.set_xlim([0, 300])
    ax.set_ylim([0, 300])
    ax.set_zlim([0, 300])
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=300)
        print(f"[saved] {out_path}")
    if show:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize a 3D point cloud from a .pt file")
    parser.add_argument("input", help="Path to the input .pt file")
    parser.add_argument("--save", help="Path to save the image (optional)", default=None)
    parser.add_argument("--size", type=float, default=0.5, help="Point size, default=0.5")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"File not found: {args.input}")

    points = load_points_from_pt(args.input)
    print(f"[info] Loaded point cloud: {points.shape} (dtype={points.dtype})")
    plot_points(points, out_path=args.save, show=True, s=args.size)

if __name__ == "__main__":
    main()
