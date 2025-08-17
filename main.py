import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required to enable 3D plotting


def load_points_from_pt(file_path):
    """Load a .pt file and return a (N, 3) numpy array."""
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
    return points.astype(np.float32, copy=False)


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres look like spheres, cubes like cubes, etc.
    This is a workaround for Matplotlib's mplot3d, which does not support equal aspect by default.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range) / 2.0

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range, x_middle + max_range])
    ax.set_ylim3d([y_middle - max_range, y_middle + max_range])
    ax.set_zlim3d([z_middle - max_range, z_middle + max_range])


def plot_points(points, out_path=None, show=True, s=1.0, elev=None, azim=None):
    """Visualize a point cloud with equal axis scales."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 颜色用 Z 值做个简易着色；如不需要可去掉 c/cmap
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                    s=s, c=points[:, 2], cmap="jet")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 等比例缩放（关键）
    set_axes_equal(ax)

    # 可选视角
    if elev is not None or azim is not None:
        ax.view_init(elev=elev if elev is not None else ax.elev,
                     azim=azim if azim is not None else ax.azim)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=300)
        print(f"[saved] {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize a 3D point cloud from a .pt file (with equal XYZ scale)")
    parser.add_argument("input", help="Path to the input .pt file")
    parser.add_argument("--save", help="Path to save the image (optional)", default=None)
    parser.add_argument("--size", type=float, default=0.5, help="Point size, default=0.5")
    parser.add_argument("--elev", type=float, default=None, help="Elevation angle for view_init (optional)")
    parser.add_argument("--azim", type=float, default=None, help="Azimuth angle for view_init (optional)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"File not found: {args.input}")

    points = load_points_from_pt(args.input)
    print(f"[info] Loaded point cloud: {points.shape} (dtype={points.dtype})")

    plot_points(points, out_path=args.save, show=True, s=args.size, elev=args.elev, azim=args.azim)


if __name__ == "__main__":
    main()
