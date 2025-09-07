#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl

# ---------- utils ----------
def equal_aspect_3d(ax, xyz):
    """Set equal aspect ratio for 3D axes based on bounding box of xyz points."""
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    max_range = max(x.max()-x.min(), y.max()-y.min(), z.max()-z.min())
    cx, cy, cz = x.mean(), y.mean(), z.mean()
    r = max_range/2 if max_range>0 else 1.0
    ax.set_xlim(cx-r, cx+r); ax.set_ylim(cy-r, cy+r); ax.set_zlim(cz-r, cz+r)

def to_zero_based(idx, n_nodes):
    """Convert indices to zero-based if they look 1-based."""
    idx = np.asarray(idx, dtype=int)
    if idx.min() >= 1 and idx.max() <= n_nodes:
        return idx - 1
    return idx

def load_group(f: h5py.File, group_path: str):
    """
    Load geometry, displacements, and scalar fields from HDF5 group.
    Expected datasets:
      - node_coordinates (N,3)
      - element_shell_node_indexes (E,4) or element_shell_node_ids (E,4)
      - node_displacement (T,N,3) [optional]
      - element_shell_thickness (T,E) or (E,) [optional]
      - element_shell_effective_plastic_strain (T,E,K) [optional]
    """
    g = f[group_path]
    coords = np.asarray(g["node_coordinates"])  # (N,3)
    if "element_shell_node_indexes" in g:
        elems = np.asarray(g["element_shell_node_indexes"])  # (E,4)
    elif "element_shell_node_ids" in g:
        elems = np.asarray(g["element_shell_node_ids"])      # fallback
    else:
        raise KeyError(f"{group_path} missing element_shell_node_indexes / element_shell_node_ids")

    disp = np.asarray(g["node_displacement"]) if "node_displacement" in g else None  # (T,N,3)

    # Optional per-element scalar fields
    thick = np.asarray(g["element_shell_thickness"]) if "element_shell_thickness" in g else None
    epsp  = np.asarray(g["element_shell_effective_plastic_strain"]) if "element_shell_effective_plastic_strain" in g else None

    return coords, elems, disp, thick, epsp

def scalar_per_element(thick, epsp, which, t_index, reduce_mode="mean"):
    """Return per-element scalar values (thickness or epsp) at timestep t_index, with optional reduction."""
    if which is None or which == "none": return None
    if which == "thickness" and thick is not None:
        if thick.ndim == 2:  # (T,E)
            t = max(0, min(t_index, thick.shape[0]-1))
            return thick[t]           # (E,)
        elif thick.ndim == 1:
            return thick
    if which == "epsp" and epsp is not None:
        arr = epsp
        if arr.ndim == 3:  # (T,E,K)
            t = max(0, min(t_index, arr.shape[0]-1))
            A = arr[t]  # (E,K)
        elif arr.ndim == 2:
            A = arr     # (E,K)
        else:
            raise ValueError(f"epsp shape not supported: {arr.shape}")
        if reduce_mode == "mean": return A.mean(axis=-1)
        if reduce_mode == "max":  return A.max(axis=-1)
        if reduce_mode == "min":  return A.min(axis=-1)
        raise ValueError(f"Invalid reduce_mode: {reduce_mode}")
    return None

def build_quads(pts, elems):
    """Build list of quad vertex coordinates from points and element connectivity."""
    quads = []
    for e in elems:
        a,b,c,d = e
        quads.append([pts[a], pts[b], pts[c], pts[d]])
    return quads

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Animate FE shell mesh deformation from HDF5.")
    ap.add_argument("h5", type=Path, help="HDF5 file")
    ap.add_argument("--group", default="OP10/blank", help="Group path, e.g. OP10/blank")
    ap.add_argument("--scale", type=float, default=1.0, help="Displacement scale factor")
    ap.add_argument("--color", choices=["none","thickness","epsp"], default="none", help="Per-element coloring")
    ap.add_argument("--reduce", choices=["mean","max","min"], default="mean", help="Reduction for multi-point scalars")
    ap.add_argument("--fps", type=int, default=10, help="Frames per second")
    ap.add_argument("--skip", type=int, default=1, help="Use every K-th timestep")
    ap.add_argument("--edge", action="store_true", help="Draw element edges")
    ap.add_argument("--out", type=Path, default=None, help="Output path (.gif/.mp4/.png). Default: auto")
    ap.add_argument("--limit_elems", type=int, default=0, help="Randomly sample this many elements for speed (0=all)")
    ap.add_argument("--cmap", default="viridis", help="Matplotlib colormap name")
    ap.add_argument("--last", action="store_true", help="Only plot the last frame as a static PNG instead of animation")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        coords, elems, disp, thick, epsp = load_group(f, args.group)

    n_nodes = coords.shape[0]
    elems = to_zero_based(elems, n_nodes)

    if disp is None:
        raise RuntimeError(f"{args.group} has no node_displacement; cannot animate deformation.")

    if disp.ndim != 3:
        raise ValueError(f"node_displacement expected (T,N,3), got {disp.shape}")
    T, N, _ = disp.shape
    frames_t = list(range(0, T, max(1, args.skip)))
    if len(frames_t) == 0: frames_t = [0]

    if args.limit_elems and args.limit_elems < elems.shape[0]:
        rng = np.random.default_rng(1234)
        idx = rng.choice(elems.shape[0], size=args.limit_elems, replace=False)
        elems = elems[idx]

    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection="3d")

    if args.last:
        # --- Static plot: last frame ---
        t_last = T - 1
        pts = coords + args.scale * disp[t_last]
        quads = build_quads(pts, elems)
        coll = Poly3DCollection(quads,
                            linewidths=0.3 if args.edge else 0.0,
                            edgecolors="k" if args.edge else None)
        coll.set_alpha(0.95)
        ax.add_collection3d(coll)

        equal_aspect_3d(ax, pts)
        ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
        ax.set_title(f"{args.group}  t={t_last}")

        if args.color != "none":
            vals = scalar_per_element(thick, epsp, args.color, t_last, args.reduce)
            norm = mpl.colors.Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))
            cmap = mpl.colormaps[args.cmap]
            fc = cmap(norm(vals))
            coll.set_facecolor(fc)
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array(vals)
            cbar = plt.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label(args.color)

        out = args.out
        if out is None:
            safe = args.group.replace("/", "_")
            out = Path(f"{args.h5.stem}_{safe}_t{t_last}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close(fig)
        print(f"[OK] Saved static PNG: {out}")

    else:
        # --- Animation ---
        pts0 = coords + args.scale * disp[frames_t[0]]
        quads0 = build_quads(pts0, elems)
        coll = Poly3DCollection(quads0,
                            linewidths=0.3 if args.edge else 0.0,
                            edgecolors="k" if args.edge else None)
        coll.set_alpha(0.95)
        ax.add_collection3d(coll)

        equal_aspect_3d(ax, pts0)
        ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
        title = ax.set_title(f"{args.group}  t={frames_t[0]}")

        mappable = None
        if args.color != "none":
            vals0 = scalar_per_element(thick, epsp, args.color, frames_t[0], args.reduce)
            norm = mpl.colors.Normalize(vmin=np.nanmin(vals0), vmax=np.nanmax(vals0))
            cmap = mpl.colormaps[args.cmap]
            fc = cmap(norm(vals0))
            coll.set_facecolor(fc)
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array(vals0)
            cbar = plt.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label(args.color)

        def update(frame_t):
            pts = coords + args.scale * disp[frame_t]
            quads = build_quads(pts, elems)
            coll.set_verts(quads)
            title.set_text(f"{args.group}  t={frame_t}")
            if args.color != "none":
                vals = scalar_per_element(thick, epsp, args.color, frame_t, args.reduce)
                fc = mappable.cmap(mappable.norm(vals))
                coll.set_facecolor(fc)
            return coll, title

        ani = animation.FuncAnimation(fig, update, frames=frames_t,
                                      blit=False, interval=1000/max(1,args.fps))

        out = args.out
        if out is None:
            safe = args.group.replace("/", "_")
            out = Path(f"{args.h5.stem}_{safe}.gif")
        out = Path(out)
        if out.suffix.lower() == ".mp4":
            try:
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=args.fps, bitrate=4000)
                ani.save(out, writer=writer, dpi=200)
            except Exception as e:
                print(f"[WARN] mp4 requires ffmpeg, falling back to GIF. ({e})")
                out = out.with_suffix(".gif")
                ani.save(out, writer="pillow", fps=args.fps, dpi=200)
        else:
            ani.save(out, writer="pillow", fps=args.fps, dpi=200)

        plt.close(fig)
        print(f"[OK] Saved animation: {out}")


if __name__ == "__main__":
    main()
