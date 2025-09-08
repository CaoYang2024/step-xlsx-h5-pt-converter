#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import h5py

def guess_col(df, keys, fallback=None):
    """Fuzzy-match one column name in `df` from a list of candidate `keys`. Returns the first match or `fallback`."""
    cols = list(df.columns)
    pattern = re.compile("|".join([re.escape(k) for k in keys]), re.IGNORECASE)
    for c in cols:
        if pattern.search(str(c)):
            return c
    return fallback

def load_points_from_xlsx(xlsx_path, colx=None, coly=None, colz=None, colt=None):
    df = pd.read_excel(xlsx_path)

    # Auto-detect column names
    cx = colx or guess_col(df, ["x", "x (mm)"])
    cy = coly or guess_col(df, ["y", "y (mm)"])
    cz = colz or guess_col(df, ["z", "z (mm)"])
    ct = colt or guess_col(df, ["thick", "thickness", "t (mm)", "dicke"])

    missing = [n for n in [cx, cy, cz, ct] if n is None]
    if missing:
        raise ValueError(
            f"[{xlsx_path}] Failed to infer column names. "
            f"Specify colx/coly/colz/colt explicitly. DataFrame columns={list(df.columns)}"
        )

    # Convert to float
    x = pd.to_numeric(df[cx], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[cy], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(df[cz], errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(df[ct], errors="coerce").to_numpy(dtype=float)

    # Drop rows with NaN in coordinates (keep 1:1 alignment with thickness)
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z, t = x[mask], y[mask], z[mask], t[mask]

    # The number of points must be a multiple of 4
    if len(x) % 4 != 0:
        raise ValueError(
            f"[{xlsx_path}] Number of points {len(x)} is not a multiple of 4; "
            "cannot form elements from every 4 points."
        )

    coords = np.stack([x, y, z], axis=1).astype(np.float64)  # (N,3)
    return coords, t

def build_mesh_from_points(coords, thickness_vec, thickness_source="first"):
    """
    coords: (N,3), with N % 4 == 0
    thickness_vec: (N,) — may have only one valid value per group of 4 (others NaN/empty)
    thickness_source: 'first' | 'last' — pick thickness from the 1st or 4th point in each group of 4
    """
    N = coords.shape[0]
    if N % 4 != 0:
        raise ValueError("Total number of coords must be a multiple of 4.")

    # node_ids start from 1
    node_ids = np.arange(1, N + 1, dtype=np.int64)

    # Every 4 points form a quad: number of elements M
    M = N // 4
    # element_shell_ids start from 1
    elem_ids = np.arange(1, M + 1, dtype=np.int64)

    # element_shell_node_ids: rows are [n1, n2, n3, n4]
    elem_nodes = np.arange(N, dtype=np.int64).reshape(M, 4) + 1  # 1-based

    # Thickness: by default pick the 1st or the 4th point in each group of 4
    if thickness_source == "first":
        pick_idx = 0
    elif thickness_source == "last":
        pick_idx = 3
    else:
        raise ValueError("thickness_source must be 'first' or 'last'.")

    # Normalize thickness_vec to length N (missing treated as NaN)
    t = np.array(thickness_vec, dtype=float)
    if t.shape[0] != N:
        # Also support per-element thickness (length M)
        if t.shape[0] == M:
            elem_thick = t.astype(np.float64)
            return node_ids, coords.astype(np.float64), elem_ids, elem_nodes, elem_thick
        else:
            raise ValueError(f"thickness length ({t.shape[0]}) is neither N ({N}) nor M ({M}).")

    # Extract per-element thickness:
    #   take pick_idx-th value; if NaN, fall back to the first non-NaN in the group; if all NaN, use 0.0
    elem_thick = np.empty((M,), dtype=np.float64)
    for i in range(M):
        group = t[i*4:(i+1)*4]
        val = group[pick_idx]
        if np.isnan(val):
            valid = group[~np.isnan(group)]
            if len(valid) > 0:
                val = valid[0]
            else:
                val = 0.0  # all NaN -> 0.0 (with a soft fallback)
        elem_thick[i] = float(val)

    return node_ids, coords.astype(np.float64), elem_ids, elem_nodes, elem_thick

def write_group(h5: h5py.File, group_name: str,
                node_ids, node_coords, elem_ids, elem_nodes, elem_thick):
    g = h5.require_group(group_name)
    # Overwrite datasets if they exist
    for name, data in [
        ("node_ids", node_ids),
        ("node_coordinates", node_coords),
        ("element_shell_ids", elem_ids),
        ("element_shell_node_ids", elem_nodes),
        ("element_shell_thickness", elem_thick),
    ]:
        if name in g:
            del g[name]
        g.create_dataset(name, data=data)

def main():
    ap = argparse.ArgumentParser(
        description="Pack multiple XLSX files (x,y,z,thickness) into HDF5 with required datasets."
    )
    ap.add_argument("xlsx", nargs="+", help="Input XLSX files")
    ap.add_argument("--out", required=True, help="Output HDF5 path")
    ap.add_argument(
        "--single-group",
        default=None,
        help="Store all inputs under a single group (e.g., 'blank'). "
             "If omitted, each file uses its stem as the group name."
    )
    ap.add_argument("--colx", default=None, help="Name of X column")
    ap.add_argument("--coly", default=None, help="Name of Y column")
    ap.add_argument("--colz", default=None, help="Name of Z column")
    ap.add_argument("--colt", default=None, help="Name of thickness column")
    ap.add_argument(
        "--thickness-source",
        choices=["first", "last"],
        default="first",
        help="When per-point thickness is sparse, pick from the 1st or 4th point in each 4-point group."
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "a") as h5:
        for xp in args.xlsx:
            xlsx_path = Path(xp)
            if not xlsx_path.exists():
                print(f"[Skip] Not found: {xlsx_path}")
                continue

            coords, tvec = load_points_from_xlsx(
                xlsx_path,
                colx=args.colx, coly=args.coly, colz=args.colz, colt=args.colt
            )
            node_ids, node_coords, elem_ids, elem_nodes, elem_thick = build_mesh_from_points(
                coords, tvec, thickness_source=args.thickness_source
            )

            gname = args.single_group if args.single_group else xlsx_path.stem
            write_group(h5, gname, node_ids, node_coords, elem_ids, elem_nodes, elem_thick)
            print(
                f"[OK] {xlsx_path.name} → group '{gname}' "
                f"(nodes={len(node_ids)}, elements={len(elem_ids)})"
            )

    print(f"[Done] Saved to {out_path}")

if __name__ == "__main__":
    main()
