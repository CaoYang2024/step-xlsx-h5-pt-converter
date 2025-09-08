#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import warnings
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch


# —— Utility Functions ————————————————————————————————————————————————————————————
def normalize_colname(name: str) -> str:
    """Normalize column names: lowercase, strip whitespace, replace Chinese brackets with English ones, and remove units in parentheses."""
    s = (name or "").strip().lower()
    # Full-width Chinese parentheses -> English
    s = s.replace("（", "(").replace("）", ")")
    # Remove text inside parentheses (units)
    if "(" in s and ")" in s:
        try:
            l = s.index("(")
            r = s.index(")", l + 1)
            s = (s[:l] + s[r + 1:]).strip()
        except Exception:
            pass
    # Remove common delimiters
    for ch in [" ", "\t", "\n", "\r", ":", "；", ";", "，", ","]:
        s = s.replace(ch, "")
    return s


def find_columns(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    """Match x, y, z, t(thickness) columns; supports headers like x(mm), thickness, dicke, etc."""
    norm_map: Dict[str, str] = {normalize_colname(c): c for c in df.columns}

    # Allowed aliases
    x_candidates = ["x"]
    y_candidates = ["y"]
    z_candidates = ["z"]
    t_candidates = ["t", "thickness", "thick", "thk"]

    def pick(cands: List[str]) -> str:
        for c in cands:
            if c in norm_map:
                return norm_map[c]
        # fallback for headers like 'xmm', 'xcoordinate' (prefix match)
        for k, v in norm_map.items():
            if any(k.startswith(c) for c in cands):
                return v
        return ""

    col_x = pick(x_candidates)
    col_y = pick(y_candidates)
    col_z = pick(z_candidates)
    col_t = pick(t_candidates)

    if not all([col_x, col_y, col_z, col_t]):
        if len(df.columns) >= 4:
            warnings.warn("Could not match columns reliably; using the first four columns as x,y,z,t fallback.")
            col_x, col_y, col_z, col_t = df.columns[:4]
        else:
            raise ValueError("Excel must contain at least 4 columns (x,y,z,t/thickness).")
    return col_x, col_y, col_z, col_t


def quantize(arr: np.ndarray, decimals: int) -> np.ndarray:
    """Quantize array values by rounding to fixed decimals (helps deduplication with float tolerance)."""
    return np.round(arr.astype(np.float64), decimals=decimals)


def dedup_points(points: np.ndarray, decimals: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deduplicate points.
    Returns (unique_points, inverse_indices):
      - unique_points: (M,3)
      - inverse_indices: (N,) such that unique_points[inverse_indices] == quantized(points)
    """
    q = quantize(points, decimals)
    dtype = np.dtype((np.void, q.dtype.itemsize * q.shape[1]))
    q_view = np.ascontiguousarray(q).view(dtype)
    _, idx_unique, inv = np.unique(q_view, return_index=True, return_inverse=True)
    unique_points = q[idx_unique]
    return unique_points.astype(np.float32), inv.astype(np.int64)


def build_quads(points: np.ndarray, tvals: np.ndarray, decimals_t: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Group every 4 rows into 1 quad face.
    For thickness, take the first non-empty/non-NaN value in the group.
    Returns (quads_points:(G,4,3), quads_t:(G,))
    """
    n = points.shape[0]
    if n < 4:
        raise ValueError("Less than 4 rows, cannot form quads.")
    if n % 4 != 0:
        warnings.warn(f"Total rows {n} is not a multiple of 4, discarding extra {n % 4} rows.")
        n = n - (n % 4)
        points = points[:n]
        tvals = tvals[:n]

    G = n // 4
    quads_points = points.reshape(G, 4, 3).copy()
    quads_t = np.zeros((G,), dtype=np.float32)

    for g in range(G):
        t_group = tvals[g * 4:(g + 1) * 4]
        # pick the first non-NaN thickness in the group; raise error if all NaN
        t_val = np.nan
        for tv in t_group:
            if tv is not None and not (isinstance(tv, float) and np.isnan(tv)):
                t_val = float(tv)
                break
        if np.isnan(t_val):
            raise ValueError(f"Group {g+1} has no valid thickness (all four rows are empty/NaN).")
        quads_t[g] = t_val

    quads_t = quantize(quads_t, decimals_t).astype(np.float32)
    return quads_points.astype(np.float32), quads_t


def indices_from_unique(points_unique: np.ndarray, quads_points: np.ndarray, decimals_xyz: int) -> np.ndarray:
    """Map the 4 coordinates of each quad to indices in the unique point table. Returns quads_idx:(G,4)."""
    q_uni = quantize(points_unique, decimals_xyz)
    key_to_idx: Dict[Tuple[float, float, float], int] = {
        (row[0], row[1], row[2]): i for i, row in enumerate(q_uni)
    }
    G = quads_points.shape[0]
    qp_q = quantize(quads_points.reshape(G * 4, 3), decimals_xyz).reshape(G, 4, 3)
    quads_idx = np.zeros((G, 4), dtype=np.int64)
    for g in range(G):
        for k in range(4):
            key = tuple(qp_q[g, k].tolist())
            if key not in key_to_idx:
                raise RuntimeError(f"Point {key} not found in unique index table; check --decimals_xyz setting.")
            quads_idx[g, k] = key_to_idx[key]
    return quads_idx


def dedup_quads(quads_idx: np.ndarray, quads_t: np.ndarray, decimals_t: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deduplicate quads: ignore vertex ordering; include quantized thickness in the key.
    Returns (faces:(M,4), t:(M,))
    """
    keys = {}
    faces_list: List[List[int]] = []
    t_list: List[float] = []

    for i in range(quads_idx.shape[0]):
        idx_sorted = tuple(sorted(quads_idx[i].tolist()))
        t_key = float(np.round(quads_t[i], decimals_t))
        key = (idx_sorted, t_key)
        if key in keys:
            continue
        keys[key] = True
        faces_list.append(list(quads_idx[i]))
        t_list.append(quads_t[i])

    faces = np.asarray(faces_list, dtype=np.int64)
    t = np.asarray(t_list, dtype=np.float32)
    return faces, t


# —— Main Process ——————————————————————————————————————————————————————————————
def main():
    ap = argparse.ArgumentParser(description="Convert Excel (x,y,z,thickness per 4 points) to .pt with deduplicated points & quads.")
    ap.add_argument("excel", type=Path, help="Input Excel path (.xlsx/.xls)")
    ap.add_argument("out_pt", type=Path, help="Output .pt path")
    ap.add_argument("--sheet", type=str, default=None, help="Worksheet name (default: first sheet)")
    ap.add_argument("--decimals_xyz", type=int, default=6, help="Quantization decimals for coordinates (dedup tolerance)")
    ap.add_argument("--decimals_t", type=int, default=6, help="Quantization decimals for thickness (dedup tolerance)")
    ap.add_argument("--unit_scale", type=float, default=1.0, help="Unit scale (e.g., use 0.001 for mm -> m)")
    args = ap.parse_args()

    # Load Excel
    df = pd.read_excel(args.excel, sheet_name=args.sheet) if args.sheet else pd.read_excel(args.excel)

    col_x, col_y, col_z, col_t = find_columns(df)
    xyz = df[[col_x, col_y, col_z]].to_numpy(dtype=np.float64)
    tvals = df[[col_t]].to_numpy(dtype=np.float64).reshape(-1)

    # Apply unit scaling (e.g. if input in mm and you want meters, set --unit_scale=0.001)
    if args.unit_scale != 1.0:
        xyz = xyz * float(args.unit_scale)

    # Group every 4 rows -> 1 quad; thickness = first non-NaN value in the group
    quads_points, quads_t = build_quads(xyz, tvals, args.decimals_t)

    # Deduplicate points
    all_points = quads_points.reshape(-1, 3)
    points_unique, _ = dedup_points(all_points, args.decimals_xyz)

    # Build quad indices into unique point table
    quads_idx = indices_from_unique(points_unique, quads_points, args.decimals_xyz)

    # Deduplicate quads
    faces, t_unique = dedup_quads(quads_idx, quads_t, args.decimals_t)

    # Package and save
    out_obj = {
        "pos": points_unique.astype(np.float32),          # (M,3)
        "faces": faces.astype(np.int64),                  # (N,4)
        # Convenience: directly usable (ID1, ID2, ID3, ID4, t)
        "faces_t_float": np.hstack([faces.astype(np.float32),
                                    t_unique.reshape(-1, 1)]).astype(np.float32)  # (N,5)
    }
    torch.save(out_obj, args.out_pt)

    print("==== Done ====")
    print(f"Input file: {args.excel}")
    print(f"Unique points M: {out_obj['pos'].shape[0]}")
    print(f"Deduplicated quads N: {out_obj['faces'].shape[0]}")
    print(f".pt saved: {args.out_pt}")


if __name__ == "__main__":
    main()
