#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Excel -> .pt conversion script
- First three columns are interpreted as x,y,z (auto-detect column names, or specify with --xyz-cols)
- Fourth column is thickness (auto-detect; default or NaN will be set to 1)
- Output: dict -> {"pos": Tensor[N,3], "x": Tensor[N,1]}  (x = thickness feature)
"""

import argparse
import os
import sys
import re
import pandas as pd
import numpy as np
import torch


def load_excel(path: str, sheet=None, no_header=False) -> pd.DataFrame:
    """Load Excel; if multiple sheets and sheet not specified, use the first one"""
    header = None if no_header else 0
    df = pd.read_excel(path, sheet_name=sheet, header=header, engine="openpyxl")
    if isinstance(df, dict):  # multiple sheets
        first_key = list(df.keys())[0]
        print(f"[Info] Multiple sheets detected, using the first one: {first_key}")
        df = df[first_key]
    return df


def parse_cols_arg(spec: str | None):
    """Parse column argument into list; convert pure digits to int, keep others as string"""
    if not spec:
        return None
    parts = [p.strip() for p in spec.split(",") if p.strip() != ""]
    cols = [int(p) if re.fullmatch(r"\d+", p) else p for p in parts]
    return cols


def select_columns(df: pd.DataFrame, cols):
    """If cols are all int -> use iloc; else use loc"""
    if cols is None:
        raise ValueError("cols is None")
    if all(isinstance(c, int) for c in cols):
        return df.iloc[:, cols]
    return df.loc[:, cols]


def find_columns_auto(df: pd.DataFrame):
    """
    Auto-detect x/y/z/thickness column names (case-insensitive, allowing units, spaces, etc.)
    Examples: 'x', 'x (mm)', 'X[mm]', ' shell thickness (mm) '
    """
    norm = {c: re.sub(r"\s+", " ", str(c)).strip().lower() for c in df.columns}

    def match_one(keys, patterns):
        for c, n in norm.items():
            for p in patterns:
                if re.search(p, n):
                    keys.append(c)
                    return

    xyz = []
    match_one(xyz, [r"(^|\W)x(\W|$)"])
    match_one(xyz, [r"(^|\W)y(\W|$)"])
    match_one(xyz, [r"(^|\W)z(\W|$)"])

    thick_col = None
    for c, n in norm.items():
        if re.search(r"thick|thickness", n):
            thick_col = c
            break

    return xyz if len(xyz) == 3 else None, thick_col


def main():
    parser = argparse.ArgumentParser(
        description="Convert Excel (x,y,z[,thickness]) to .pt with {'pos': (N,3), 'x': (N,1)}"
    )
    parser.add_argument("excel", help="Input Excel file path (.xlsx/.xls)")
    parser.add_argument("out_pt", help="Output .pt file path")
    parser.add_argument("--sheet", default=None, help="Sheet name or index (default: first)")
    parser.add_argument("--no-header", action="store_true", help="Excel has no header, first row is data")
    parser.add_argument(
        "--xyz-cols",
        default=None,
        help="Indices or names for x,y,z columns, comma separated. E.g. '0,1,2' or 'x (mm),y (mm),z (mm)'",
    )
    parser.add_argument(
        "--thick-col",
        default=None,
        help="Index or name of thickness column. E.g. '3' or 'shell thickness (mm)'. Default: auto-detect or use 1",
    )
    parser.add_argument(
        "--drop-nan-xyz",
        action="store_true",
        help="Drop rows where x/y/z contain NaN (recommended)",
    )
    args = parser.parse_args()

    # Load Excel
    try:
        sheet = None
        if args.sheet is not None:
            sheet = int(args.sheet) if re.fullmatch(r"\d+", str(args.sheet)) else args.sheet
        df = load_excel(args.excel, sheet=sheet, no_header=args.no_header)
    except Exception as e:
        print(f"[Error] Failed to load Excel: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(df, pd.DataFrame):
        print("[Error] Could not load DataFrame.", file=sys.stderr)
        sys.exit(1)

    # Handle no-header case: assume first 3 cols are xyz, 4th is thickness if exists
    if args.no_header:
        if args.xyz_cols is None:
            args.xyz_cols = "0,1,2"
        if args.thick_col is None and df.shape[1] >= 4:
            args.thick_col = "3"

    # Parse user-specified columns
    xyz_cols_arg = parse_cols_arg(args.xyz_cols)
    thick_col_arg = None if args.thick_col is None else (int(args.thick_col) if re.fullmatch(r"\d+", str(args.thick_col)) else args.thick_col)

    # Auto-detect columns
    if xyz_cols_arg is None:
        auto_xyz, auto_thick = find_columns_auto(df)
        if auto_xyz is None:
            print("[Error] Could not auto-detect x,y,z columns. Use --xyz-cols to specify.", file=sys.stderr)
            print("        Available columns:", list(df.columns), file=sys.stderr)
            sys.exit(1)
        xyz_cols = auto_xyz
        print(f"[Info] Auto-detected xyz columns: {xyz_cols}")
    else:
        xyz_cols = xyz_cols_arg

    if thick_col_arg is None:
        _, auto_thick = find_columns_auto(df)
        if auto_thick is not None:
            thick_col = auto_thick
            print(f"[Info] Auto-detected thickness column: {thick_col}")
        else:
            thick_col = None
            print("[Info] No thickness column found, using default value 1 for all.")
    else:
        thick_col = thick_col_arg

    # Select xyz
    try:
        xyz_df = select_columns(df, xyz_cols)
    except Exception:
        print("[Error] Failed to select x,y,z columns. Available:", list(df.columns), file=sys.stderr)
        sys.exit(1)

    # Convert to numpy
    xyz = xyz_df.to_numpy(dtype=np.float32, copy=True)

    # Handle NaN in xyz
    if args.drop_nan_xyz:
        mask_valid = ~np.isnan(xyz).any(axis=1)
        dropped = int((~mask_valid).sum())
        if dropped > 0:
            print(f"[Info] Dropped {dropped} rows with NaN in x/y/z")
        xyz = xyz[mask_valid]
        df_effective = df.loc[xyz_df.index[mask_valid]]
    else:
        if np.isnan(xyz).any():
            print("[Info] NaN detected in x/y/z, replaced with 0. Use --drop-nan-xyz for strict removal.")
            xyz = np.nan_to_num(xyz, nan=0.0)
        df_effective = df

    # Thickness
    if thick_col is None:
        thickness_arr = np.ones((xyz.shape[0],), dtype=np.float32)
    else:
        try:
            if isinstance(thick_col, int):
                thick_series = df_effective.iloc[:, thick_col]
            else:
                thick_series = df_effective.loc[:, thick_col]
        except Exception:
            print("[Info] Specified thickness column not found, using default 1.")
            thick_series = None

        if thick_series is None:
            thickness_arr = np.ones((xyz.shape[0],), dtype=np.float32)
        else:
            thickness_arr = thick_series.to_numpy(dtype=np.float32, copy=True)
            # Adjust length after drop
            if len(thickness_arr) != xyz.shape[0]:
                try:
                    if isinstance(thick_col, int):
                        if len(thickness_arr) > xyz.shape[0]:
                            thickness_arr = thickness_arr[: xyz.shape[0]]
                        else:
                            pad = np.ones((xyz.shape[0] - len(thickness_arr),), dtype=np.float32)
                            thickness_arr = np.concatenate([thickness_arr, pad], axis=0)
                    else:
                        thick_series = df_effective.loc[:, thick_col]
                        thickness_arr = thick_series.to_numpy(dtype=np.float32, copy=True)
                except Exception:
                    pass
            # Replace NaN with 1
            nan_count = int(np.isnan(thickness_arr).sum())
            if nan_count > 0:
                print(f"[Info] {nan_count} NaN values in thickness, replaced with 1.")
                thickness_arr = np.nan_to_num(thickness_arr, nan=1.0)

    # Reshape thickness -> (N,1)
    thickness_arr = thickness_arr.reshape(-1, 1)

    # Convert to torch.Tensor
    pos = torch.from_numpy(xyz.astype(np.float32, copy=False))
    x = torch.from_numpy(thickness_arr.astype(np.float32, copy=False))

    # Final object
    obj = {"pos": pos, "x": x}

    # Save
    out_dir = os.path.dirname(os.path.abspath(args.out_pt))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    try:
        torch.save(obj, args.out_pt)
    except Exception as e:
        print(f"[Error] Failed to save .pt: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[Done] Saved to: {args.out_pt}")
    print(f" - pos: shape={tuple(pos.shape)}, dtype={pos.dtype}")
    print(f" - x(thickness): shape={tuple(x.shape)}, dtype={x.dtype}")


if __name__ == "__main__":
    main()
