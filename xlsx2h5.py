#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import h5py

def guess_col(df, keys, fallback=None):
    """在 df 列名里模糊匹配 keys（列表），返回第一个匹配到的列名。"""
    cols = list(df.columns)
    pattern = re.compile("|".join([re.escape(k) for k in keys]), re.IGNORECASE)
    for c in cols:
        if pattern.search(str(c)):
            return c
    return fallback

def load_points_from_xlsx(xlsx_path, colx=None, coly=None, colz=None, colt=None):
    df = pd.read_excel(xlsx_path)

    # 自动猜列名
    cx = colx or guess_col(df, ["x", "x (mm)"])
    cy = coly or guess_col(df, ["y", "y (mm)"])
    cz = colz or guess_col(df, ["z", "z (mm)"])
    ct = colt or guess_col(df, ["thick", "thickness", "t (mm)", "dicke"])

    missing = [n for n in [cx, cy, cz, ct] if n is None]
    if missing:
        raise ValueError(f"[{xlsx_path}] 无法识别列名，请用参数指定 colx/coly/colz/colt。DataFrame columns={list(df.columns)}")

    # 取值为 float
    x = pd.to_numeric(df[cx], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[cy], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(df[cz], errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(df[ct], errors="coerce").to_numpy(dtype=float)

    # 去掉任何含 NaN 的行（保持 1:1 对齐）
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z, t = x[mask], y[mask], z[mask], t[mask]

    # 点数必须是 4 的倍数
    if len(x) % 4 != 0:
        raise ValueError(f"[{xlsx_path}] 点的数量 {len(x)} 不是 4 的整数倍，无法每 4 点构单元。")

    coords = np.stack([x, y, z], axis=1).astype(np.float64)  # (N,3)
    return coords, t

def build_mesh_from_points(coords, thickness_vec, thickness_source="first"):
    """
    coords: (N,3), N%4==0
    thickness_vec: (N,), 可能只有每 4 个点里有一个非空值，其它为 NaN/空
    thickness_source: 'first' | 'last'  取每组 4 点中的第1个或第4个点厚度
    """
    N = coords.shape[0]
    if N % 4 != 0:
        raise ValueError("coords 总数必须是 4 的整数倍。")

    # node_ids 从 1 开始
    node_ids = np.arange(1, N + 1, dtype=np.int64)

    # 每 4 点构成一个四边形：元素数 M
    M = N // 4
    # element_shell_ids 从 1 开始
    elem_ids = np.arange(1, M + 1, dtype=np.int64)

    # element_shell_node_ids: 每行 [n1, n2, n3, n4]
    elem_nodes = np.arange(N, dtype=np.int64).reshape(M, 4) + 1  # 1-based

    # 厚度：默认取每组 4 点的第 1 个点或第 4 个点
    if thickness_source == "first":
        pick_idx = 0
    elif thickness_source == "last":
        pick_idx = 3
    else:
        raise ValueError("thickness_source 只能是 'first' 或 'last'")

    # 先把 thickness_vec 统一为长度 N 的数组（缺失用 NaN）
    t = np.array(thickness_vec, dtype=float)
    if t.shape[0] != N:
        # 如果用户给的是每4个一个值（长度 M），也兼容一下：
        if t.shape[0] == M:
            # 直接视为每个元素一个厚度
            elem_thick = t.astype(np.float64)
            return node_ids, coords.astype(np.float64), elem_ids, elem_nodes, elem_thick
        else:
            raise ValueError(f"thickness 长度({t.shape[0]})既不是 N({N}) 也不是 M({M})。")

    # 抽取每组的 pick_idx 个点的厚度，若为 NaN 则尝试在该组内寻找非 NaN 的一个
    elem_thick = np.empty((M,), dtype=np.float64)
    for i in range(M):
        group = t[i*4:(i+1)*4]
        # 首选 pick
        val = group[pick_idx]
        if np.isnan(val):
            # 组内找第一个非 NaN 的值
            valid = group[~np.isnan(group)]
            if len(valid) > 0:
                val = valid[0]
            else:
                # 都是 NaN 就置 0，并给个提醒
                val = 0.0
        elem_thick[i] = float(val)

    return node_ids, coords.astype(np.float64), elem_ids, elem_nodes, elem_thick

def write_group(h5: h5py.File, group_name: str,
                node_ids, node_coords, elem_ids, elem_nodes, elem_thick):
    g = h5.require_group(group_name)
    # 覆盖写（如果已有则先删再写）
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
    ap = argparse.ArgumentParser(description="Pack multiple XLSX (x,y,z,thickness) into HDF5 with required datasets.")
    ap.add_argument("xlsx", nargs="+", help="XLSX files")
    ap.add_argument("--out", required=True, help="Output HDF5 path")
    ap.add_argument("--single-group", default=None, help="Put all data into this single group (e.g., 'blank'). If omitted, each file uses its stem as group name.")
    ap.add_argument("--colx", default=None, help="Name of X column")
    ap.add_argument("--coly", default=None, help="Name of Y column")
    ap.add_argument("--colz", default=None, help="Name of Z column")
    ap.add_argument("--colt", default=None, help="Name of thickness column")
    ap.add_argument("--thickness-source", choices=["first", "last"], default="first",
                    help="Pick thickness from the 1st or 4th point in each group of 4 when per-point thickness is sparse.")
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

            if args.single_group:
                gname = args.single_group
            else:
                gname = xlsx_path.stem  # 文件名做组名

            write_group(h5, gname, node_ids, node_coords, elem_ids, elem_nodes, elem_thick)
            print(f"[OK] {xlsx_path.name} → group '{gname}' "
                  f"(nodes={len(node_ids)}, elements={len(elem_ids)})")

    print(f"[Done] Saved to {out_path}")

if __name__ == "__main__":
    main()
