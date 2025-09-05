#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import warnings
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch


# —— 工具函数 ————————————————————————————————————————————————————————————
def normalize_colname(name: str) -> str:
    """标准化列名：小写、去空白、把中文括号替换为英文括号、去掉括号内单位等。"""
    s = (name or "").strip().lower()
    # 全角中文括号 -> 英文
    s = s.replace("（", "(").replace("）", ")")
    # 去掉括号（单位）
    if "(" in s and ")" in s:
        try:
            l = s.index("(")
            r = s.index(")", l + 1)
            s = (s[:l] + s[r + 1:]).strip()
        except Exception:
            pass
    # 常见分隔符去掉
    for ch in [" ", "\t", "\n", "\r", ":", "；", ";", "，", ","]:
        s = s.replace(ch, "")
    return s


def find_columns(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    """匹配 x,y,z,t(thickness) 列；支持表头如 x（mm）/x(mm)/thickness 等。"""
    norm_map: Dict[str, str] = {normalize_colname(c): c for c in df.columns}

    # 允许的别名
    x_candidates = ["x"]
    y_candidates = ["y"]
    z_candidates = ["z"]
    t_candidates = ["t", "thickness", "thick", "thk"]

    def pick(cands: List[str]) -> str:
        for c in cands:
            if c in norm_map:
                return norm_map[c]
        # 兼容形如 'xmm'、'xcoordinate'（开头包含）
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
            warnings.warn("未准确匹配到列名，按前四列作为 x,y,z,t 兜底。")
            col_x, col_y, col_z, col_t = df.columns[:4]
        else:
            raise ValueError("Excel 至少需要 4 列（x,y,z,t/thickness）。")
    return col_x, col_y, col_z, col_t


def quantize(arr: np.ndarray, decimals: int) -> np.ndarray:
    """按小数位数量化（用于去重容忍浮点误差）"""
    return np.round(arr.astype(np.float64), decimals=decimals)


def dedup_points(points: np.ndarray, decimals: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    点去重：返回 (unique_points, inverse_indices)
    - unique_points: (M,3)
    - inverse_indices: (N,) 使得 unique_points[inverse_indices] == quantized(points)
    """
    q = quantize(points, decimals)
    dtype = np.dtype((np.void, q.dtype.itemsize * q.shape[1]))
    q_view = np.ascontiguousarray(q).view(dtype)
    _, idx_unique, inv = np.unique(q_view, return_index=True, return_inverse=True)
    unique_points = q[idx_unique]
    return unique_points.astype(np.float32), inv.astype(np.int64)


def build_quads(points: np.ndarray, tvals: np.ndarray, decimals_t: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    将每 4 行组成 1 个网格，厚度只取该组内的“第一个非空/非 NaN 值”：
    返回 (quads_points:(G,4,3), quads_t:(G,))
    """
    n = points.shape[0]
    if n < 4:
        raise ValueError("数据行数不足 4，无法组成网格。")
    if n % 4 != 0:
        warnings.warn(f"总行数 {n} 不是 4 的倍数，多出的 {n % 4} 行将被丢弃。")
        n = n - (n % 4)
        points = points[:n]
        tvals = tvals[:n]

    G = n // 4
    quads_points = points.reshape(G, 4, 3).copy()
    quads_t = np.zeros((G,), dtype=np.float32)

    for g in range(G):
        t_group = tvals[g * 4:(g + 1) * 4]
        # 厚度取该组第一个非 NaN 值；如果都没有则报错
        t_val = np.nan
        for tv in t_group:
            if tv is not None and not (isinstance(tv, float) and np.isnan(tv)):
                t_val = float(tv)
                break
        if np.isnan(t_val):
            raise ValueError(f"第 {g+1} 组未找到 thickness（四行均为空/NaN）。")
        quads_t[g] = t_val

    quads_t = quantize(quads_t, decimals_t).astype(np.float32)
    return quads_points.astype(np.float32), quads_t


def indices_from_unique(points_unique: np.ndarray, quads_points: np.ndarray, decimals_xyz: int) -> np.ndarray:
    """将 quad 中的 4 个坐标映射到唯一点表索引，返回 quads_idx:(G,4)。"""
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
                raise RuntimeError(f"找不到点 {key} 对应的唯一索引，请检查 --decimals_xyz 设置。")
            quads_idx[g, k] = key_to_idx[key]
    return quads_idx


def dedup_quads(quads_idx: np.ndarray, quads_t: np.ndarray, decimals_t: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    网格去重：忽略 4 点顺序；t 量化后作为 key 的一部分。
    返回去重后的 (faces:(M,4), t:(M,))
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


# —— 主流程 ——————————————————————————————————————————————————————————————
def main():
    ap = argparse.ArgumentParser(description="Convert Excel (x,y,z,thickness per 4 points) to .pt with de-duplicated points & quads.")
    ap.add_argument("excel", type=Path, help="输入 Excel 路径（.xlsx/.xls）")
    ap.add_argument("out_pt", type=Path, help="输出 .pt 路径")
    ap.add_argument("--sheet", type=str, default=None, help="工作表名（默认第一个）")
    ap.add_argument("--decimals_xyz", type=int, default=6, help="点坐标量化小数位（用于点去重）")
    ap.add_argument("--decimals_t", type=int, default=6, help="厚度量化小数位（用于网格去重）")
    ap.add_argument("--unit_scale", type=float, default=1.0, help="单位缩放（例如 mm->m 可设为 0.001）")
    args = ap.parse_args()

    # 读取 Excel
    df = pd.read_excel(args.excel, sheet_name=args.sheet) if args.sheet else pd.read_excel(args.excel)

    col_x, col_y, col_z, col_t = find_columns(df)
    xyz = df[[col_x, col_y, col_z]].to_numpy(dtype=np.float64)
    tvals = df[[col_t]].to_numpy(dtype=np.float64).reshape(-1)

    # 单位换算（若表头是 mm，默认 unit_scale=1.0；需要转米可传 0.001）
    if args.unit_scale != 1.0:
        xyz = xyz * float(args.unit_scale)

    # 每 4 行 -> 1 个网格；厚度取组内第一个非 NaN
    quads_points, quads_t = build_quads(xyz, tvals, args.decimals_t)

    # 点去重
    all_points = quads_points.reshape(-1, 3)
    points_unique, _ = dedup_points(all_points, args.decimals_xyz)

    # 建立网格索引
    quads_idx = indices_from_unique(points_unique, quads_points, args.decimals_xyz)

    # 网格去重
    faces, t_unique = dedup_quads(quads_idx, quads_t, args.decimals_t)

    # 组织并保存
    out_obj = {
        "pos": points_unique.astype(np.float32),          # (M,3)
        "faces": faces.astype(np.int64),                  # (N,4)
        # 方便下游直接用 (ID1, ID2, ID3, ID4, t)：
        "faces_t_float": np.hstack([faces.astype(np.float32),
                                    t_unique.reshape(-1, 1)]).astype(np.float32)  # (N,5)
    }
    torch.save(out_obj, args.out_pt)

    print("==== Done ====")
    print(f"输入文件: {args.excel}")
    print(f"唯一点数 M: {out_obj['pos'].shape[0]}")
    print(f"去重后网格数 N: {out_obj['faces'].shape[0]}")
    print(f".pt 已保存: {args.out_pt}")


if __name__ == "__main__":
    main()
