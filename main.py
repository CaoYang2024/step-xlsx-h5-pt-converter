#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch


def load_excel(path: str, sheet: str | int | None = None) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)

    # 如果返回的是字典，取第一个工作表
    if isinstance(df, dict):
        first_key = list(df.keys())[0]
        df = df[first_key]

    df = df.dropna(how="all")  # 删除全空行

    # 判断是否是纯数字且无列名语义
    has_text_header = any(isinstance(c, str) for c in df.columns)

    # 尝试基于列名匹配
    colmap = {}
    if has_text_header:
        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        # 可能的列名别称
        x_key = next((k for k in ["x", "xs", "pos_x"] if k in cols_lower), None)
        y_key = next((k for k in ["y", "ys", "pos_y"] if k in cols_lower), None)
        z_key = next((k for k in ["z", "zs", "pos_z"] if k in cols_lower), None)
        t_key = next((k for k in ["thickness", "t", "d", "厚度"] if k in cols_lower), None)

        if x_key and y_key and z_key:
            colmap["x"] = cols_lower[x_key]
            colmap["y"] = cols_lower[y_key]
            colmap["z"] = cols_lower[z_key]
            if t_key:
                colmap["thickness"] = cols_lower[t_key]
            df2 = df[[colmap["x"], colmap["y"], colmap["z"]]].copy()
            if "thickness" in colmap:
                df2["thickness"] = df[colmap["thickness"]]
            return df2.rename(columns={
                colmap["x"]: "x",
                colmap["y"]: "y",
                colmap["z"]: "z",
                colmap.get("thickness", "thickness"): "thickness",
            })

    # 若无可用表头/匹配失败，按列序取：前3列为x,y,z，第4列为厚度（若存在）
    # 只保留前4列，避免后续无关列干扰
    base = df.iloc[:, :4].copy()

    # 保证至少有前三列
    if base.shape[1] < 3:
        raise ValueError(f"Excel 列数不足，需要至少3列(x,y,z)，当前只有 {base.shape[1]} 列。")

    base.columns = ["x", "y", "z"] + (["thickness"] if base.shape[1] >= 4 else [])
    return base

def to_torch_data(df: pd.DataFrame):
    from torch_geometric.data import Data

    # 过滤掉 x,y,z 中存在 NaN 的行
    df = df.dropna(subset=["x", "y", "z"])
    # 转 float32
    pos = df[["x", "y", "z"]].to_numpy(dtype=np.float32)

    data = Data()
    data.pos = torch.from_numpy(pos)

    # 可选厚度列：若存在且非全空，就作为 node feature 存到 data.x (N,1)
    if "thickness" in df.columns:
        # 如果厚度这一列全是 NaN 或空，忽略
        if not df["thickness"].dropna().empty:
            thickness = df["thickness"].fillna(0).to_numpy(dtype=np.float32).reshape(-1, 1)
            if thickness.shape[0] != pos.shape[0]:
                # 理论上不会发生，因为我们只按有效行取 pos
                thickness = thickness[:pos.shape[0]]
            data.x = torch.from_numpy(thickness)

    return data

def main():
    parser = argparse.ArgumentParser(
        description="将 Excel 点云数据(x,y,z[,thickness]) 转为 .pt (torch_geometric Data)"
    )
    parser.add_argument("excel", help="输入 Excel 路径，例如: /path/to/points.xlsx")
    parser.add_argument("-o", "--out", required=True, help="输出 .pt 路径，例如: /path/to/output.pt")
    parser.add_argument("--sheet", help="工作表名或索引(0-based)，默认第一个", default=None)
    args = parser.parse_args()

    sheet = None
    if args.sheet is not None:
        try:
            sheet = int(args.sheet)
        except ValueError:
            sheet = args.sheet  # 字符串表名

    if not os.path.isfile(args.excel):
        print(f"找不到输入文件: {args.excel}", file=sys.stderr)
        sys.exit(1)

    try:
        df = load_excel(args.excel, sheet)
        data = to_torch_data(df)
        # 保存为 .pt
        torch.save(data, args.out)
        n = data.pos.shape[0]
        has_th = hasattr(data, "x") and data.x is not None
        print(f"已保存: {args.out}")
        print(f"点数: {n}, 厚度列: {'有' if has_th else '无'}")
    except Exception as e:
        print(f"转换失败: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
