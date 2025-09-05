#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
h5final.py
一次性生成:
  <out>.h5
  ├── blank/           # 每个 xlsx 一个子组: blank/<xlsx_stem>/
  ├── die/             # 来自 STEP
  ├── binder/
  └── punch/

blank/<xlsx_stem>/ 写入:
  node_ids (N,) int64
  node_coordinates (N,3) float64
  element_shell_ids (M,) int64
  element_shell_node_ids (M,4) int64    # 1-based
  element_shell_thickness (M,) 或 (t,M) float64

die/binder/punch 写入:
  node_ids (N,) int64
  node_coordinates (N,3) float64
  element_shell_node_indexes (M,4) int64  # 0-based
  element_shell_node_ids (M,4) int64      # 1-based
  element_shell_ids (M,) int64

文件属性（file attributes）写入:
  radii2, radii1, delta, cr, height (标量)
  Parameters (JSON 字符串，包含上述五个键)
  source_tag (原始目录/标签名)

依赖:
  pip install numpy pandas h5py openpyxl gmsh
"""

import argparse
import re
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import h5py


# --------- XLSX -> blank/<stem> ---------
def _guess_col(df: pd.DataFrame, keys) -> Optional[str]:
    pat = re.compile("|".join([re.escape(k) for k in keys]), re.IGNORECASE)
    for c in df.columns:
        if pat.search(str(c)):
            return c
    return None


def load_points_thickness_from_xlsx(
    xlsx_path: Path,
    colx: Optional[str] = None,
    coly: Optional[str] = None,
    colz: Optional[str] = None,
    colt: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_excel(xlsx_path)
    cx = colx or _guess_col(df, ["x", "x (mm)"])
    cy = coly or _guess_col(df, ["y", "y (mm)"])
    cz = colz or _guess_col(df, ["z", "z (mm)"])
    ct = colt or _guess_col(df, ["thick", "thickness", "t (mm)", "dicke"])
    if not all([cx, cy, cz, ct]):
        raise ValueError(f"[{xlsx_path}] Cannot infer x/y/z/thickness columns; got {list(df.columns)}")

    x = pd.to_numeric(df[cx], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[cy], errors="coerce").to_numpy(float)
    z = pd.to_numeric(df[cz], errors="coerce").to_numpy(float)
    t = pd.to_numeric(df[ct], errors="coerce").to_numpy(float)

    # 过滤掉坐标 NaN 的行（保持 t 对齐）
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z, t = x[mask], y[mask], z[mask], t[mask]

    if x.size % 4 != 0:
        raise ValueError(f"[{xlsx_path}] number of points {x.size} is not multiple of 4.")

    coords = np.stack([x, y, z], axis=1).astype(np.float64)  # (N,3)
    return coords, t


def build_quads_from_points(
    coords: np.ndarray,
    thickness_vec: np.ndarray,
    thickness_source: str = "first",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    每 4 点组成一个四边形元素。
    thickness_vec 支持长度 N(逐点, 稀疏亦可) 或 M(逐元素)，M=N//4。
    thickness_source: 'first' 或 'last' —— 从每组4点取第1或第4个点厚度；若该点为 NaN 则回退到组内第一个非 NaN；组内都 NaN 则置 0。
    """
    N = coords.shape[0]
    if N % 4 != 0:
        raise ValueError("coords length must be multiple of 4.")
    node_ids = np.arange(1, N + 1, dtype=np.int64)
    M = N // 4
    elem_ids = np.arange(1, M + 1, dtype=np.int64)
    elem_node_ids = np.arange(N, dtype=np.int64).reshape(M, 4) + 1  # 1-based

    t = np.asarray(thickness_vec, dtype=float)
    if t.shape[0] == M:
        elem_thick = t.astype(np.float64)
    else:
        if t.shape[0] != N:
            raise ValueError(f"thickness length {t.shape[0]} not M({M}) or N({N}).")
        pick = 0 if thickness_source == "first" else 3
        elem_thick = np.empty((M,), dtype=np.float64)
        for i in range(M):
            group = t[i * 4 : (i + 1) * 4]
            v = group[pick]
            if np.isnan(v):
                valid = group[~np.isnan(group)]
                v = float(valid[0]) if valid.size else 0.0
            elem_thick[i] = float(v)

    return node_ids, coords.astype(np.float64), elem_ids, elem_node_ids.astype(np.int64), elem_thick


def write_blank_xlsx_group(
    h5: h5py.File,
    blank_base: str,
    xlsx: Path,
    node_ids: np.ndarray,
    node_coords: np.ndarray,
    elem_ids: np.ndarray,
    elem_node_ids: np.ndarray,
    elem_thick: np.ndarray,
) -> None:
    g = h5.require_group(f"{blank_base}/{xlsx.stem}")
    for name in ["node_ids", "node_coordinates", "element_shell_ids", "element_shell_node_ids", "element_shell_thickness"]:
        if name in g:
            del g[name]
    g.create_dataset("node_ids", data=node_ids)
    g.create_dataset("node_coordinates", data=node_coords)
    g.create_dataset("element_shell_ids", data=elem_ids)
    g.create_dataset("element_shell_node_ids", data=elem_node_ids)
    g.create_dataset("element_shell_thickness", data=elem_thick)


# --------- STEP -> tools/* ---------
def gmsh_surface_mesh(
    step_path: Path, target_size: Optional[float], prefer_quads: bool
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    用 gmsh 对 STEP 做 2D 表面网格。
    返回:
      coords: (N,3) float64，被2D单元使用的节点坐标
      pack:   dict{ 'quads':(Mq,4), 'tris':(Mt,3), 'used_tags':(N,) }  —— 节点标签为 Gmsh NodeTag
    """
    import gmsh  # 延迟导入，便于报错时提示安装

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    try:
        gmsh.open(str(step_path))

        # 网格尺寸
        if target_size is not None and target_size > 0:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", target_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_size)

        # 优先生成四边（做版本兼容）
        if prefer_quads:
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
            try:
                ents = gmsh.model.getEntities(2)
            except Exception:
                ents = []
            for (dim, tag) in ents:
                try:
                    gmsh.model.mesh.recombine(dim, tag)  # 常见签名
                except TypeError:
                    try:
                        gmsh.model.mesh.setRecombine(dim, tag, True)  # 某些版本
                    except Exception:
                        pass
                except Exception:
                    pass

        gmsh.model.mesh.generate(2)

        types, elemTagsList, nodeTagsList = gmsh.model.mesh.getElements(2)
        quads = np.zeros((0, 4), dtype=np.int64)
        tris = np.zeros((0, 3), dtype=np.int64)
        for et, etags, ntags in zip(types, elemTagsList, nodeTagsList):
            _, _, _, nPerElem, _, _ = gmsh.model.mesh.getElementProperties(et)
            conn = np.array(ntags, dtype=np.int64).reshape(-1, nPerElem)
            if nPerElem == 4:
                quads = np.vstack([quads, conn])
            elif nPerElem == 3:
                tris = np.vstack([tris, conn])

        used_tags = []
        if quads.size:
            used_tags.append(quads.reshape(-1))
        if tris.size:
            used_tags.append(tris.reshape(-1))
        used_tags = np.unique(np.concatenate(used_tags)) if used_tags else np.array([], dtype=np.int64)

        all_tags, all_coords, _ = gmsh.model.mesh.getNodes()
        all_tags = np.array(all_tags, dtype=np.int64)
        all_coords = np.array(all_coords, dtype=float).reshape(-1, 3)

        tag_to_idx = {int(t): i for i, t in enumerate(all_tags)}
        coords = np.empty((len(used_tags), 3), dtype=np.float64)
        for i, t in enumerate(used_tags):
            coords[i] = all_coords[tag_to_idx[int(t)]]

        pack = {"quads": quads, "tris": tris, "used_tags": used_tags}
        return coords, pack
    finally:
        gmsh.finalize()


def build_elem_indexes(
    used_tags: np.ndarray, quads: np.ndarray, tris: np.ndarray, tri_policy: str = "pad_last"
) -> np.ndarray:
    """
    把 Gmsh 的 NodeTag 映射成 0-based 索引连通性。
    tri_policy:
      - pad_last: 三角 [i,j,k] -> [i,j,k,k]
      - skip    : 跳过三角
    """
    tag_to_zero = {int(t): i for i, t in enumerate(used_tags)}
    parts: List[np.ndarray] = []
    if quads.size:
        q = np.vectorize(lambda t: tag_to_zero[int(t)], otypes=[np.int64])(quads)
        parts.append(q.astype(np.int64))
    if tris.size:
        if tri_policy == "pad_last":
            t = np.vectorize(lambda v: tag_to_zero[int(v)], otypes=[np.int64])(tris)
            t = np.column_stack([t, t[:, 2:3]]).astype(np.int64)
            parts.append(t)
        elif tri_policy == "skip":
            pass
        else:
            raise ValueError("tri_policy must be 'pad_last' or 'skip'")
    return np.vstack(parts) if parts else np.zeros((0, 4), dtype=np.int64)


def write_tool_group(h5: h5py.File, tool_group: str, node_coords: np.ndarray, elem_node_indexes: np.ndarray) -> None:
    g = h5.require_group(tool_group)
    for name in ["node_ids", "node_coordinates", "element_shell_node_indexes", "element_shell_node_ids", "element_shell_ids"]:
        if name in g:
            del g[name]

    node_ids = np.arange(1, node_coords.shape[0] + 1, dtype=np.int64)
    g.create_dataset("node_ids", data=node_ids)
    g.create_dataset("node_coordinates", data=node_coords.astype(np.float64))
    g.create_dataset("element_shell_node_indexes", data=elem_node_indexes.astype(np.int64))
    g.create_dataset("element_shell_node_ids", data=(elem_node_indexes + 1).astype(np.int64))
    g.create_dataset("element_shell_ids", data=np.arange(1, elem_node_indexes.shape[0] + 1, dtype=np.int64))


# --------- H5 attributes（从标签解析） ---------
def parse_sim_tag(name: str) -> Dict[str, float]:
    """
    从目录名或文件名解析 radii2/radii1/cr/delta/height，作为 H5 属性存储。
    例: tool_radii2_50_radii1_5_cr_1.1_delta_0_height_25
    """
    kv: Dict[str, float] = {}
    pats = {
        "radii2": r"radii2_([\-0-9]+(?:\.[0-9]+)?)",
        "radii1": r"radii1_([\-0-9]+(?:\.[0-9]+)?)",
        "cr": r"cr_([\-0-9]+(?:\.[0-9]+)?)",
        "delta": r"delta_([\-0-9]+(?:\.[0-9]+)?)",
        "height": r"height_([\-0-9]+(?:\.[0-9]+)?)",
    }
    for k, pat in pats.items():
        m = re.search(pat, name)
        if m:
            try:
                kv[k] = float(m.group(1))
            except Exception:
                pass
    return kv


# --------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Build one HDF5 with blank/ + die/ + binder/ + punch/ from XLSX & STEP.")
    ap.add_argument("--out", required=True, help="Output HDF5 path.")
    ap.add_argument("--xlsx-dir", required=False, help="Directory containing .xlsx for blank/")
    ap.add_argument("--recursive", action="store_true", help="Search XLSX recursively")
    ap.add_argument("--colx", default=None)
    ap.add_argument("--coly", default=None)
    ap.add_argument("--colz", default=None)
    ap.add_argument("--colt", default=None)
    ap.add_argument(
        "--thickness-source",
        choices=["first", "last"],
        default="first",
        help="Pick thickness from the 1st or 4th point in each group of 4 when per-point thickness is sparse.",
    )
    ap.add_argument("--die-step", default=None)
    ap.add_argument("--binder-step", default=None)
    ap.add_argument("--punch-step", default=None)
    ap.add_argument("--size", type=float, default=None, help="Target mesh size for STEP meshing")
    ap.add_argument("--quad", action="store_true", help="Prefer quads for STEP meshing")
    ap.add_argument("--tri-policy", choices=["pad_last", "skip"], default="pad_last")
    ap.add_argument(
        "--tag-from",
        default=None,
        help="Optional: parse parameters (radii2/radii1/delta/cr/height) from this path's name and store to H5 attrs.",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "a") as h5:
        # === 文件属性：单值属性 + 合并 JSON 的 'Parameters' ===
        if args.tag_from:
            tag_name = Path(args.tag_from).name
            attrs = parse_sim_tag(tag_name)  # {'radii2':..., 'radii1':..., 'delta':..., 'cr':..., 'height':...}
            attrs["source_tag"] = tag_name

            # 单个标量属性（若存在）
            for k in ("radii2", "radii1", "delta", "cr", "height"):
                v = attrs.get(k, None)
                if v is not None:
                    h5.attrs[k] = float(v)

            # 合并 JSON 属性：Parameters
            params_payload = {k: attrs.get(k, None) for k in ("radii2", "radii1", "delta", "cr", "height")}
            h5.attrs["Parameters"] = json.dumps(params_payload, ensure_ascii=False)

            # 附带记录原始标签
            h5.attrs["source_tag"] = tag_name

        # ---- XLSX -> blank/<stem> ----
        if args.xlsx_dir:
            base = Path(args.xlsx_dir)
            files = list(base.rglob("*.xlsx") if args.recursive else base.glob("*.xlsx"))
            files.sort()
            for xp in files:
                try:
                    coords, tvec = load_points_thickness_from_xlsx(xp, args.colx, args.coly, args.colz, args.colt)
                    node_ids, node_coords, elem_ids, elem_node_ids, elem_thick = build_quads_from_points(
                        coords, tvec, thickness_source=args.thickness_source
                    )
                    write_blank_xlsx_group(h5, "blank", xp, node_ids, node_coords, elem_ids, elem_node_ids, elem_thick)
                    print(f"[blank] {xp.name}: nodes={len(node_ids)}, elems={len(elem_ids)}")
                except Exception as e:
                    print(f"[blank][Skip] {xp} ({e})")

        # ---- STEP -> tools/* ----
        def handle_step(step_path: Optional[str], group_name: str) -> None:
            if not step_path:
                return
            sp = Path(step_path)
            if not sp.exists():
                print(f"[tools][Skip] {group_name}: not found {sp}")
                return
            try:
                coords, pack = gmsh_surface_mesh(sp, target_size=args.size, prefer_quads=args.quad)
            except ModuleNotFoundError:
                raise SystemExit("gmsh Python API not found. Please run: pip install gmsh")
            elem_idx = build_elem_indexes(pack["used_tags"], pack["quads"], pack["tris"], tri_policy=args.tri_policy)
            write_tool_group(h5, group_name, coords, elem_idx)
            print(f"[tools] {group_name}: nodes={coords.shape[0]}, elems={elem_idx.shape[0]}")

        handle_step(args.die_step, "die")
        handle_step(args.binder_step, "binder")
        handle_step(args.punch_step, "punch")

    print(f"[Done] H5 -> {out_path}")


if __name__ == "__main__":
    main()
