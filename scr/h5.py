#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py, argparse, json, numpy as np

def _to_py(v):
    # 把 numpy 标量/bytes 转成可读的 Python 值
    if isinstance(v, bytes):
        try: return v.decode('utf-8')
        except: return v
    if isinstance(v, np.ndarray) and v.shape == ():
        v = v[()]  # 标量数组
    if isinstance(v, np.generic):
        return v.item()
    return v

def _print_attrs(obj, prefix=""):
    if len(obj.attrs) == 0:
        return
    print(f"{prefix}@attrs:")
    for k in sorted(obj.attrs.keys()):
        v = _to_py(obj.attrs[k])
        # 尝试把 Parameters 按 JSON 展开显示
        if k == "Parameters" and isinstance(v, (str, bytes)):
            try:
                parsed = json.loads(v if isinstance(v, str) else v.decode("utf-8"))
                print(f"{prefix}  {k} = {parsed} (JSON)")
                continue
            except Exception:
                pass
        print(f"{prefix}  {k} = {v!r}")

def show_all(filename, attrs_only=False):
    with h5py.File(filename, "r") as f:
        print(f"file: {filename}\n" + "="*60)
        # 文件顶层属性
        _print_attrs(f, prefix="/")
        if attrs_only:
            return

        def visitor(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"\n{name}/")
                _print_attrs(obj, prefix=name+"/")
            elif isinstance(obj, h5py.Dataset):
                print(f"\n{name}: shape={obj.shape}, dtype={obj.dtype}")
                _print_attrs(obj, prefix=name)
        f.visititems(visitor)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="查看 HDF5 文件的属性与数据集形状")
    ap.add_argument("h5file", help="输入 .h5 文件路径")
    ap.add_argument("--attrs-only", action="store_true", help="只显示属性，不显示数据集形状")
    args = ap.parse_args()
    show_all(args.h5file, attrs_only=args.attrs_only)
