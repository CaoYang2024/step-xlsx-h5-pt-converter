#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py, argparse, json, numpy as np

def _to_py(v):
    # Convert numpy scalars/bytes into readable Python values
    if isinstance(v, bytes):
        try:
            return v.decode('utf-8')
        except:
            return v
    if isinstance(v, np.ndarray) and v.shape == ():
        v = v[()]  # scalar array
    if isinstance(v, np.generic):
        return v.item()
    return v

def _print_attrs(obj, prefix=""):
    if len(obj.attrs) == 0:
        return
    print(f"{prefix}@attrs:")
    for k in sorted(obj.attrs.keys()):
        v = _to_py(obj.attrs[k])
        # Try to parse "Parameters" as JSON for better readability
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
        # Top-level file attributes
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
    ap = argparse.ArgumentParser(description="Show HDF5 file attributes and dataset shapes")
    ap.add_argument("h5file", help="Path to input .h5 file")
    ap.add_argument("--attrs-only", action="store_true", help="Show only attributes without dataset shapes")
    args = ap.parse_args()
    show_all(args.h5file, attrs_only=args.attrs_only)
