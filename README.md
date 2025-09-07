# mesh-format-converter
A lightweight toolkit for converting forming simulation data (XLSX, STEP) into unified HDF5 / PyTorch formats, with visualization and animation utilities.
## A collection of scripts to convert and visualize forming mesh data:
- **h5final.py** — Build unified HDF5 from XLSX + STEP
- **xlsx2h5.py** — Convert XLSX files to HDF5 groups
- **xlsx2pt.py** — Convert XLSX to PyTorch .pt (deduplicated quads)
- **step2pt.py** — Mesh STEP with Gmsh, export to PyTorch .pt
- **step2h5.py** — Mesh STEP with Gmsh, export to HDF5 groups
- **h5plot.py** — Animate deformations (GIF/MP4/PNG)
- **h5_shape.py** — Inspect attributes and dataset shapes
- **plot_pt.py** — Render .pt mesh to PNG (thickness coloring)
- **pt2png.py** — Quick 3D preview from .pt
- **png2gif.py** — Make GIF from PNG sequence

## [h5nal.py](/scr/h5final.py)

`h5final.py` is a utility script for converting **XLSX (blank sheet data)** and **STEP (tool geometry)** files into a structured **HDF5 (.h5)** file.


## h5_shape.py

A simple Python CLI tool to explore **HDF5 (.h5)** files.  
It prints out **file attributes, group attributes, and dataset shapes**.  
Additionally, if a dataset or attribute named `Parameters` contains JSON, it will be parsed and displayed in a human-readable format.

---

## Features
- Show **all groups, datasets, and attributes** in an `.h5` file.
- Print dataset **shapes** and **dtypes**.
- Automatically decode `numpy` scalars and `bytes` into Python values.
- Pretty-print attributes, and try to expand `Parameters` if it is JSON.

---