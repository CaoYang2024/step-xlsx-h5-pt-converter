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
---
## [h5final.py](/scr/h5final.py)

`h5final.py` is a utility script for converting **XLSX (blank sheet data)** and **STEP (tool geometry)** files into a structured **HDF5 (.h5)** file.
### Usage
#### A. Full pipeline (XLSX + STEP → HDF5) (example)
```bash
python h5final.py \
  --out outputs/full_build.h5 \
  --xlsx-dir data/blank_xlsx \
  --die-step data/die.step \
  --binder-step data/binder.step \
  --punch-step data/punch.step \
  --size 5.0 --quad \
  --tag-from tool_radii2_50_radii1_5_cr_1.1_delta_0_height_25
```
#### Only convert STEP
```bash
python h5final.py \
  --out outputs/blank_only.h5 \
  --xlsx-dir data/blank_xlsx
  ```

#### Only convert STEP
```bash
python h5final.py \
  --out outputs/tools_only.h5 \
  --die-step data/die.step \
  --size 4.0 --quad
  ```
---
## [h5_shape.py](/scr/h5_shape)

A simple Python CLI tool to explore **HDF5 (.h5)** files.  
It prints out **file attributes, group attributes, and dataset shapes**.  
Additionally, if a dataset or attribute named `Parameters` contains JSON, it will be parsed and displayed in a human-readable format.

---
### Usage
``` bash
python h5show.py your_file.h5
```
---

## [h5plot](/scr/h5plot.py)

This script visualizes **FE shell meshes** stored in an HDF5 file and can:

- Render a static deformed frame (PNG).

- Create a time-series animation (GIF/MP4) using node_displacement.

- Optionally color faces by thickness (element_shell_thickness) or effective plastic strain (element_shell_effective_plastic_strain).

It expects a group like OP10/blank containing the usual datasets:

- node_coordinates (N,3)

- element_shell_node_indexes (E,4) or element_shell_node_ids (E,4)

- node_displacement (T,N,3) (required for animation)

-  element_shell_thickness (E,) or (T,E),

- element_shell_effective_plastic_strain (T,E,K).
