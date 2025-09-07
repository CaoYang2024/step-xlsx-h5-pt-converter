# mesh-format-converter
A lightweight toolkit for converting forming simulation data (XLSX, STEP) into unified HDF5 / PyTorch formats, with visualization and animation utilities.
![output of bash_pt2gif](/outputs/animation.gif)
## A collection of scripts to convert and visualize forming mesh data:
- **h5final.py** — Build unified HDF5 from XLSX + STEP
- **xlsx2h5.py** — Convert XLSX files to HDF5 groups
- **xlsx2pt.py** — Convert XLSX to PyTorch .pt (deduplicated quads)
- **step2pt.py** — Mesh STEP with Gmsh, export to PyTorch .pt
- **step2h5.py** — Mesh STEP with Gmsh, export to HDF5 groups
- **h5plot.py** — Animate deformations (GIF/MP4/PNG)
- **h5_shape.py** — Inspect attributes and dataset shapes
- **pt2png.py** — Quick 3D preview from .pt
- **png2gif.py** — Make GIF from PNG sequence
---
## Batch Data Conversion Pipeline
- **bash_h5final.sh** — Batch convert simulation folders into a unified HDF5 (.h5) format using `h5final.py`
- **bash_pt2gif.sh** — End-to-end pipeline from Excel to visualization:
**Excel → PT → PNG → GIF**.
- **bash_step2pt_xlsx2pt.sh** — Batch convert Excel (.xlsx/.xls) and STEP (.step) files to PyTorch tensors (.pt).
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
---
It expects a group like OP10/blank containing the usual datasets:

- node_coordinates (N,3)
- element_shell_node_indexes (E,4) or element_shell_node_ids (E,4)
- node_displacement (T,N,3) (required for animation)
-  element_shell_thickness (E,) or (T,E),
- element_shell_effective_plastic_strain (T,E,K).
---
### Usage
#### 1) Animate deformation
```bash 
python h5_anim.py \
  path/to/data.h5 \
  --group OP10/blank \
  --scale 1.0 \
  --fps 12 \
  --skip 2 \
  --color thickness \
  --edge
  ```
#### 2) Static last frame
```bash
python h5_anim.py path/to/data.h5 \
  --group OP10/blank \
  --scale 1.2 \
  --last \
  --color epsp \
  --reduce max \
  --edge \
  --out last_frame.png
  ```
---
## [pt2png.py](/scr/pt2png.py)
This script loads a mesh stored in a PyTorch .pt file and renders it as a 3D figure.
It supports:

- Vertex positions (pos, shape (N,3))

- Quad faces (faces, shape (F,4))

- Optional thickness values (faces_t_float, if available) for per-face coloring

The result is saved as a high-resolution PNG image.

---

Usage 
---
```bash
python plot_mesh.py input.pt --png output.png
```
---
## [png2gif.py](/scr/png2gif.py)
This script combines a sequence of PNG images into an animated GIF.
It supports:

- Input as either a directory of PNG files or a list of files

- Automatic natural sorting by numeric suffix (e.g., Tiefgezogenes Bauteil_15000.png)

- Automatic resizing to the largest canvas so all frames fit consistently

- Optional timeline annotation (numbers extracted from filenames)

- Configurable frame rate (fps) and loop count
---

Usage 
---
```bash
python make_gif.py <inputs> --gif <output.gif> [options]
```

Arguments
---
inputs

- A single directory containing PNGs matching Tiefgezogenes Bauteil_<number>.png

- Or one or more explicit PNG file paths

`--gif <path>` → Output GIF file path (required)

`--fps <int>`→ Frames per second (default: 5)

`--loop <int>` → Loop count (default: 0 = infinite)

`--no-annotate` → Disable timeline annotation overlay
