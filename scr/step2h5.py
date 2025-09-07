#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import h5py

# Gmsh Python API
import gmsh


def generate_surface_mesh_from_step(step_path: Path,
                                    target_size: float = None,
                                    prefer_quads: bool = True
                                    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate a surface mesh from a STEP file using Gmsh.
    Returns:
      coords: (N,3) float64 — node coordinates (only nodes used by 2D elements)
      elems:  dict containing:
          'quads': (M4, 4) int64 — node tags (Gmsh NodeTag, not indices)
          'tris' : (M3, 3) int64 — same as above
          'used_node_tags': node tags actually used
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    # Open STEP geometry
    gmsh.open(str(step_path))

    # Global mesh size (optional)
    if target_size is not None and target_size > 0:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", target_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_size)

    # Prefer generating quads
    if prefer_quads:
        # Recombine triangles into quads where possible
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        # Slight quad optimization
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        # Apply recombination to all surfaces
        for (dim, tag) in gmsh.model.getEntities(2):
            gmsh.model.mesh.recombine(dim, tag)

    # Generate 2D (surface) mesh
    gmsh.model.mesh.generate(2)

    # Get all 2D elements
    types, elemTagsList, nodeTagsList = gmsh.model.mesh.getElements(2)

    quads = np.zeros((0, 4), dtype=np.int64)
    tris  = np.zeros((0, 3), dtype=np.int64)

    for et, etags, ntags in zip(types, elemTagsList, nodeTagsList):
        # Query number of nodes per element for this type
        _, _, _, nPerElem, _, _ = gmsh.model.mesh.getElementProperties(et)
        conn = np.array(ntags, dtype=np.int64).reshape(-1, nPerElem)
        if nPerElem == 4:
            quads = np.vstack([quads, conn])
        elif nPerElem == 3:
            tris = np.vstack([tris, conn])
        # Ignore other element dimensions (lines, volumes, etc.)

    # Collect node tags used by 2D elements
    used_tags = []
    if quads.size:
        used_tags.append(quads.reshape(-1))
    if tris.size:
        used_tags.append(tris.reshape(-1))
    if used_tags:
        used_tags = np.unique(np.concatenate(used_tags))
    else:
        used_tags = np.array([], dtype=np.int64)

    # Extract node coordinates (only keep used_tags)
    # getNodes() returns (nodeTags, coords, _)
    all_tags, all_coords, _ = gmsh.model.mesh.getNodes()
    all_tags = np.array(all_tags, dtype=np.int64)
    all_coords = np.array(all_coords, dtype=float).reshape(-1, 3)

    # Build mapping: tag -> coordinate
    tag_to_idx = {int(t): i for i, t in enumerate(all_tags)}
    coords = np.empty((len(used_tags), 3), dtype=np.float64)
    for i, t in enumerate(used_tags):
        coords[i] = all_coords[tag_to_idx[int(t)]]

    gmsh.finalize()

    elems = {"quads": quads, "tris": tris, "used_node_tags": used_tags}
    return coords, elems


def build_connectivity_indexes(used_node_tags: np.ndarray,
                               elems: Dict[str, np.ndarray],
                               tri_policy: str = "pad_last"
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map Gmsh node tags (not necessarily contiguous, start from 1) to 0-based indices.
    Returns:
      elem_node_indexes: (M,4) int64 — 0-based connectivity
      elem_ids:          (M,)   int64 — element IDs starting from 1

    If only triangles exist, handle according to tri_policy:
      - 'pad_last': [i,j,k] -> [i,j,k,k]
      - 'skip'    : skip triangles (only keep quads)
    """
    # tag -> 0-based index
    tag_to_zero = {int(t): i for i, t in enumerate(used_node_tags)}

    quads = elems.get("quads", np.zeros((0, 4), dtype=np.int64))
    tris  = elems.get("tris",  np.zeros((0, 3), dtype=np.int64))

    parts: List[np.ndarray] = []
    if quads.size:
        q_idx = np.vectorize(lambda t: tag_to_zero[int(t)], otypes=[np.int64])(quads)
        parts.append(q_idx.astype(np.int64))

    if tris.size:
        if tri_policy == "pad_last":
            # [i,j,k] → [i,j,k,k]
            t_idx = np.vectorize(lambda t: tag_to_zero[int(t)], otypes=[np.int64])(tris)
            t_pad = np.column_stack([t_idx, t_idx[:, 2:3]]).astype(np.int64)
            parts.append(t_pad)
        elif tri_policy == "skip":
            pass
        else:
            raise ValueError("tri-policy must be 'pad_last' or 'skip'.")

    if parts:
        elem_node_indexes = np.vstack(parts)
    else:
        elem_node_indexes = np.zeros((0, 4), dtype=np.int64)

    elem_ids = np.arange(1, elem_node_indexes.shape[0] + 1, dtype=np.int64)
    return elem_node_indexes, elem_ids


def write_h5(out_path: Path, group: str,
             node_coords: np.ndarray,
             node_ids_name: str,
             elem_node_indexes: np.ndarray,
             elem_ids: np.ndarray):
    """
    Write mesh data to HDF5 in the specified group.
    Creates:
      - node_ids (or custom dataset name)
      - node_coordinates
      - element_shell_node_indexes
      - element_shell_ids
    """
    with h5py.File(out_path, "a") as h5:
        g = h5.require_group(group)

        # node_ids / node_coordinates
        if node_ids_name in g: del g[node_ids_name]
        if "node_coordinates" in g: del g["node_coordinates"]

        g.create_dataset(node_ids_name, data=np.arange(1, node_coords.shape[0] + 1, dtype=np.int64))
        g.create_dataset("node_coordinates", data=node_coords.astype(np.float64))

        # elements
        if "element_shell_node_indexes" in g: del g["element_shell_node_indexes"]
        if "element_shell_ids" in g: del g["element_shell_ids"]

        g.create_dataset("element_shell_node_indexes", data=elem_node_indexes.astype(np.int64))
        g.create_dataset("element_shell_ids", data=elem_ids.astype(np.int64))


def main():
    ap = argparse.ArgumentParser(description="Convert a STEP file to HDF5 with node & shell element datasets.")
    ap.add_argument("--step", required=True, help="STEP file path (*.step|*.stp)")
    ap.add_argument("--out", required=True, help="Output HDF5 file")
    ap.add_argument("--group", default="tools/die", help="HDF5 group (e.g., tools/die)")
    ap.add_argument("--size", type=float, default=None, help="Target mesh size (CharacteristicLength)")
    ap.add_argument("--quad", action="store_true", help="Try to recombine triangles into quads")
    ap.add_argument("--tri-policy", choices=["pad_last", "skip"], default="pad_last",
                    help="When only triangles present: pad to [i,j,k,k] or skip them.")
    ap.add_argument("--dataset-node-id-name", default="node_ids",
                    help="Dataset name for node IDs (default: node_ids; use 'node_id' if required).")
    args = ap.parse_args()

    step_path = Path(args.step)
    if not step_path.exists():
        raise FileNotFoundError(step_path)

    coords, elems = generate_surface_mesh_from_step(
        step_path,
        target_size=args.size,
        prefer_quads=args.quad
    )

    # Map Gmsh node tags to 0-based connectivity
    elem_node_indexes, elem_ids = build_connectivity_indexes(
        elems["used_node_tags"], elems, tri_policy=args.tri_policy
    )

    write_h5(Path(args.out), args.group, coords, args.dataset_node_id_name,
             elem_node_indexes, elem_ids)

    print(f"[OK] STEP → H5 completed. group='{args.group}', "
          f"nodes={coords.shape[0]}, elems={elem_node_indexes.shape[0]}")


if __name__ == "__main__":
    main()
