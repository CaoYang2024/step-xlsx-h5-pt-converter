import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import ticker
from ddacs import DDACSDataset
from ddacs.utils import extract_point_cloud, extract_mesh, display_structure, extract_element_thickness

# Setup data directory
data_dir = Path("/home/RUS_CIP/st186635/format_transformate/")

# Load dataset
dataset = DDACSDataset(data_dir, "h5")
print(f"Loaded {len(dataset)} simulations")

# Get a sample simulation
sim_id, metadata, h5_path = dataset[200]
print(f"Sample simulation: {sim_id}")
print(f"File: {h5_path}")
print(f"Metadata: {metadata}")

# Load and analyze metadata
df = dataset._metadata
print(f"Metadata structure:")
print(f"  Total simulations: {len(df)}")
print(f"  Columns: {list(df.columns)}")
print("  > Note: The column 'ID' is not returned in dataset.")

# Show parameter meanings and ranges
print(f"\nParameters (physical values):")

# Note: Update these when you have the actual unnormalized data
desc = dataset.get_metadata_descriptions()
for key in desc:
    print(f"  > {key:4} - {desc[key]}")

print(f"\nSample simulations:")
print(df.head())

# Explore the HDF5 simulation structure
with h5py.File(h5_path, 'r') as f:
    print("Operations:", list(f.keys()))

    # OP10: Forming operation
    print("\nOP10 - Forming Operation:")
    op10 = f['OP10']
    print("  Components:", list(op10.keys()))

    # Analyze each component
    print("\nComponent Details:")
    for component in ['blank', 'die', 'binder', 'punch']:
        if component in op10:
            comp = op10[component]
            nodes = comp['node_coordinates'].shape[0]

            if 'node_displacement' in comp:
                disp_shape = comp['node_displacement'].shape
                timesteps = disp_shape[0] if len(disp_shape) == 3 else 1
                print(f"  {component}: {nodes} nodes, {timesteps} timesteps")
            else:
                print(f"  {component}: {nodes} nodes, no displacement")

    # OP20: Springback operation (if exists)
    if 'OP20' in f:
        print("\nOP20 - Springback Operation:")
        op20 = f['OP20']
        print("  Components:", list(op20.keys()))
        if 'blank' in op20:
            springback_shape = op20['blank']['node_displacement'].shape
            print(f"  Springback timesteps: {springback_shape[0]}")

# For whole content use display_structure
# display_structure(h5_path)

try:
    import scienceplots
    plt.style.use(['science'])

except ImportError as exc:
    print("For proper plotting, use scienceplots")

# Standard figure sizes (width, height) in inches
FIGURE_SIZES = {
    'single_col': (3.5, 2.6),      # Single column
    'single_col_cb': (5, 3.5),      # Single column, with colorbar
    'single_col_tall': (3.5, 3.5), # Single column, square-ish
    'double_col': (7.0, 3.0),      # Double column, wide
    'double_col_tall': (7.0, 4.5), # Double column, taller
    'square': (3.5, 3.5),          # Square single column
    'poster': (10, 8),             # For presentations/posters
}

# Visualization constants for consistency
FIGURE_SIZE = FIGURE_SIZES['double_col']
FIGURE_DPI = 150
AXIS_LIMITS = [0, 150]
VIEW_ELEVATION = 30
VIEW_AZIMUTH = 45
TIMESTEP_MAX_FORMING = 3

# Component settings
COMPONENT_COLORS = {'blank': 'red', 'die': 'blue', 'punch': 'green', 'binder': 'orange'}
COMPONENT_NAMES = {'blank': 'Blank (Workpiece)', 'die': 'Die (Lower Tool)',
                   'punch': 'Punch (Upper Tool)', 'binder': 'Binder (Clamp)'}

# Visualize complete forming setup
fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
ax = fig.add_subplot(111, projection='3d')

for component in COMPONENT_COLORS.keys():
    try:
        coords = extract_point_cloud(h5_path, component, timestep=TIMESTEP_MAX_FORMING)
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                  c=COMPONENT_COLORS[component], label=COMPONENT_NAMES[component],
                  s=0.5, alpha=0.7)
        print(f"{component}: {coords.shape[0]} nodes")
    except Exception as e:
        print(f"Could not load {component}: {e}")

ax.set_title(f'Deep Drawing Setup - Simulation {sim_id} - Timestep {TIMESTEP_MAX_FORMING}')
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')
ax.set_xlim(AXIS_LIMITS)
ax.set_ylim(AXIS_LIMITS)
ax.set_zlim(AXIS_LIMITS)
ax.view_init(VIEW_ELEVATION, VIEW_AZIMUTH)
ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
plt.savefig(f"{"picture"}.png")

# Mesh visualization of blank component
fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
ax = fig.add_subplot(111, projection='3d')

vertices, triangles = extract_mesh(h5_path, 'blank', timestep=TIMESTEP_MAX_FORMING)
faces = vertices[triangles]

mesh = Poly3DCollection(faces, alpha=0.8, facecolor=COMPONENT_COLORS['blank'],
                       edgecolor='black', linewidth=0.1)
ax.add_collection3d(mesh)

print(f"Mesh: {vertices.shape[0]} vertices, {triangles.shape[0]} faces")

ax.set_title(f'Blank Mesh - Simulation {sim_id} - Timestep {TIMESTEP_MAX_FORMING}')
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')
ax.set_xlim(AXIS_LIMITS)
ax.set_ylim(AXIS_LIMITS)
ax.set_zlim(AXIS_LIMITS)
ax.view_init(VIEW_ELEVATION, VIEW_AZIMUTH)
plt.tight_layout()
plt.show()
plt.savefig(f"{"blank"}.png")



# Thickness visualization
timestep = TIMESTEP_MAX_FORMING
vertices, triangles = extract_mesh(h5_path, 'blank', timestep=timestep)
thickness = extract_element_thickness(h5_path, 'blank', timestep=timestep)
# Map element data to triangles
thickness_per_triangle = np.repeat(thickness, 2)[:len(triangles)]

fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
ax = fig.add_subplot(111, projection='3d')

faces = vertices[triangles]
norm_thickness = (thickness_per_triangle - thickness.min()) / (thickness.max() + 0.00001 - thickness.min())
colors = plt.cm.viridis(norm_thickness)
collection = Poly3DCollection(faces, facecolors=colors, edgecolors=colors, alpha=1)
ax.add_collection3d(collection)
ax.plot_surface
# Add colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0.85, 1.15))
cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
cbar.set_label('Thickness [mm]')

ax.set_title(f'Thickness Distribution - Simulation {sim_id} - Timestep {timestep}')
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')
ax.set_xlim(AXIS_LIMITS)
ax.set_ylim(AXIS_LIMITS)
ax.set_zlim(AXIS_LIMITS)
ax.view_init(VIEW_ELEVATION, VIEW_AZIMUTH)
plt.tight_layout()
plt.savefig(f"{timestep}.png")
plt.show()
print(f"Thickness range: {thickness.min():.4f} - {thickness.max():.4f} mm")