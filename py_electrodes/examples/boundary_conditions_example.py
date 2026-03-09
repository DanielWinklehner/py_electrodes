"""
Example 3: Axis-Aligned Boundary Conditions for Field Solver
Computes ray-surface intersections in ±x, ±y, ±z directions.
Useful for PIC field solvers with conductor boundaries.
Visualizes mesh nodes with boundary intersections marked.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from py_electrodes.py_electrodes import PyElectrode, PyElectrodeAssembly
import matplotlib.patches as patches

if __name__ == '__main__':

    # Create assembly with electrode
    print("Creating electrode assembly...")
    assembly = PyElectrodeAssembly("Field Solver Boundaries")

    # Create a simple geometry: cylindrical electrode
    geo_str = """SetFactory("OpenCASCADE");
Geometry.NumSubEdges = 100; // nicer display of curve
Mesh.CharacteristicLengthMax = 0.005;  // maximum mesh size

    // Cylinder
    Cylinder(1) = {0, 0, -0.08, 0, 0, 0.16, 0.06, 2 * Pi};
    """

    electrode = PyElectrode(name="RFQ-like Geometry", voltage=1000.0, geo_str=geo_str)
    electrode.generate_mesh(brep_h=0.025)
    assembly.add_electrode(electrode)

    print(f"Assembly ready with {len(assembly.electrodes)} electrode(s)")

    assembly.show(show_screen=True)

    # Create a coarse mesh grid for field solver
    # Nevermind centers of cells, this is just to test if ray-surface
    # intersections along the exes work
    print("Creating field mesh nodes...")
    n_x, n_y, n_z = 13, 13, 13

    x = np.linspace(-0.15, 0.15, n_x)
    y = np.linspace(-0.15, 0.15, n_y)
    z = np.linspace(-0.15, 0.15, n_z)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    mesh_nodes = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    print(f"Field mesh: {n_x}x{n_y}x{n_z} = {len(mesh_nodes)} nodes")

    # Compute axis-aligned surface intersections
    print("Computing boundary intersections (axis-aligned rays)...")
    try:
        intersections = assembly.compute_axis_aligned_surface_intersections(
            mesh_nodes, axes='all', use_gpu=True)
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")
        raise

    # Analyze and classify nodes
    print("\n" + "=" * 50)
    print("Analyzing mesh node classification...")
    print("=" * 50)

    inside_mask = np.zeros(len(mesh_nodes), dtype=bool)
    boundary_mask = np.zeros(len(mesh_nodes), dtype=bool)
    outside_mask = np.ones(len(mesh_nodes), dtype=bool)

    # Mesh size for boundary detection
    mesh_sizes = [x[1] - x[0], x[1] - x[0],
                  y[1] - y[0], y[1] - y[0],
                  z[1] - z[0], z[1] - z[0]]

    directions = ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']


    for node_idx, node_data in enumerate(intersections):
        # Count intersections per direction per electrode
        hit_counts = {}
        min_distance_to_surface = np.inf

        for dir_idx, direction in enumerate(directions):
            dir_data = node_data[direction]

            for elec_id, elec_data in dir_data.items():
                if elec_data:
                    if len(elec_data["distances"])%2 != 0:
                        inside_mask[node_idx] = True
                        outside_mask[node_idx] = False
                        boundary_mask[node_idx] = False
                    elif np.any(elec_data["distances"] <= mesh_sizes[dir_idx]):
                        boundary_mask[node_idx] = True
                        outside_mask[node_idx] = False
                        inside_mask[node_idx] = False

    n_inside = np.sum(inside_mask)
    n_boundary = np.sum(boundary_mask)
    n_outside = np.sum(outside_mask)
    n_total = len(mesh_nodes)

    print(f"\nNode classification:")
    print(f"  Inside:   {n_inside:4d} ({100 * n_inside / n_total:5.1f}%)")
    print(f"  Boundary: {n_boundary:4d} ({100 * n_boundary / n_total:5.1f}%)")
    print(f"  Outside:  {n_outside:4d} ({100 * n_outside / n_total:5.1f}%)")
    print(f"  Total:    {n_total:4d}")

    # Visualization
    fig = plt.figure(figsize=(14, 12))

    # Colors for classification
    colors = np.zeros(n_total)
    colors[inside_mask] = 0  # Red
    colors[boundary_mask] = 1  # Orange
    colors[outside_mask] = 2  # Blue

    color_map = {0: 'red', 1: 'orange', 2: 'blue'}
    labels = {0: 'Inside', 1: 'Boundary', 2: 'Outside'}

    # Plot 1: 3D view
    ax1 = fig.add_subplot(221, projection='3d')

    for class_id in [0, 1, 2]:
        mask = colors == class_id
        ax1.scatter(mesh_nodes[mask, 0], mesh_nodes[mask, 1], mesh_nodes[mask, 2],
                    c=color_map[class_id], s=30, alpha=0.6, label=labels[class_id])

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Node Classification')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(-0.15, 0.15)
    ax1.set_ylim(-0.15, 0.15)
    ax1.set_zlim(-0.15, 0.15)

    # Find middle indices
    mid_x_idx = n_x // 2
    mid_y_idx = n_y // 2
    mid_z_idx = n_z // 2

    # Plot 2: XY slice (at middle z)
    ax2 = fig.add_subplot(222)

    z_slice_mask = np.abs(mesh_nodes[:, 2] - z[mid_z_idx]) < 1e-6
    slice_indices = np.where(z_slice_mask)[0]

    for class_id in [0, 1, 2]:
        mask = colors[slice_indices] == class_id
        ax2.scatter(mesh_nodes[slice_indices[mask], 0], mesh_nodes[slice_indices[mask], 1],
                    c=color_map[class_id], s=30, alpha=0.6, label=labels[class_id])

    circle = patches.Circle((0, 0), radius=0.06, fill=False,
                            edgecolor='green', linewidth=2, linestyle='--', label='Electrode')
    ax2.add_patch(circle)

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'XY Slice (z={z[mid_z_idx]:.4f})')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # Plot 3: XZ slice (at middle y)
    ax3 = fig.add_subplot(223)

    y_slice_mask = np.abs(mesh_nodes[:, 1] - y[mid_y_idx]) < 1e-6
    slice_indices = np.where(y_slice_mask)[0]

    for class_id in [0, 1, 2]:
        mask = colors[slice_indices] == class_id
        ax3.scatter(mesh_nodes[slice_indices[mask], 0], mesh_nodes[slice_indices[mask], 2],
                    c=color_map[class_id], s=30, alpha=0.6, label=labels[class_id])

    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title(f'XZ Slice (y={y[mid_y_idx]:.4f})')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    rect_xz = patches.Rectangle((-0.06, -0.08), 0.12, 0.16,
                                fill=False, edgecolor='green',
                                linewidth=2, linestyle='--', label='Electrode')
    ax3.add_patch(rect_xz)

    # Plot 4: YZ slice (at middle x)
    ax4 = fig.add_subplot(224)

    x_slice_mask = np.abs(mesh_nodes[:, 0] - x[mid_x_idx]) < 1e-6
    slice_indices = np.where(x_slice_mask)[0]

    for class_id in [0, 1, 2]:
        mask = colors[slice_indices] == class_id
        ax4.scatter(mesh_nodes[slice_indices[mask], 1], mesh_nodes[slice_indices[mask], 2],
                    c=color_map[class_id], s=30, alpha=0.6, label=labels[class_id])

    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title(f'YZ Slice (x={x[mid_x_idx]:.4f})')
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')

    rect_yz = patches.Rectangle((-0.06, -0.08), 0.12, 0.16,
                                fill=False, edgecolor='green',
                                linewidth=2, linestyle='--', label='Electrode')
    ax4.add_patch(rect_yz)

    plt.tight_layout()
    plt.savefig('boundary_conditions_example.png', dpi=150)
    print("\nSaved: boundary_conditions_example.png")
    plt.show()