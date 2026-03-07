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

if __name__ == '__main__':

    # Create assembly with electrode
    print("Creating electrode assembly...")
    assembly = PyElectrodeAssembly("Field Solver Boundaries")

    # Create a simple geometry: cylindrical electrode
    geo_str = """
    SetFactory("OpenCASCADE");
    Mesh.CharacteristicLengthMax = 0.025;

    // Cylinder with conical end-caps
    Cylinder(1) = {0, 0, -0.08, 0, 0, 0.06, 0.06, 2*Pi};
    Cone(2) = {0, 0, -0.08, 0, 0, 0.03, 0.06, 0.02, 2*Pi};
    Cone(3) = {0, 0, -0.02, 0, 0, 0.03, 0.02, 0.06, 2*Pi};

    s() = Surface "*";
    Physical Surface(100) = {s()};
    """

    electrode = PyElectrode(name="RFQ-like Geometry", voltage=1000.0, geo_str=geo_str)
    electrode.generate_mesh(brep_h=0.025)
    assembly.add_electrode(electrode)

    print(f"Assembly ready with {len(assembly.electrodes)} electrode(s)")

    assembly.show(show_screen=True)


    # Create a coarse mesh grid for field solver
    print("Creating field mesh nodes...")
    n_x, n_y, n_z = 10, 10, 12

    x = np.linspace(-0.15, 0.15, n_x)
    y = np.linspace(-0.15, 0.15, n_y)
    z = np.linspace(-0.12, 0.08, n_z)

    mesh_nodes = np.array(np.meshgrid(x, y, z, indexing='ij')).reshape(3, -1).T
    mesh_nodes = mesh_nodes.astype(np.float32)

    print(f"Field mesh: {n_x}×{n_y}×{n_z} = {len(mesh_nodes)} nodes")

    # Compute axis-aligned surface intersections
    print("Computing boundary intersections (axis-aligned rays)...")
    try:
        intersections = assembly.compute_axis_aligned_surface_intersections(
            mesh_nodes, axes='all', use_gpu=False, chunk_size=None
        )
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")
        raise

    # Count boundary nodes in each direction
    print("\nBoundary intersection summary:")
    print("-" * 40)
    for axis in ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']:
        n_hits = np.sum(intersections[axis]['hit_mask'])
        print(f"{axis:3s}: {n_hits:3d} / {len(mesh_nodes)} nodes")

    # Visualization
    fig = plt.figure(figsize=(16, 6))

    # Plot 1: All mesh nodes colored by boundary hits
    ax1 = fig.add_subplot(131, projection='3d')

    # Determine which nodes have boundary hits
    has_boundary = np.zeros(len(mesh_nodes), dtype=bool)
    for axis in ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']:
        has_boundary |= intersections[axis]['hit_mask']

    # Plot interior nodes
    interior_mask = ~has_boundary
    if np.any(interior_mask):
        ax1.scatter(mesh_nodes[interior_mask, 0],
                    mesh_nodes[interior_mask, 1],
                    mesh_nodes[interior_mask, 2],
                    c='blue', s=30, alpha=0.4, label='Interior')

    # Plot boundary nodes
    if np.any(has_boundary):
        ax1.scatter(mesh_nodes[has_boundary, 0],
                    mesh_nodes[has_boundary, 1],
                    mesh_nodes[has_boundary, 2],
                    c='red', s=50, alpha=0.8, label='Boundary')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Field Mesh Nodes\n(Interior vs. Boundary)')
    ax1.legend()

    # Plot 2: X-direction intersections
    ax2 = fig.add_subplot(132, projection='3d')

    x_plus_hits = intersections['x+']['hit_mask']
    x_minus_hits = intersections['x-']['hit_mask']

    # Plot nodes without x-intersections
    no_x_hits = ~(x_plus_hits | x_minus_hits)
    if np.any(no_x_hits):
        ax2.scatter(mesh_nodes[no_x_hits, 0],
                    mesh_nodes[no_x_hits, 1],
                    mesh_nodes[no_x_hits, 2],
                    c='blue', s=30, alpha=0.3)

    # Plot x+ intersections
    if np.any(x_plus_hits):
        ax2.scatter(mesh_nodes[x_plus_hits, 0],
                    mesh_nodes[x_plus_hits, 1],
                    mesh_nodes[x_plus_hits, 2],
                    c='red', s=60, marker='^', label='+x hits')

        # Draw rays to intersection points
        for i in np.where(x_plus_hits)[0]:
            ax2.plot([mesh_nodes[i, 0], intersections['x+']['hit_points'][i, 0]],
                     [mesh_nodes[i, 1], intersections['x+']['hit_points'][i, 1]],
                     [mesh_nodes[i, 2], intersections['x+']['hit_points'][i, 2]],
                     'r--', alpha=0.3, linewidth=0.5)

    # Plot x- intersections
    if np.any(x_minus_hits):
        ax2.scatter(mesh_nodes[x_minus_hits, 0],
                    mesh_nodes[x_minus_hits, 1],
                    mesh_nodes[x_minus_hits, 2],
                    c='green', s=60, marker='v', label='-x hits')

        for i in np.where(x_minus_hits)[0]:
            ax2.plot([mesh_nodes[i, 0], intersections['x-']['hit_points'][i, 0]],
                     [mesh_nodes[i, 1], intersections['x-']['hit_points'][i, 1]],
                     [mesh_nodes[i, 2], intersections['x-']['hit_points'][i, 2]],
                     'g--', alpha=0.3, linewidth=0.5)

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('X-Direction Intersections\n(±x rays)')
    ax2.legend()

    # Plot 3: Distance distribution
    ax3 = fig.add_subplot(133)

    all_distances = []
    all_labels = []

    for axis in ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']:
        dists = intersections[axis]['distances'][intersections[axis]['hit_mask']]
        if len(dists) > 0:
            all_distances.append(dists)
            all_labels.append(axis)

    ax3.boxplot(all_distances, labels=all_labels)
    ax3.set_ylabel('Distance to surface (m)')
    ax3.set_title('Distance Distribution\n(by ray direction)')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig('boundary_conditions_example.png', dpi=150)
    print("\nSaved: boundary_conditions_example.png")
    plt.show()

    # Optional: Print detailed boundary info for one node
    print("\n" + "=" * 50)
    print("Example: First boundary node details")
    print("=" * 50)

    boundary_indices = np.where(has_boundary)[0]
    if len(boundary_indices) > 0:
        idx = boundary_indices[0]
        print(f"\nNode index: {idx}")
        print(f"Position: {mesh_nodes[idx]}")
        print(f"\nIntersections in each direction:")

        for axis in ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']:
            if intersections[axis]['hit_mask'][idx]:
                hit_pt = intersections[axis]['hit_points'][idx]
                dist = intersections[axis]['distances'][idx]
                print(f"  {axis}: distance={dist:.4f} m, point={hit_pt}")
            else:
                print(f"  {axis}: no intersection")