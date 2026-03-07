"""
Example 1: Ray-Surface Intersection
Demonstrates finding where rays intersect electrode surfaces.
Visualizes rays and intersection points in 3D.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from py_electrodes.py_electrodes import PyElectrode, PyElectrodeAssembly

if __name__ == '__main__':

    # Create a simple cylindrical electrode
    geo_str = """
    SetFactory("OpenCASCADE");
    Mesh.CharacteristicLengthMax = 0.02;

    // Cylinder: radius 0.1, height 0.2, centered at origin
    Cylinder(1) = {0, 0, -0.1, 0, 0, 0.2, 0.1, 2*Pi};

    s() = Surface "*";
    Physical Surface(100) = {s()};
    """

    # Create electrode
    print("Creating electrode...")
    electrode = PyElectrode(name="Test Cylinder", voltage=1000.0, geo_str=geo_str)

    # Generate mesh
    print("Generating mesh...")
    electrode.generate_mesh(brep_h=0.02)

    # Create test rays from a point cloud, all pointing in different directions
    n_rays = 36
    n_elevations = 3
    total_rays = n_rays * n_elevations

    ray_origins = []
    ray_directions = []

    # Create rays emanating from points at different distances/angles
    for elev in range(n_elevations):
        radius_start = 0.05 + elev * 0.1

        for i in range(n_rays):
            angle = 2 * np.pi * i / n_rays

            # Starting point (on a circle at different radii)
            x = radius_start * np.cos(angle)
            y = radius_start * np.sin(angle)
            z = -0.15 + elev * 0.15

            ray_origins.append([x, y, z])

            # Direction (pointing toward axis)
            dx = -np.cos(angle) * 0.7 + np.random.randn() * 0.1
            dy = -np.sin(angle) * 0.7 + np.random.randn() * 0.1
            dz = np.random.randn() * 0.1

            ray_directions.append([dx, dy, dz])

    ray_origins = np.array(ray_origins, dtype=np.float32)
    ray_directions = np.array(ray_directions, dtype=np.float32)

    # Normalize directions
    ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=1, keepdims=True)

    print(f"Testing {total_rays} rays...")

    # Compute ray-surface intersections
    # Try GPU first, fall back to CPU
    try:
        hit_mask, hit_points, distances = electrode.ray_surface_intersection(
            ray_origins, ray_directions, use_gpu=True
        )
        print("Using GPU acceleration")
    except Exception as e:
        print(f"GPU failed ({e}), using CPU")
        hit_mask, hit_points, distances = electrode.ray_surface_intersection(
            ray_origins, ray_directions, use_gpu=False
        )

    n_hits = np.sum(hit_mask)
    print(f"Rays hitting: {n_hits} / {total_rays} ({100 * n_hits / total_rays:.1f}%)")

    # Visualization
    fig = plt.figure(figsize=(14, 5))

    # Plot 1: 3D view
    ax1 = fig.add_subplot(121, projection='3d')

    # Plot all ray origins
    ax1.scatter(ray_origins[:, 0], ray_origins[:, 1], ray_origins[:, 2],
                c='blue', s=20, alpha=0.6, label='Ray origins')

    # Plot rays (as lines from origin to origin + direction * distance)
    max_plot_dist = 0.5
    for i in range(total_rays):
        end_pt = ray_origins[i] + ray_directions[i] * min(distances[i], max_plot_dist)
        ax1.plot([ray_origins[i, 0], end_pt[0]],
                 [ray_origins[i, 1], end_pt[1]],
                 [ray_origins[i, 2], end_pt[2]],
                 'b-', alpha=0.3, linewidth=0.5)

    # Plot intersection points (only for hits)
    if n_hits > 0:
        ax1.scatter(hit_points[hit_mask, 0],
                    hit_points[hit_mask, 1],
                    hit_points[hit_mask, 2],
                    c='red', s=50, marker='x', label='Intersections')

    # Plot electrode mesh
    mesh_data = __import__('meshio').read(electrode.gmsh_file)
    vertices = mesh_data.points
    for cell_block in mesh_data.cells:
        if cell_block.type == "triangle":
            triangles = cell_block.data
            break

    # Plot triangles
    for tri in triangles[::10]:  # Plot every 10th triangle for clarity
        pts = vertices[tri]
        pts_closed = np.vstack([pts, pts[0]])
        ax1.plot(pts_closed[:, 0], pts_closed[:, 1], pts_closed[:, 2],
                 'g-', alpha=0.3, linewidth=0.5)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Ray-Surface Intersections')
    ax1.legend()
    ax1.set_xlim([-0.3, 0.3])
    ax1.set_ylim([-0.3, 0.3])
    ax1.set_zlim([-0.2, 0.2])

    # Plot 2: Distance histogram
    ax2 = fig.add_subplot(122)
    ax2.hist(distances[hit_mask], bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Distance to surface (m)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Distance Distribution ({n_hits} hits)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ray_intersection_example.png', dpi=150)
    print("Saved: ray_intersection_example.png")
    plt.show()