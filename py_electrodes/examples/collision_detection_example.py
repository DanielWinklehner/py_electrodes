"""
Example 2: Particle Collision Detection
Simulates particle trajectories and detects collisions with electrodes.
Shows particles, their trajectories, collision points, AND electrode geometry.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from py_electrodes.py_electrodes import PyElectrode, PyElectrodeAssembly

if __name__ == '__main__':

    # Create assembly with multiple electrodes
    print("Creating electrode assembly...")
    assembly = PyElectrodeAssembly("Collision Test")

    # Electrode 1: Cylinder (center)
    geo_str_1 = """
    SetFactory("OpenCASCADE");
    Mesh.CharacteristicLengthMax = 0.03;
    Cylinder(1) = {0, 0, -0.05, 0, 0, 0.1, 0.08, 2*Pi};
    s() = Surface "*";
    Physical Surface(100) = {s()};
    """

    elec1 = PyElectrode(name="Central Cylinder", voltage=0.0, geo_str=geo_str_1)
    elec1.generate_mesh(brep_h=0.03)
    assembly.add_electrode(elec1)

    # Electrode 2: Plate (outer wall)
    geo_str_2 = """
    SetFactory("OpenCASCADE");
    Mesh.CharacteristicLengthMax = 0.03;
    Box(2) = {-0.2, -0.2, -0.06, 0.4, 0.4, 0.12};
    s() = Surface "*";
    Physical Surface(200) = {s()};
    """

    elec2 = PyElectrode(name="Outer Box", voltage=1000.0, geo_str=geo_str_2)
    elec2.generate_mesh(brep_h=0.03)
    assembly.add_electrode(elec2)

    print(f"Assembly has {len(assembly.electrodes)} electrodes")

    # Simulate particle trajectories
    n_particles = 50

    # Starting positions: scattered around origin
    positions_start = np.random.randn(n_particles, 3) * 0.1

    # Ending positions: straight-line trajectories in random directions
    velocities = np.random.randn(n_particles, 3)
    velocities = velocities / np.linalg.norm(velocities, axis=1, keepdims=True)

    # Move particles forward by 0.15 meters
    positions_end = positions_start + velocities * 0.15

    print(f"Simulating {n_particles} particle trajectories...")

    # Check for collisions
    collision_data = assembly.segment_intersects_surface(
        positions_start, positions_end, use_gpu=False
    )

    n_collisions = np.sum(collision_data['hit_mask'])
    print(f"Collisions detected: {n_collisions} / {n_particles} ({100 * n_collisions / n_particles:.1f}%)")

    # Visualization
    fig = plt.figure(figsize=(16, 5))

    # Plot 1: 3D trajectories with electrodes
    ax1 = fig.add_subplot(131, projection='3d')

    # PLOT ELECTRODES FIRST (background)
    colors_elec = ['cyan', 'magenta']

    for elec_idx, (elec_uuid, electrode) in enumerate(assembly.electrodes.items()):
        if electrode.gmsh_file is None:
            continue

        try:
            import meshio

            # Load mesh
            mesh_data = meshio.read(electrode.gmsh_file)
            vertices = mesh_data.points

            # Find triangles
            triangles = None
            for cell_block in mesh_data.cells:
                if cell_block.type == "triangle":
                    triangles = cell_block.data
                    break

            if triangles is None:
                continue

            # Plot every nth triangle for clarity (avoid overdraw)
            step = max(1, len(triangles) // 20)

            for tri in triangles[::step]:
                pts = vertices[tri]
                pts_closed = np.vstack([pts, pts[0]])

                ax1.plot(pts_closed[:, 0], pts_closed[:, 1], pts_closed[:, 2],
                         color=colors_elec[elec_idx], alpha=0.2, linewidth=0.5)

            # Fill triangles with semi-transparent surface
            for tri in triangles[::step]:
                pts = vertices[tri]
                ax1.plot_trisurf(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    color=colors_elec[elec_idx], alpha=0.05, linewidth=0
                )

            print(f"  {electrode.name}: {len(triangles)} triangles")

        except Exception as e:
            print(f"  Warning: Could not plot {electrode.name}: {e}")

    # Plot all particles
    for i in range(n_particles):
        if collision_data['hit_mask'][i]:
            # Particle hits wall - plot trajectory to collision point
            ax1.plot([positions_start[i, 0], collision_data['hit_points'][i, 0]],
                     [positions_start[i, 1], collision_data['hit_points'][i, 1]],
                     [positions_start[i, 2], collision_data['hit_points'][i, 2]],
                     'r-', alpha=0.6, linewidth=1.5)
        else:
            # Particle makes it through - plot full trajectory
            ax1.plot([positions_start[i, 0], positions_end[i, 0]],
                     [positions_start[i, 1], positions_end[i, 1]],
                     [positions_start[i, 2], positions_end[i, 2]],
                     'b-', alpha=0.3, linewidth=0.5)

    # Plot collision points
    if n_collisions > 0:
        ax1.scatter(collision_data['hit_points'][collision_data['hit_mask'], 0],
                    collision_data['hit_points'][collision_data['hit_mask'], 1],
                    collision_data['hit_points'][collision_data['hit_mask'], 2],
                    c='red', s=100, marker='x', linewidths=2, label='Collisions')

    # Plot starting positions
    ax1.scatter(positions_start[:, 0], positions_start[:, 1], positions_start[:, 2],
                c='green', s=50, marker='o', alpha=0.7, label='Start positions')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Particle Trajectories, Collisions & Electrodes')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlim([-0.25, 0.25])
    ax1.set_ylim([-0.25, 0.25])
    ax1.set_zlim([-0.1, 0.1])

    # Plot 2: Collision fraction along trajectory
    ax2 = fig.add_subplot(132)
    collision_fractions = collision_data['hit_fractions'][collision_data['hit_mask']]
    ax2.hist(collision_fractions, bins=15, alpha=0.7, edgecolor='black', color='red')
    ax2.set_xlabel('Position along trajectory')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Collision Timing ({n_collisions} hits)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Summary statistics
    ax3 = fig.add_subplot(133)
    ax3.axis('off')

    # Collision statistics by electrode
    stats_text = f"Collision Statistics\n"
    stats_text += f"{'=' * 35}\n\n"
    stats_text += f"Total particles: {n_particles}\n"
    stats_text += f"Collisions: {n_collisions}\n"
    stats_text += f"Hit rate: {100 * n_collisions / n_particles:.1f}%\n\n"
    stats_text += f"By electrode:\n"

    for elec_idx, electrode_uuid in collision_data['electrode_index_map'].items():
        electrode = assembly.electrodes[electrode_uuid]
        elec_hits = np.sum(collision_data['electrode_ids'] == elec_idx)
        if elec_hits > 0:
            stats_text += f"  {electrode.name}:\n"
            stats_text += f"    {elec_hits} particles\n"
            stats_text += f"    ({100 * elec_hits / n_particles:.1f}%)\n"

    if n_collisions > 0:
        avg_frac = np.mean(collision_fractions)
        stats_text += f"\nAvg collision\n"
        stats_text += f"position: {avg_frac:.2f}\n"
        stats_text += f"(0=start, 1=end)\n\n"

        min_frac = np.min(collision_fractions)
        max_frac = np.max(collision_fractions)
        stats_text += f"Min: {min_frac:.2f}\n"
        stats_text += f"Max: {max_frac:.2f}\n"

    ax3.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round',
                                                   facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('collision_detection_example.png', dpi=150)
    print("\nSaved: collision_detection_example.png")
    plt.show()