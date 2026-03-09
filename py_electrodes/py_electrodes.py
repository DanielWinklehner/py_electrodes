import numpy as np
import sys
import os
import time
import uuid
# noinspection PyUnresolvedReferences
from .py_electrodes_occ import *  # PyCharm Commmunity doesn't recognize cython pyx files :)
import shutil
# noinspection PyPackageRequirements
import quaternion  # package name is numpy-quaternion
from .settings import SettingsHandler
from .tk_filedialog import FileDialog
from .gmsh_installer import GmshInstaller
# import copy
from pathlib import Path
import warnings
import logging
import enum
import subprocess
import tempfile
import atexit

TEMP_DIR = tempfile.mkdtemp(prefix="py_electrodes_")
atexit.register(lambda: shutil.rmtree(TEMP_DIR, ignore_errors=True))

__author__ = "Daniel Winklehner"
__doc__ = """Create electrodes using gmsh and pythonocc-core for use in field calculations and particle tracking"""

# --- Set global variables from Settings.txt file--- #
settings = SettingsHandler()
DEBUG = (settings["DEBUG"] == "True")
DECIMALS = float(settings["DECIMALS"])
GMSH_EXE = settings["GMSH_EXE"]
DEBUG_OUTPUT_DIR = settings["DEBUG_DIR"]
OCC_GRADIENT1 = [int(item) for item in settings["OCC_GRADIENT1"].split("]")[0].split("[")[1].split(",")]
OCC_GRADIENT2 = [int(item) for item in settings["OCC_GRADIENT2"].split("]")[0].split("[")[1].split(",")]

# Debug output directory (persists after program completion)
if DEBUG and not os.path.exists(DEBUG_OUTPUT_DIR):
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

# Define the axis directions and vane rotations:
X = 0
Y = 1
Z = 2
AXES = {"X": 0, "Y": 1, "Z": 2}
XYZ = range(3)  # All directions as a list

HAVE_GMSH = True
# Quick test if gmsh path is correct
if not Path(GMSH_EXE).is_file():
    print("I cannot find gmsh in the path specified in Settings.txt, starting gmsh installer...")
    gi = GmshInstaller()
    GMSH_EXE = gi.run()
    if GMSH_EXE is None:
        HAVE_GMSH = False

# --- Test if we have OCC and Viewer
HAVE_OCC = False
try:
    from OCC.Extend.DataExchange import *
    from OCC.Display.SimpleGui import init_display
    from OCC.Core.gp import gp_Vec, gp_Quaternion
    from OCC.Core.TopoDS import TopoDS_Compound
    from OCC.Core.BRep import BRep_Builder

    HAVE_OCC = True

except ImportError:
    init_display = None
    gp_Vec = gp_Quaternion = None
    if DEBUG:
        print("Something went wrong during OCC import. No OpenCasCade support outside gmsh possible!")

# --- Try importing BEMPP
HAVE_BEMPP = False
try:
    import bempp_cl.api
    from bempp_cl.api.shapes.shapes import __generate_grid_from_geo_string as generate_from_string
    from bempp_cl.api.grid import Grid as BemppGrid

    HAVE_BEMPP = True
except ImportError:
    print("Couldn't import BEMPP, no meshing or BEM field calculation will be possible.")
    bempp = None
    generate_from_string = None

# --- Try import meshio
HAVE_MESHIO = False
try:
    import meshio

    HAVE_MESHIO = True
except ImportError:
    meshio = None

# --- Try importing mpi4py, if it fails, we fall back to single processor
try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    HOST = MPI.Get_processor_name()
    # print("Process {} of {} on host {} started!".format(RANK + 1, SIZE, HOST))
except ImportError:
    if DEBUG:
        print("Could not import MPI/mpi4py, falling back to python multiprocessing where appropriate!")
    MPI = None
    COMM = None
    RANK = 0
    SIZE = 1
    import socket

    HOST = socket.gethostname()

# --- GPU acceleration (optional) --- #
HAVE_WARP = False
HAVE_KAOLIN = False
HAVE_TRIMESH = False

try:
    import warp as wp
    wp.init()
    HAVE_WARP = True
except (ImportError, RuntimeError) as e:
    if DEBUG:
        print(f"Warp not available: {e}")

try:
    import trimesh
    HAVE_TRIMESH = True
except ImportError:
    print("Warning: trimesh not available. Collision detection will not work.")
    print("Install with: pip install trimesh")
    trimesh = None

# Logger for graceful warnings
logger = logging.getLogger(__name__)


def _warp_to_numpy(warp_array):
    """Convert Warp array to numpy, handling different Warp versions"""
    try:
        # Try newer API first
        return wp.to_numpy(warp_array)
    except AttributeError:
        try:
            # Try alternative methods
            return np.array(warp_array.numpy())
        except (AttributeError, TypeError):
            try:
                # Last resort: direct numpy conversion
                import ctypes
                return np.ctypeslib.as_array(warp_array.ptr, shape=warp_array.shape)
            except Exception:
                raise RuntimeError(
                    f"Cannot convert Warp array to numpy. "
                    f"Warp version: {wp.__version__ if hasattr(wp, '__version__') else 'unknown'}"
                )


class AxisDirection(enum.Enum):
    """Axis-aligned ray directions"""
    X_POS = (1.0, 0.0, 0.0)
    X_NEG = (-1.0, 0.0, 0.0)
    Y_POS = (0.0, 1.0, 0.0)
    Y_NEG = (0.0, -1.0, 0.0)
    Z_POS = (0.0, 0.0, 1.0)
    Z_NEG = (0.0, 0.0, -1.0)

# Convenience dict - convert tuples to numpy arrays
AXIS_DIRECTIONS = {
    'x+': np.array(AxisDirection.X_POS.value, dtype=np.float32),
    'x-': np.array(AxisDirection.X_NEG.value, dtype=np.float32),
    'y+': np.array(AxisDirection.Y_POS.value, dtype=np.float32),
    'y-': np.array(AxisDirection.Y_NEG.value, dtype=np.float32),
    'z+': np.array(AxisDirection.Z_POS.value, dtype=np.float32),
    'z-': np.array(AxisDirection.Z_NEG.value, dtype=np.float32),
}

class CoordinateTransformation3D(object):

    def __init__(self):

        self._translation = np.array([0.0, 0.0, 0.0])
        self._rotation = np.quaternion(1.0, 0.0, 0.0, 0.0)

    def apply_to_points(self, points):

        points = np.asarray(points)
        assert points.ndim == 2 and points[0].shape == (3,), "points array needs to be of shape (N, 3)"

        # Rotation first
        points = quaternion.rotate_vectors(self._rotation, points)

        # Then translation
        points[:, 0] += self._translation[0]
        points[:, 1] += self._translation[1]
        points[:, 2] += self._translation[2]

        return points

    def apply_inverse_to_points(self, points):

        points = np.asarray(points)
        assert points.ndim == 2 and points[0].shape == (3,), "points array needs to be of shape (N, 3)"

        # Translation first
        points[:, 0] -= self._translation[0]
        points[:, 1] -= self._translation[1]
        points[:, 2] -= self._translation[2]

        # Then rotation
        inv_rot = self._rotation.inverse()
        points = quaternion.rotate_vectors(inv_rot, points)

        return points

    @property
    def translation(self):
        return self._translation

    @property
    def rotation(self):
        return self._rotation

    def set_translation(self, translation, absolute=True):
        """
        Sets the new translation
        :param translation: list, tuple or numpy array with three elements dx, dy, dz
        :param absolute: whether this shift replaces the old shift or is added
        :return: 
        """

        translation = np.asarray(translation)

        if translation.shape == (3,):
            if absolute:
                self._translation = translation
            else:
                self._translation += translation

    def set_rotation_from_angle_axis(self, angle, axis, absolute=True):
        """
        Quaternion object can be generated from rotation axis where length of vector = rotation angle
        :param angle: rotation angle in rad
        :param axis: rotation axis, can be unnormalized
        :param absolute: Replace existing rotation or append (multiply with existing)
        :return:
        """
        qaxis = np.copy(axis)
        qaxis /= np.linalg.norm(qaxis)
        qaxis *= angle
        quat = quaternion.from_rotation_vector(qaxis)

        if absolute:
            self._rotation = quat
        else:
            self._rotation = quat * self._rotation


class PyElectrodeAssembly(object):
    """
    The Assembly's transformation will only be applied when it is displayed (show),
    exported to file, or a full mesh is calculated from it.
    It will not be applied to the individual electrodes!
    """

    def __init__(self,
                 name="New Assembly"):

        self._name = name
        self._electrodes = {}
        self._full_mesh = None  # BEMPP full mesh
        self._transformation = CoordinateTransformation3D()

    @property
    def electrodes(self):
        return self._electrodes

    @property
    def local_to_global_transformation(self):
        return self._transformation

    @local_to_global_transformation.setter
    def local_to_global_transformation(self, trafo):

        if isinstance(trafo, CoordinateTransformation3D):
            self._transformation = trafo
        else:
            print("Can only set the full transformation as a CoordinateTransformation3D object! "
                  "Consider using set_translation(), set_rotation_angle_axis()")

    def bounding_box(self):
        """Get combined bounding box of all electrodes"""
        bounds_list = [e.bounding_box() for e in self._electrodes.values()
                       if e.bounding_box() is not None]

        if not bounds_list:
            return None

        # Unpack and merge
        xmins = [b[0] for b in bounds_list]
        xmaxs = [b[1] for b in bounds_list]
        ymins = [b[2] for b in bounds_list]
        ymaxs = [b[3] for b in bounds_list]
        zmins = [b[4] for b in bounds_list]
        zmaxs = [b[5] for b in bounds_list]

        return (min(xmins), max(xmaxs), min(ymins), max(ymaxs), min(zmins), max(zmaxs))

    def export(self, filename):
        # TODO: Add transformation! -DW

        file_type = filename.split('.')[-1]

        assert file_type in ['step', 'stp', 'stl', 'iges'], "File type must be '.step', '.stp', '.stl', or '.iges'!"

        # Assemble all occ_electrodes into a compound
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for key, item in self.electrodes.items():
            builder.Add(compound, item.occ_obj.ds_shape)

        # Save compound to file
        if file_type == 'step' or file_type == 'stp':
            write_step_file(compound, filename, application_protocol="AP203")
        elif file_type == 'stl':
            write_stl_file(compound, filename)
        elif file_type == 'iges':
            write_iges_file(compound, filename)
        else:
            return 1

        return 0

    def set_translation(self, translation, absolute=True):

        translation = np.asarray(translation)
        if not translation.shape == (3,):
            print("Shift has to be a 3 x 1 array of dx, dy, dz")
            return 1

        self._transformation.set_translation(translation, absolute=absolute)
        self._full_mesh = None

    def set_rotation_angle_axis(self, angle, axis, absolute=True):

        axis = np.asarray(axis)
        if not axis.shape == (3,) and type(angle) == float:
            print("Angle has to be a float (rad) and axis be of shape (3, )")
            return 1

        self._transformation.set_rotation_from_angle_axis(angle, axis, absolute=absolute)
        self._full_mesh = None

        print("Sorry, rotation of an assembly is not yet implemented!")

    @staticmethod
    def _debug_message(*args, rank=0):
        if RANK == rank and DEBUG:
            print(*args)
            sys.stdout.flush()
        return 0

    def add_electrode(self, electrode):

        assert isinstance(electrode, PyElectrode), "Can only add PyElectrode objects to PyElectrodeAssembly!"

        self._electrodes[electrode.id] = electrode

    def points_inside(self, _points):

        _ts = time.time()

        _points = self._transformation.apply_inverse_to_points(_points)  # To take into account the assembly glob. trafo

        self._debug_message("\n*** Calculating is_inside for {} points ***".format(_points.shape[0]))

        _mask = np.zeros(_points.shape[0], dtype=bool)

        for _id, _electrode in self._electrodes.items():
            self._debug_message("[{}] Working on electrode object {}".format(
                time.strftime('%H:%M:%S', time.gmtime(int(time.time() - _ts))), _electrode.name))

            _mask = _mask | _electrode.points_inside(_points)

        return _mask

    def get_electrode_by_name(self, name):
        """
        Get all electrodes with the given name
        :param name: str
        :return: list of electrodes with the given name. Empty list if there aren't any.
        """

        _elecs = []

        for _, _elec in self.electrodes.items():
            if name == _elec.name:
                _elecs.append(_elec)

        if len(_elecs) == 1:

            return _elecs[0]

        else:

            return _elecs


    def get_bempp_mesh(self, brep_h=None):
        """Assemble full bempp mesh from electrode meshes (in-memory)"""

        if not HAVE_BEMPP:
            print("BEMPP not found. Aborting!")
            return 1

        if RANK == 0:
            # Collect existing bempp_domain assignments
            existing_domains = [e.bempp_domain for e in self._electrodes.values()
                                if e.bempp_domain is not None]

            # Start domain counter after max existing domain (avoid collisions)
            domain_counter = max(existing_domains) + 1 if existing_domains else 1

            # Initialize empty arrays
            vertices = np.zeros([3, 0])
            elements = np.zeros([3, 0])
            vertex_counter = 0
            domains = np.zeros([0], int)

            # Assemble all electrode meshes
            for _id, _electrode in self._electrodes.items():

                # Generate mesh if not already done
                if _electrode._gmsh_msh is None:
                    _electrode.generate_mesh(brep_h=brep_h)

                # Extract from in-memory mesh
                _vertices = _electrode._gmsh_msh['vertices'].T  # (3, N)
                _elements = _electrode._gmsh_msh['elements'].T  # (3, M)

                # Determine domain index (respect pre-assigned domains)
                if _electrode.bempp_domain is not None:
                    dom_idx = _electrode.bempp_domain
                else:
                    dom_idx = domain_counter
                    domain_counter += 1

                _domain_ids = np.ones(_elements.shape[1], int) * dom_idx

                # Concatenate
                vertices = np.concatenate((vertices, _vertices), axis=1)
                elements = np.concatenate((elements, _elements + vertex_counter), axis=1)
                domains = np.concatenate((domains, _domain_ids), axis=0)

                # Update electrode's domain
                _electrode.bempp_domain = dom_idx
                vertex_counter += _vertices.shape[1]

            self._full_mesh = {
                "verts": vertices,
                "elems": elements,
                "domns": domains
            }

            # Apply assembly global transformation
            vertices_transformed = self._transformation.apply_to_points(vertices.T).T
            self._full_mesh["verts"] = vertices_transformed

            if DEBUG:
                bempp_cl.api.PLOT_BACKEND = "gmsh"
                BemppGrid(vertices_transformed, elements, domains).plot()

        if MPI is not None:
            self._full_mesh = COMM.bcast(self._full_mesh, root=0)
            COMM.barrier()

        return self._full_mesh

    def segment_intersects_surface(self, point_start, point_end, use_gpu=None,
                                   chunk_size=None):
        """
        Batch collision check for all particles across all electrodes.
        Returns which particles hit which electrodes.

        Returns:
            collision_data: dict {
                'hit_mask': (N,) bool array,
                'hit_points': (N, 3) intersection points,
                'electrode_ids': (N,) which electrode index was hit (-1 if none),
                'hit_fractions': (N,) position along segment [0-1],
                'electrode_index_map': dict mapping index to electrode UUID,
            }
        """

        N = len(point_start)

        if chunk_size is None:
            chunk_size = N

        collision_data = {
            'hit_mask': np.zeros(N, dtype=bool),
            'hit_points': np.full((N, 3), np.nan, dtype=np.float32),
            'electrode_ids': np.full(N, -1, dtype=np.int32),
            'hit_fractions': np.full(N, np.nan, dtype=np.float32),
            'electrode_index_map': {},  # Maps int index to UUID
        }

        # Create mapping of electrode ID (UUID) to integer index
        electrode_id_map = {elec_id: idx for idx, elec_id in enumerate(self.electrodes.keys())}
        collision_data['electrode_index_map'] = {v: k for k, v in electrode_id_map.items()}

        for elec_id, electrode in self.electrodes.items():
            elec_idx = electrode_id_map[elec_id]  # Get integer index

            # Process in chunks to save memory
            for i in range(0, N, chunk_size):
                end_idx = min(i + chunk_size, N)
                chunk_start = point_start[i:end_idx]
                chunk_end = point_end[i:end_idx]

                hit_mask, hit_pts, hit_fracs = electrode.segment_intersects_surface(
                    chunk_start, chunk_end, use_gpu=use_gpu
                )

                # Only update if this is the first hit for these particles
                first_hit = hit_mask & ~collision_data['hit_mask'][i:end_idx]

                collision_data['hit_mask'][i:end_idx][first_hit] = True
                collision_data['hit_points'][i:end_idx][first_hit] = hit_pts[first_hit]
                collision_data['electrode_ids'][i:end_idx][first_hit] = elec_idx  # Use integer index
                collision_data['hit_fractions'][i:end_idx][first_hit] = hit_fracs[first_hit]

        return collision_data

    def ray_surface_intersection(self, ray_origins, ray_directions, use_gpu=None,
                                 return_first_hit_only=False, chunk_size=None):
        """
        Batch ray-surface intersection across all electrodes.

        Returns:
            hit_data: dict {
                'hit_mask': (N,) bool,
                'hit_points': list of (M_i, 3) arrays - ragged,
                'distances': list of (M_i,) arrays - ragged,
                'electrode_ids': list of (M_i,) arrays - electrode ID for each hit,
            }
        """

        N = len(ray_origins)

        if chunk_size is None:
            chunk_size = N

        hit_data = {
            'hit_mask': np.zeros(N, dtype=bool),
            'hit_points': [np.empty((0, 3)) for _ in range(N)],
            'distances': [np.array([]) for _ in range(N)],
            'electrode_ids': [np.array([], dtype=np.int32) for _ in range(N)],
        }

        # Create electrode ID mapping
        electrode_id_map = {elec_id: idx for idx, elec_id in enumerate(self.electrodes.keys())}

        # Process each electrode
        for elec_id, electrode in self._electrodes.items():
            elec_idx = electrode_id_map[elec_id]

            for i in range(0, N, chunk_size):
                end_idx = min(i + chunk_size, N)
                chunk_origins = ray_origins[i:end_idx]
                chunk_dirs = ray_directions[i:end_idx]

                hit_mask, hit_pts, dists = electrode.ray_surface_intersection(
                    chunk_origins, chunk_dirs, use_gpu=use_gpu,
                    return_first_hit_only=return_first_hit_only
                )

                # Append all hits from this electrode
                for local_idx, ray_hit in enumerate(hit_mask):
                    global_idx = i + local_idx

                    if ray_hit:
                        hit_data['hit_mask'][global_idx] = True
                        hit_data['hit_points'][global_idx] = np.vstack([
                            hit_data['hit_points'][global_idx],
                            hit_pts[local_idx]
                        ])
                        hit_data['distances'][global_idx] = np.hstack([
                            hit_data['distances'][global_idx],
                            dists[local_idx]
                        ])
                        hit_data['electrode_ids'][global_idx] = np.hstack([
                            hit_data['electrode_ids'][global_idx],
                            np.full(len(dists[local_idx]), elec_idx, dtype=np.int32)
                        ])

        return hit_data

    def compute_axis_aligned_surface_intersections(self, mesh_nodes, axes='all',
                                                   use_gpu=None, chunk_size=None):
        """
        Ultra-fast Mega-Kernel implementation.
        Computes all 6 cardinal directions simultaneously entirely on the GPU.
        Returns flat NumPy arrays instead of nested dictionaries.
        """
        import warp as wp
        import numpy as np

        if not HAVE_WARP or not use_gpu:
            raise RuntimeError("This optimized path requires NVIDIA Warp (use_gpu=True).")

        N = len(mesh_nodes)

        # Pre-allocate flat result arrays on CPU, copy to GPU
        min_dists_np = np.full((N, 6), 1e30, dtype=np.float32)
        hit_counts_np = np.zeros((N, 6), dtype=np.int32)

        origins_wp = wp.array(mesh_nodes, dtype=wp.vec3f)
        min_dists_wp = wp.array(min_dists_np, dtype=wp.float32)
        hit_counts_wp = wp.array(hit_counts_np, dtype=wp.int32)

        # Helper function inside Warp to get exact cardinal vectors
        @wp.func
        def get_ray_dir(d: int):
            if d == 0: return wp.vec3f(1.0, 0.0, 0.0)
            if d == 1: return wp.vec3f(-1.0, 0.0, 0.0)
            if d == 2: return wp.vec3f(0.0, 1.0, 0.0)
            if d == 3: return wp.vec3f(0.0, -1.0, 0.0)
            if d == 4: return wp.vec3f(0.0, 0.0, 1.0)
            if d == 5: return wp.vec3f(0.0, 0.0, -1.0)
            return wp.vec3f(0.0, 0.0, 0.0)

        @wp.kernel
        def mega_kernel_raycast(
                mesh: wp.uint64,
                origins: wp.array(dtype=wp.vec3f),
                min_dists: wp.array(dtype=wp.float32, ndim=2),
                hit_counts: wp.array(dtype=wp.int32, ndim=2),
                max_hits: int
        ):
            tid = wp.tid()
            orig = origins[tid]

            # Loop over all 6 directions (x+, x-, y+, y-, z+, z-)
            for d in range(6):
                direction = get_ray_dir(d)

                current_origin = orig
                accumulated_t = float(0.0)
                hits = int(0)
                first_hit_t = float(1e30)

                for _ in range(max_hits):
                    # Query BVH
                    query = wp.mesh_query_ray(mesh, current_origin, direction, float(1e6))
                    if not query.result:
                        break

                    hit_t = accumulated_t + query.t

                    # Capture only the closest hit distance for the Shortley-Weller stencil
                    if hits == 0:
                        first_hit_t = hit_t

                    hits += 1

                    # Step slightly past the hit to continue ray
                    eps = float(1e-4)
                    current_origin = current_origin + direction * (query.t + eps)
                    accumulated_t = hit_t + eps

                # Write results to global memory
                if hits > 0:
                    hit_counts[tid, d] += hits

                    # If this electrode is closer than a previously checked one, update min distance
                    if first_hit_t < min_dists[tid, d]:
                        min_dists[tid, d] = first_hit_t

        # Launch kernel sequentially for each electrode.
        # They will safely accumulate into the same hit_counts and min_dists arrays.
        for elec_id, electrode in self.electrodes.items():
            mesh_wp = electrode.get_warp_mesh()

            wp.launch(
                mega_kernel_raycast,
                dim=N,
                inputs=[mesh_wp.id, origins_wp, min_dists_wp, hit_counts_wp, 50]
            )

        wp.synchronize()

        # Bring back to CPU instantly
        final_min_dists = min_dists_wp.numpy()
        final_hit_counts = hit_counts_wp.numpy()

        return final_min_dists, final_hit_counts

    # def compute_axis_aligned_surface_intersections(self, mesh_nodes, axes='all',
    #                                                use_gpu=None, chunk_size=None):
    #     """
    #     Compute ray-surface intersections for mesh nodes in all cardinal directions.
    #     Returns ALL intersections per ray (not just first).
    #
    #     Returns:
    #         intersections: list of dicts, one per mesh node, structured as:
    #             [
    #                 {  # Node 0
    #                     'x+': {electrode_id_1: {...}, electrode_id_2: {...}},
    #                     'x-': {electrode_id_3: {...}},
    #                     'y+': {...},
    #                     ...
    #                 },
    #                 {...},  # Node 1
    #             ]
    #     """
    #
    #     if axes == 'all':
    #         axes = list(AXIS_DIRECTIONS.keys())
    #
    #     mesh_nodes = np.asarray(mesh_nodes, dtype=np.float32)
    #     assert mesh_nodes.ndim == 2 and mesh_nodes.shape[1] == 3, \
    #         "mesh_nodes must be (N, 3) array"
    #
    #     N = len(mesh_nodes)
    #
    #     # Initialize result: list of dicts, one per node
    #     intersections = [{} for _ in range(N)]
    #
    #     # Create electrode ID mapping
    #     electrode_id_map = {elec_id: idx for idx, elec_id in enumerate(self.electrodes.keys())}
    #
    #     # Process each direction
    #     for axis_name in axes:
    #         if axis_name not in AXIS_DIRECTIONS:
    #             raise ValueError(f"Unknown axis: {axis_name}")
    #
    #         ray_direction = AXIS_DIRECTIONS[axis_name]
    #
    #         # Create ray directions for all mesh nodes
    #         ray_directions = np.tile(ray_direction, (N, 1)).astype(np.float32)
    #
    #         # Get intersections in this direction (all hits, not first only)
    #         hit_data = self.ray_surface_intersection(
    #             mesh_nodes, ray_directions,
    #             use_gpu=use_gpu,
    #             return_first_hit_only=False,
    #             chunk_size=chunk_size
    #         )
    #
    #         # Populate intersections structure
    #         for node_idx in range(N):
    #             intersections[node_idx][axis_name] = {}
    #
    #             # For each hit on this ray
    #             if hit_data['hit_mask'][node_idx]:
    #                 electrode_ids = hit_data['electrode_ids'][node_idx]
    #                 distances = hit_data['distances'][node_idx]
    #                 points = hit_data['hit_points'][node_idx]
    #
    #                 # Group by electrode
    #                 for elec_idx in np.unique(electrode_ids):
    #                     mask = electrode_ids == elec_idx
    #
    #                     intersections[node_idx][axis_name][elec_idx] = {
    #                         'distances': distances[mask],
    #                         'points': points[mask],
    #                     }
    #
    #     return intersections

    def compute_axis_aligned_surface_intersections(self, mesh_nodes, axes='all',
                                                   use_gpu=None, chunk_size=None):
        """
        Computes all 6 cardinal directions simultaneously.
        Returns flat NumPy arrays:
            min_dists_np : (N, 6) float array of closest wall distances
            hit_counts_np : (N, 6) int array of total boundary crossings
        """
        import numpy as np

        # Auto-detect if not specified
        if use_gpu is None:
            use_gpu = HAVE_WARP

        if use_gpu and not HAVE_WARP:
            import warnings
            warnings.warn("GPU requested but Warp not available. Falling back to CPU.", RuntimeWarning)
            use_gpu = False

        if use_gpu:
            return self._compute_axis_aligned_gpu(mesh_nodes)
        else:
            return self._compute_axis_aligned_cpu(mesh_nodes, chunk_size)


    def _compute_axis_aligned_gpu(self, mesh_nodes):
        """
        Ultra-fast Mega-Kernel implementation.
        Computes all 6 cardinal directions simultaneously entirely on the GPU.
        Returns flat NumPy arrays instead of nested dictionaries.
        """
        N = len(mesh_nodes)

        # Pre-allocate flat result arrays on CPU, copy to GPU
        min_dists_np = np.full((N, 6), 1e30, dtype=np.float32)
        hit_counts_np = np.zeros((N, 6), dtype=np.int32)

        origins_wp = wp.array(mesh_nodes, dtype=wp.vec3f)
        min_dists_wp = wp.array(min_dists_np, dtype=wp.float32)
        hit_counts_wp = wp.array(hit_counts_np, dtype=wp.int32)

        # Helper function inside Warp to get exact cardinal vectors
        @wp.func
        def get_ray_dir(d: int):
            if d == 0: return wp.vec3f(1.0, 0.0, 0.0)
            if d == 1: return wp.vec3f(-1.0, 0.0, 0.0)
            if d == 2: return wp.vec3f(0.0, 1.0, 0.0)
            if d == 3: return wp.vec3f(0.0, -1.0, 0.0)
            if d == 4: return wp.vec3f(0.0, 0.0, 1.0)
            if d == 5: return wp.vec3f(0.0, 0.0, -1.0)
            return wp.vec3f(0.0, 0.0, 0.0)

        @wp.kernel
        def mega_kernel_raycast(
                mesh: wp.uint64,
                origins: wp.array(dtype=wp.vec3f),
                min_dists: wp.array(dtype=wp.float32, ndim=2),
                hit_counts: wp.array(dtype=wp.int32, ndim=2),
                max_hits: int
        ):
            tid = wp.tid()
            orig = origins[tid]

            # Loop over all 6 directions (x+, x-, y+, y-, z+, z-)
            for d in range(6):
                direction = get_ray_dir(d)

                current_origin = orig
                accumulated_t = float(0.0)
                hits = int(0)
                first_hit_t = float(1e30)

                for _ in range(max_hits):
                    # Query BVH
                    query = wp.mesh_query_ray(mesh, current_origin, direction, float(1e6))
                    if not query.result:
                        break

                    hit_t = accumulated_t + query.t

                    # Capture only the closest hit distance for the Shortley-Weller stencil
                    if hits == 0:
                        first_hit_t = hit_t

                    hits += 1

                    # Step slightly past the hit to continue ray
                    eps = float(1e-4)
                    current_origin = current_origin + direction * (query.t + eps)
                    accumulated_t = hit_t + eps

                # Write results to global memory
                if hits > 0:
                    hit_counts[tid, d] += hits

                    # If this electrode is closer than a previously checked one, update min distance
                    if first_hit_t < min_dists[tid, d]:
                        min_dists[tid, d] = first_hit_t

        # Launch kernel sequentially for each electrode.
        # They will safely accumulate into the same hit_counts and min_dists arrays.
        for elec_id, electrode in self.electrodes.items():
            mesh_wp = electrode.get_warp_mesh()

            wp.launch(
                mega_kernel_raycast,
                dim=N,
                inputs=[mesh_wp.id, origins_wp, min_dists_wp, hit_counts_wp, 50]
            )

        wp.synchronize()

        # Bring back to CPU instantly
        final_min_dists = min_dists_wp.numpy()
        final_hit_counts = hit_counts_wp.numpy()

        return final_min_dists, final_hit_counts

    def _compute_axis_aligned_cpu(self, mesh_nodes, chunk_size=None):
        """
        Vectorized CPU fallback using trimesh.
        Returns flat (N, 6) arrays identical to the GPU method.
        """
        import numpy as np

        N = len(mesh_nodes)

        # Pre-allocate output arrays
        min_dists_np = np.full((N, 6), 1e30, dtype=np.float32)
        hit_counts_np = np.zeros((N, 6), dtype=np.int32)

        # Standard axes directions mapping to indices 0-5
        axes_vectors = [
            (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)
        ]

        if chunk_size is None:
            chunk_size = 100000  # Trimesh can use a lot of RAM, chunking is safe

        for elec_id, electrode in self.electrodes.items():
            # Assume PyElectrode has a method to get the trimesh object
            mesh = electrode._get_trimesh()

            for dir_idx, ray_dir in enumerate(axes_vectors):
                direction_array = np.tile(ray_dir, (N, 1)).astype(np.float32)

                # Process in chunks to save memory
                for i in range(0, N, chunk_size):
                    end_idx = min(i + chunk_size, N)
                    chunk_origins = mesh_nodes[i:end_idx]
                    chunk_dirs = direction_array[i:end_idx]

                    # Call trimesh intersection
                    locations, index_ray, _ = mesh.ray.intersects_location(
                        ray_origins=chunk_origins,
                        ray_directions=chunk_dirs,
                        multiple_hits=True
                    )

                    if len(locations) == 0:
                        continue

                    # Shift index_ray back to global node indices
                    global_ray_indices = index_ray + i

                    # 1. Update Hit Counts (Count occurrences of each ray index)
                    unique_rays, counts = np.unique(global_ray_indices, return_counts=True)
                    hit_counts_np[unique_rays, dir_idx] += counts

                    # 2. Update Minimum Distances
                    # Calculate exact distances for all returned hits
                    origins_of_hits = mesh_nodes[global_ray_indices]
                    distances = np.linalg.norm(locations - origins_of_hits, axis=1)

                    # Group by ray_index and find the minimum distance per ray
                    # Using np.minimum.at handles duplicate ray indices automatically
                    np.minimum.at(min_dists_np[:, dir_idx], global_ray_indices, distances)

        return min_dists_np, hit_counts_np

    def compute_surface_distances(self, mesh_nodes, exclude_electrode_id=None,
                                  use_gpu=None):  # Keep for API compatibility
        """
        Compute signed distance from mesh nodes to nearest electrode surface.
        For PIC field solvers with conductor boundaries.

        Note: Distance calculation is CPU-only. Ray intersections (bottleneck) use GPU.

        Args:
            mesh_nodes: (N, 3) array of field grid points
            exclude_electrode_id: Optional - don't compute distance to this electrode
            use_gpu: Ignored (kept for API compatibility, always uses CPU)

        Returns:
            signed_distances: (N,) - negative inside, positive outside
            nearest_electrode_id: (N,) - which electrode is nearest
            nearest_surface_point: (N, 3) - closest point on any surface
        """

        N = len(mesh_nodes)

        # Initialize output arrays
        signed_distances = np.full(N, np.inf, dtype=np.float32)
        nearest_electrode_id = np.full(N, -1, dtype=np.int32)
        nearest_surface_point = np.full((N, 3), np.nan, dtype=np.float32)

        # Create mapping
        electrode_id_map = {elec_id: idx for idx, elec_id in enumerate(self.electrodes.keys())}

        # Process each electrode (CPU only)
        for elec_id, electrode in self._electrodes.items():
            elec_idx = electrode_id_map[elec_id]

            if exclude_electrode_id is not None and elec_id == exclude_electrode_id:
                continue

            dist, pts = electrode.signed_distance_to_surface(mesh_nodes)

            # Update only where this electrode is closer
            closer = np.abs(dist) < np.abs(signed_distances)

            signed_distances[closer] = dist[closer]
            nearest_electrode_id[closer] = elec_idx
            nearest_surface_point[closer] = pts[closer]

        return signed_distances, nearest_electrode_id, nearest_surface_point


    def show(self, display=None, show_screen=False):

        if not HAVE_OCC:
            print("OCC couldn't be loaded, no ViewScreen available!")
            return 1

        if display is None:
            display, start_display, _, _ = init_display()
            display.set_bg_gradient_color(OCC_GRADIENT1, OCC_GRADIENT2)

        for _id, _electrode in self._electrodes.items():
            if _electrode is not None:
                # --- Apply tranformation --- #
                # Apply global transformation to _occ_obj
                # TODO: Write function in CoordinateTransformation3D that returns the OCC gp_Vec and gp_Quaternion
                _electrode.occ_obj.apply_second_transformation(translation=gp_Vec(self._transformation.translation[0],
                                                                                  self._transformation.translation[1],
                                                                                  self._transformation.translation[2]),
                                                               rotation=gp_Quaternion(self._transformation.rotation.x,
                                                                                      self._transformation.rotation.y,
                                                                                      self._transformation.rotation.z,
                                                                                      self._transformation.rotation.w))
                # display transformed elec
                display, ais_shape = _electrode.show(display=display)  # ais_shape.GetObject().Shape() holds DS_Shape

                # after adding to display, back-transform (two steps: translation before rotation)
                _electrode.occ_obj.apply_second_transformation(translation=gp_Vec(-self._transformation.translation[0],
                                                                                  -self._transformation.translation[1],
                                                                                  -self._transformation.translation[2]))
                inv_quat = self._transformation.rotation.inverse()
                _electrode.occ_obj.apply_second_transformation(rotation=gp_Quaternion(inv_quat.x,
                                                                                      inv_quat.y,
                                                                                      inv_quat.z,
                                                                                      inv_quat.w))

        if show_screen:
            display.FitAll()
            display.Repaint()
            start_display()

        return display, start_display


class PyElectrode(object):
    def __init__(self,
                 name="New Electrode",
                 voltage=0,
                 geo_str=None):

        self._id = uuid.uuid1()
        self._name = name
        self._voltage = voltage
        self._color = 'RED'
        self._debug = DEBUG

        self._originated_from = ""
        self._orig_file = None
        self._geo_str = geo_str
        self._geo_file = None
        self._stl_file = None
        self._gmsh_file = None
        self._gmsh_msh = None
        self._occ_obj = None
        self._bempp_domain = None
        self._brep_h = 0.005

        self._transformation = CoordinateTransformation3D()

        self._initialized = False

        if self._geo_str is not None:
            self._originated_from = "geo_str"
            self.generate_from_geo_str(self._geo_str)

    @property
    def brep_h(self):
        return self._brep_h

    @brep_h.setter
    def brep_h(self, brep_h):
        self._brep_h = brep_h

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        assert color in ['RED', 'BLUE', 'GREEN', 'BLACK'], "For now, colors are restricted to RED, BLUE, GREEN, BLACK."
        self._color = color

    @property
    def bempp_domain(self):
        return self._bempp_domain

    @bempp_domain.setter
    def bempp_domain(self, domain):
        assert isinstance(domain, int), "Domain index for BEMPP has to be an integer!"
        self._bempp_domain = domain

    @property
    def gmsh_file(self):
        return self._gmsh_file

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def occ_obj(self):
        return self._occ_obj

    @property
    def voltage(self):
        return self._voltage

    @property
    def local_to_global_transformation(self):
        return self._transformation

    @local_to_global_transformation.setter
    def local_to_global_transformation(self, trafo):

        if isinstance(trafo, CoordinateTransformation3D):
            self._transformation = trafo
        else:
            print("Can only set the full transformation as a CoordinateTransformation3D object! "
                  "Consider using set_translation(), set_rotation_angle_axis()")

    def bounding_box(self):
        """Get bounding box of electrode from OCC object"""
        if self._occ_obj is None:
            return None
        return self._occ_obj.get_bounds()  # Returns (xmin, xmax, ymin, ymax, zmin, zmax)

    def set_translation(self, translation, absolute=True):

        translation = np.asarray(translation)
        if not translation.shape == (3,):
            print("Shift has to be a 3 x 1 array of dx, dy, dz")
            return 1

        self._transformation.set_translation(translation, absolute=absolute)

    def set_rotation_angle_axis(self, angle, axis, absolute=True):

        self._transformation.set_rotation_from_angle_axis(angle, axis, absolute=absolute)

    @staticmethod
    def _debug_message(*args, rank=0):
        if RANK == rank and DEBUG:
            print(*args)
            sys.stdout.flush()
        return 0

    def generate_mesh(self, brep_h=None):
        """Generate 2D surface mesh using gmsh Python API"""

        if self._orig_file is None:
            print("No geometry loaded yet!")
            return 1

        if brep_h is None:
            brep_h = self._brep_h

        import gmsh

        try:
            # Initialize gmsh
            gmsh.initialize()

            # Open geometry
            gmsh.open(self._orig_file)
            gmsh.model.occ.synchronize()

            # Get rotation (quaternion to angle-axis)
            v_rot = quaternion.as_rotation_vector(self._transformation.rotation)
            angle = np.linalg.norm(v_rot)

            # Apply rotation if non-negligible
            if angle > 1.0 / DECIMALS:
                axis = v_rot / angle
                origin = [0.0, 0.0, 0.0]

                gmsh.model.occ.rotate(gmsh.model.getEntities(),
                                      origin[0], origin[1], origin[2],
                                      axis[0], axis[1], axis[2], angle)

            # Apply translation
            tx, ty, tz = self._transformation.translation
            if np.abs(tx) + np.abs(ty) + np.abs(tz) > 1.0 / DECIMALS:

                if DEBUG:
                    print(f"Applying translation {tx}, {ty}, {tz} to electrode {self.name}")

                gmsh.model.occ.translate(gmsh.model.getEntities(), tx, ty, tz)

            gmsh.model.occ.synchronize()

            # Note: geo files have mesh size information, brep does not
            if self._originated_from == "brep":
                gmsh.model.mesh.setSize(gmsh.model.getEntities(0), brep_h)
            gmsh.model.mesh.generate(2)

            # Extract mesh data
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            elem_types, elem_tags, elem_connectivity = gmsh.model.mesh.getElements(dim=2)

            # Reshape data
            vertices = node_coords.reshape(-1, 3).astype(np.float32)

            # Handle multiple element types (should be just triangles, but be safe)
            all_elements = []
            all_tags = []
            for elem_type, elem_tag, elem_conn in zip(elem_types, elem_tags, elem_connectivity):
                # Connectivity is flat, reshape to (n_elements, nodes_per_element)
                n_elems = len(elem_tag)
                nodes_per_elem = len(elem_conn) // n_elems
                elements = elem_conn.reshape(n_elems, nodes_per_elem)

                # Convert to 0-indexed
                elements = elements - 1
                all_elements.append(elements)
                all_tags.append(elem_tag)

            # Concatenate all elements
            elements = np.vstack(all_elements).astype(np.int32)
            element_tags = np.hstack(all_tags).astype(np.int32)

            # Store in memory
            self._gmsh_msh = {
                'vertices': vertices,
                'elements': elements,
                'element_tags': element_tags,
            }

            self._debug_message(f"Mesh generated: {len(vertices)} vertices, {len(elements)} elements")

            # Save .msh file if debugging
            if DEBUG:
                self._gmsh_file = os.path.join(DEBUG_OUTPUT_DIR, "{}.msh".format(self._id))
                gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
                gmsh.write(self._gmsh_file)
                self._debug_message(f"Mesh saved to {self._gmsh_file} (debug mode)")

        except Exception as e:
            self._debug_message(f"gmsh error: {e}")
            return 1

        finally:
            try:
                gmsh.finalize()
            except:
                pass

        return 0

    def generate_from_geo_str(self, geo_str=None):
        if geo_str is not None:
            self._geo_str = geo_str

        if self._geo_str is not None:
            self._originated_from = "geo_str"
            self._orig_file = os.path.join(TEMP_DIR, "{}.geo".format(self._id))
            with open(self._orig_file, 'w') as _of:
                _of.write(self._geo_str)
            self._generate_from_geo()

        else:
            self._debug_message("No geo string found!")

        return 0

    def generate_from_file(self, filename=None, input_units=None):
        """
        Loads the electrode object from file. Extension can be .brep, .geo, .stl, .stp
        :param filename: input file name.
        :input_units: input units. Cave: Will not be applied for geo strings and .geo files.
        :return:

        TODO: Handle scaling of input units. Currently, the OCC object can be scaled, but the
              meshing depends on the original input file and is unscaled!!!


        """

        if filename is None:
            fd = FileDialog()
            filename = fd.get_filename()

        if filename is None:
            return 0

        if os.path.isfile(filename):
            name, ext = os.path.splitext(filename)

            assert ext in [".brep", ".geo", ".stl"], \
                "Extension has to be .brep, .geo, .stl"

            self._orig_file = filename

            if ext.lower() == ".brep":
                self._originated_from = "brep"
                self._generate_from_brep(input_units=input_units)
            elif ext.lower() == ".geo":
                self._originated_from = "geo_file"
                self._generate_from_geo()
            elif ext.lower() == ".stl":
                self._originated_from = "stl"
                self._generate_from_stl(input_units=input_units)
            elif ext.lower() in [".stp", ".step"]:
                self._originated_from = "step"
                self._generate_from_step(input_units=input_units)

            return 0

        else:

            return 1

    def _generate_from_brep(self, input_units=None):
        self._debug_message("Generating from brep")

        self._occ_obj = PyOCCElectrode(debug=DEBUG)
        self._occ_obj.translation = self._transformation.translation
        self._occ_obj.rotation = self._transformation.rotation
        error = self._occ_obj.load_from_brep(self._orig_file, input_units=input_units)

        if error:
            return error
        else:
            self._initialized = True
            return 0

    def _generate_from_step(self, input_units=None):
        self._debug_message("Generating from step")

        self._occ_obj = PyOCCElectrode(debug=DEBUG)
        self._occ_obj.translation = self._transformation.translation
        self._occ_obj.rotation = self._transformation.rotation
        error = self._occ_obj.load_from_step(self._orig_file, input_units=input_units)

        if error:
            return error
        else:
            self._initialized = True
            return 0

    def _generate_from_geo(self):
        """Generate OCC object from .geo file using gmsh Python API"""
        self._debug_message("Generating from geo (using gmsh API)")

        import gmsh

        geo_fn = self._orig_file
        brep_fn = os.path.join(TEMP_DIR, "{}.brep".format(self._id))

        try:
            gmsh.initialize()
            gmsh.open(geo_fn)
            gmsh.model.geo.synchronize()
            gmsh.write(brep_fn)

            self._debug_message(f"BREP written to {brep_fn}")

        except Exception as e:
            self._debug_message(f"gmsh error: {e}")
            return 1

        finally:
            # Ensure gmsh is finalized
            try:
                gmsh.finalize()
            except:
                pass

        # Now load the BREP file with OCC (existing code)
        self._occ_obj = PyOCCElectrode(debug=DEBUG)
        self._occ_obj.translation = self._transformation.translation
        self._occ_obj.rotation = self._transformation.rotation
        error = self._occ_obj.load_from_brep(brep_fn)

        if error:
            return error
        else:
            self._initialized = True
            return 0

    def _generate_from_stl(self):
        self._debug_message("Generating from stl")

        self._occ_obj = PyOCCElectrode(debug=DEBUG)
        self._occ_obj.translation = self._transformation.translation
        self._occ_obj.rotation = self._transformation.rotation
        error = self._occ_obj.load_from_stl(self._orig_file)

        if error:
            return error
        else:
            self._initialized = True
            return 0

    def get_warp_mesh(self):
        """Build and cache the NVIDIA Warp BVH mesh."""
        import warp as wp
        import numpy as np

        if not hasattr(self, '_warp_mesh') or self._warp_mesh is None:
            mesh = self._get_trimesh()
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)

            # Cache the Warp mesh object natively
            self._warp_mesh = wp.Mesh(
                points=wp.array(vertices, dtype=wp.vec3f),
                indices=wp.array(faces.flatten(), dtype=wp.int32)
            )

        return self._warp_mesh

    def _get_trimesh(self):
        """
        Cached trimesh object for collision detection.
        Uses in-memory mesh from self._gmsh_msh.
        """
        if not hasattr(self, '_trimesh_cache') or self._trimesh_cache is None:
            if not HAVE_TRIMESH:
                raise ImportError(
                    "trimesh required for collision detection. "
                    "Install with: pip install trimesh"
                )

            # Ensure mesh is generated and stored in memory
            if self._gmsh_msh is None:
                self.generate_mesh()

            vertices = self._gmsh_msh['vertices']
            triangles = self._gmsh_msh['elements']

            if triangles is None or len(triangles) == 0:
                raise RuntimeError(f"No triangles in mesh for {self.name}")

            self._trimesh_cache = trimesh.Trimesh(
                vertices=vertices,
                faces=triangles,
                process=False
            )

        return self._trimesh_cache

    def ray_surface_intersection(self, ray_origins, ray_directions, use_gpu=None,
                                 return_first_hit_only=False):
        """
        Find surface intersection for arbitrary rays.

        Args:
            ray_origins: (N, 3) array of ray starting points
            ray_directions: (N, 3) array of ray directions (should be normalized)
            use_gpu: None (auto-detect), True (force GPU), False (force CPU)

        Returns:
            hit_mask: (N,) boolean - True if ray hits surface
            hit_points: (N, 3) - surface intersection points (NaN if no hit)
            distances: (N,) - distance to intersection (inf if no hit)

        Example:
            > elec = PyElectrode("wall")
            > origins = np.array([[0, 0, 0], [1, 1, 1]])
            > directions = np.array([[1, 0, 0], [1, 0, 0]])  # +x direction
            > hit, pts, dists = elec.ray_surface_intersection(origins, directions)
        """

        # Auto-detect if not specified
        if use_gpu is None:
            use_gpu = HAVE_WARP

        if use_gpu and not HAVE_WARP:
            warnings.warn(
                "GPU requested but Warp not available. "
                "Install with: pip install warp-lang. "
                "Falling back to CPU.",
                RuntimeWarning
            )
            use_gpu = False

        ray_origins = np.asarray(ray_origins, dtype=np.float32)
        ray_directions = np.asarray(ray_directions, dtype=np.float32)

        assert ray_origins.shape == ray_directions.shape, \
            "ray_origins and ray_directions must have same shape"
        assert ray_origins.ndim == 2 and ray_origins.shape[1] == 3, \
            "rays must be (N, 3) arrays"

        # Normalize directions
        ray_directions = ray_directions / np.linalg.norm(
            ray_directions, axis=1, keepdims=True
        )

        if use_gpu:
            return self._ray_surface_intersection_gpu(
                ray_origins, ray_directions, return_first_hit_only
            )
        else:
            return self._ray_surface_intersection_cpu(
                ray_origins, ray_directions, return_first_hit_only
            )

    def _ray_surface_intersection_cpu(self, ray_origins, ray_directions,
                                      return_first_hit_only=False):
        """CPU implementation using trimesh ray casting"""

        mesh = self._get_trimesh()

        # Ray-mesh intersection
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=not return_first_hit_only  # Single first hit per ray
        )

        # Initialize output arrays
        N = len(ray_origins)
        hit_mask = np.zeros(N, dtype=bool)
        hit_points = np.full((N, 3), np.nan, dtype=np.float32)
        distances = np.full(N, np.inf, dtype=np.float32)

        # Process each intersection (can be multiple per ray)
        hit_mask = np.zeros(N, dtype=bool)
        hit_points_list = [[] for _ in range(N)]  # Ragged lists
        hit_distances_list = [[] for _ in range(N)]

        for i, loc in enumerate(locations):
            ray_idx = index_ray[i]
            dist_to_hit = np.linalg.norm(loc - ray_origins[ray_idx])

            hit_mask[ray_idx] = True
            hit_points_list[ray_idx].append(loc)
            hit_distances_list[ray_idx].append(dist_to_hit)

        # Convert to arrays
        hit_points = [np.array(pts) if pts else np.empty((0, 3)) for pts in hit_points_list]
        hit_distances = [np.array(dists) if dists else np.array([]) for dists in hit_distances_list]

        return hit_mask, hit_points, hit_distances

    def _ray_surface_intersection_gpu(self, ray_origins, ray_directions,
                                      return_first_hit_only=False):
        """GPU implementation using NVIDIA Warp's BVH acceleration"""

        if not HAVE_WARP:
            raise RuntimeError("Warp not available")

        try:
            mesh = self._get_trimesh()

            # Ensure numpy arrays
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            ray_origins = np.asarray(ray_origins, dtype=np.float32)
            ray_directions = np.asarray(ray_directions, dtype=np.float32)

            N = len(ray_origins)
            MAX_HITS_PER_RAY = 50

            # 1. Build Warp BVH Mesh
            vertices_wp = wp.array(vertices, dtype=wp.vec3f)
            faces_wp = wp.array(faces.flatten(), dtype=wp.int32)
            mesh_wp = wp.Mesh(points=vertices_wp, indices=faces_wp)

            # 2. Convert ray arrays to Warp
            ray_origins_wp = wp.from_numpy(ray_origins)
            ray_directions_wp = wp.from_numpy(ray_directions)

            # 3. Initialize output arrays (flattened)
            hit_count = wp.zeros(N, dtype=wp.int32)
            hit_distances = wp.zeros(N * MAX_HITS_PER_RAY, dtype=wp.float32)
            hit_points = wp.zeros(N * MAX_HITS_PER_RAY, dtype=wp.vec3f)

            @wp.kernel
            def ray_mesh_intersect_bvh(
                    mesh: wp.uint64,
                    origins: wp.array(dtype=wp.vec3f),
                    directions: wp.array(dtype=wp.vec3f),
                    return_first: wp.bool,
                    max_hits: wp.int32,
                    counts: wp.array(dtype=wp.int32),
                    distances: wp.array(dtype=wp.float32),
                    points: wp.array(dtype=wp.vec3f),
            ):
                ray_id = wp.tid()

                orig_origin = origins[ray_id]
                direction = directions[ray_id]

                current_origin = orig_origin
                accumulated_t = float(0.0)
                hits_found = int(0)

                # Loop to support multiple hits. Capped safely by max_hits.
                for _ in range(max_hits):
                    # Query the BVH tree (max distance 1e6)
                    query = wp.mesh_query_ray(mesh, current_origin, direction, float(1e6))

                    if not query.result:
                        break  # Ray missed or exited the mesh entirely

                    # Calculate absolute intersection point from the ORIGINAL origin
                    hit_t = accumulated_t + query.t
                    hit_point = current_origin + direction * query.t

                    # Store the hit
                    linear_idx = ray_id * max_hits + hits_found
                    distances[linear_idx] = hit_t
                    points[linear_idx] = hit_point

                    hits_found += 1

                    if return_first:
                        break

                    # Advance the ray slightly past the current intersection to find the next one
                    eps = float(1e-4)
                    current_origin = hit_point + direction * eps
                    accumulated_t = hit_t + eps

                # Record the total number of valid hits for this ray
                counts[ray_id] = hits_found

            # Launch kernel
            wp.launch(
                ray_mesh_intersect_bvh,
                dim=N,
                inputs=[
                    mesh_wp.id,  # Pass the BVH id
                    ray_origins_wp,
                    ray_directions_wp,
                    wp.bool(return_first_hit_only),
                    wp.int32(MAX_HITS_PER_RAY),
                    hit_count,
                    hit_distances,
                    hit_points,
                ]
            )

            # Ensure GPU finishes execution
            wp.synchronize()

            # Copy to numpy (preserving your custom helper function if it handles specific memory unmapping)
            # If `_warp_to_numpy` is just `.numpy()`, you can directly use `hit_count.numpy()` instead
            hit_count_np = _warp_to_numpy(hit_count)
            hit_distances_raw = _warp_to_numpy(hit_distances)
            hit_points_raw = _warp_to_numpy(hit_points)

            # Build hit mask (which rays had any hits)
            hit_mask = hit_count_np > 0

            # Reshape flattened arrays back to (N, MAX_HITS_PER_RAY) for slicing
            hit_distances_2d = hit_distances_raw.reshape((N, MAX_HITS_PER_RAY))
            hit_points_2d = hit_points_raw.reshape((N, MAX_HITS_PER_RAY, 3))

            hit_distances_ragged = [
                hit_distances_2d[i, :hit_count_np[i]].astype(np.float32)
                for i in range(N)
            ]
            hit_points_ragged = [
                hit_points_2d[i, :hit_count_np[i]].astype(np.float32)
                for i in range(N)
            ]

            return hit_mask, hit_points_ragged, hit_distances_ragged

        except Exception as e:
            logger.warning(f"GPU ray intersection failed: {e}. Falling back to CPU.")
            return self._ray_surface_intersection_cpu(
                ray_origins, ray_directions, return_first_hit_only
            )


    # def _ray_surface_intersection_gpu(self, ray_origins, ray_directions,
    #                                   return_first_hit_only=False):
    #     """GPU implementation using NVIDIA Warp (supports multiple hits per ray)
    #
    #     TODO: Consider refactoring to triangle-centric parallelization:
    #           Current: N GPU threads (rays), each loops M triangles sequentially
    #           Better:  M GPU threads (triangles), each tests N rays in parallel
    #           Rationale: For typical meshes M >> N, better GPU utilization despite atomic overhead
    #     """
    #
    #     if not HAVE_WARP:
    #         raise RuntimeError("Warp not available")
    #
    #     try:
    #         mesh = self._get_trimesh()
    #
    #         # Ensure numpy arrays
    #         vertices = np.asarray(mesh.vertices, dtype=np.float32)
    #         faces = np.asarray(mesh.faces, dtype=np.int32)
    #         ray_origins = np.asarray(ray_origins, dtype=np.float32)
    #         ray_directions = np.asarray(ray_directions, dtype=np.float32)
    #
    #         # Convert to Warp
    #         vertices_wp = wp.from_numpy(vertices)
    #         faces_wp = wp.from_numpy(faces.flatten())
    #         ray_origins_wp = wp.from_numpy(ray_origins)
    #         ray_directions_wp = wp.from_numpy(ray_directions)
    #
    #         N = len(ray_origins)
    #         n_triangles = len(faces)
    #         MAX_HITS_PER_RAY = 50
    #
    #         # Initialize output arrays (flattened)
    #         hit_count = wp.zeros(N, dtype=wp.int32)
    #         hit_distances = wp.zeros(N * MAX_HITS_PER_RAY, dtype=wp.float32)
    #         hit_points = wp.zeros(N * MAX_HITS_PER_RAY, dtype=wp.vec3f)
    #
    #         @wp.kernel
    #         def ray_triangle_intersect(
    #                 origins: wp.array(dtype=wp.vec3f),
    #                 directions: wp.array(dtype=wp.vec3f),
    #                 vertices: wp.array(dtype=wp.vec3f),
    #                 faces: wp.array(dtype=wp.int32),
    #                 n_tri: wp.int32,
    #                 return_first: wp.bool,
    #                 max_hits: wp.int32,
    #                 counts: wp.array(dtype=wp.int32),
    #                 distances: wp.array(dtype=wp.float32),
    #                 points: wp.array(dtype=wp.vec3f),
    #         ):
    #             ray_id = wp.tid()
    #
    #             origin = origins[ray_id]
    #             direction = directions[ray_id]
    #
    #             closest_t = float(1e30)
    #             closest_point = wp.vec3f(0.0, 0.0, 0.0)
    #
    #             # Check all triangles
    #             for tri_id in range(n_tri):
    #                 # Get vertex indices
    #                 base = tri_id * 3
    #                 v0_idx = faces[base]
    #                 v1_idx = faces[base + 1]
    #                 v2_idx = faces[base + 2]
    #
    #                 # Get vertices
    #                 v0 = vertices[v0_idx]
    #                 v1 = vertices[v1_idx]
    #                 v2 = vertices[v2_idx]
    #
    #                 # Möller-Trumbore
    #                 edge1 = v1 - v0
    #                 edge2 = v2 - v0
    #                 h = wp.cross(direction, edge2)
    #                 a = wp.dot(edge1, h)
    #
    #                 if wp.abs(a) < 1e-7:
    #                     continue
    #
    #                 inv_a = 1.0 / a
    #                 s = origin - v0
    #                 u = wp.dot(s, h) * inv_a
    #
    #                 if u < 0.0 or u > 1.0:
    #                     continue
    #
    #                 q = wp.cross(s, edge1)
    #                 v = wp.dot(direction, q) * inv_a
    #
    #                 if v < 0.0 or u + v > 1.0:
    #                     continue
    #
    #                 t = wp.dot(edge2, q) * inv_a
    #
    #                 if t > 1e-6:
    #                     if return_first:
    #                         # Only keep closest hit
    #                         if t < closest_t:
    #                             closest_t = t
    #                             closest_point = origin + direction * t
    #                     else:
    #                         # Keep all hits
    #                         hit_idx = wp.atomic_add(counts, ray_id, 1)
    #                         linear_idx = ray_id * MAX_HITS_PER_RAY + hit_idx
    #                         distances[linear_idx] = t
    #                         points[linear_idx] = origin + direction * t
    #
    #             # If return_first_hit_only, store the closest hit
    #             if return_first:
    #                 counts[ray_id] = 1
    #                 linear_idx = ray_id * MAX_HITS_PER_RAY
    #                 distances[linear_idx] = closest_t
    #                 points[linear_idx] = closest_point
    #
    #         # Launch kernel
    #         wp.launch(
    #             ray_triangle_intersect,
    #             dim=N,
    #             inputs=[
    #                 ray_origins_wp,
    #                 ray_directions_wp,
    #                 vertices_wp,
    #                 faces_wp,
    #                 wp.int32(n_triangles),
    #                 wp.bool(return_first_hit_only),
    #                 wp.int32(MAX_HITS_PER_RAY),
    #                 hit_count,
    #                 hit_distances,
    #                 hit_points,
    #             ]
    #         )
    #
    #         # Copy to numpy
    #         hit_count_np = _warp_to_numpy(hit_count)
    #         hit_distances_raw = _warp_to_numpy(hit_distances)
    #         hit_points_raw = _warp_to_numpy(hit_points)
    #
    #         # Build hit mask (which rays had any hits)
    #         hit_mask = hit_count_np > 0
    #
    #         # Reshape flattened arrays back to (N, MAX_HITS_PER_RAY) for slicing
    #         hit_distances_2d = hit_distances_raw.reshape((N, MAX_HITS_PER_RAY))
    #         hit_points_2d = hit_points_raw.reshape((N, MAX_HITS_PER_RAY, 3))
    #
    #         hit_distances_ragged = [
    #             hit_distances_2d[i, :hit_count_np[i]].astype(np.float32)
    #             for i in range(N)
    #         ]
    #         hit_points_ragged = [
    #             hit_points_2d[i, :hit_count_np[i]].astype(np.float32)
    #             for i in range(N)
    #         ]
    #
    #         return hit_mask, hit_points_ragged, hit_distances_ragged
    #
    #     except Exception as e:
    #         logger.warning(f"GPU ray intersection failed: {e}. Falling back to CPU.")
    #         return self._ray_surface_intersection_cpu(
    #             ray_origins, ray_directions, return_first_hit_only
    #         )


    def _ray_mesh_intersect_kernel_gpu(self, ray_origins, ray_directions, max_distance=None):
        """GPU kernel for ray-mesh intersection - alternative simpler version"""

        if not HAVE_WARP:
            raise RuntimeError("Warp not available")

        mesh = self._get_trimesh()

        vertices = wp.from_numpy(mesh.vertices.astype(np.float32))
        faces_flat = mesh.faces.flatten().astype(np.int32)
        faces = wp.from_numpy(faces_flat)

        ray_origins_wp = wp.from_numpy(ray_origins)
        ray_directions_wp = wp.from_numpy(ray_directions)

        N = len(ray_origins)
        n_triangles = len(mesh.faces)

        hit_mask = wp.zeros(N, dtype=wp.uint8)
        hit_points = wp.zeros((N, 3), dtype=wp.float32)
        hit_distances = wp.zeros(N, dtype=wp.float32)

        if max_distance is None:
            max_distance = 1e30

        @wp.kernel
        def kernel(
                origins: wp.array(dtype=wp.vec3f),
                directions: wp.array(dtype=wp.vec3f),
                verts: wp.array(dtype=wp.vec3f),
                tri_indices: wp.array(dtype=wp.int32),
                n_tri: wp.int32,
                max_d: wp.float32,
                out_hits: wp.array(dtype=wp.uint8),
                out_pts: wp.array(dtype=wp.vec3f),
                out_dists: wp.array(dtype=wp.float32),
        ):
            i = wp.tid()

            orig = origins[i]
            dirn = directions[i]
            best_t = max_d
            best_pt = wp.vec3f(0.0)

            for t_idx in range(n_tri):
                idx_base = t_idx * 3
                i0 = tri_indices[idx_base]
                i1 = tri_indices[idx_base + 1]
                i2 = tri_indices[idx_base + 2]

                a_pt = verts[i0]
                b_pt = verts[i1]
                c_pt = verts[i2]

                ab = b_pt - a_pt
                ac = c_pt - a_pt
                pvec = wp.cross(dirn, ac)
                det = wp.dot(ab, pvec)

                if wp.abs(det) < 1e-7:
                    continue

                inv_det = 1.0 / det
                tvec = orig - a_pt
                u = wp.dot(tvec, pvec) * inv_det

                if u < 0.0 or u > 1.0:
                    continue

                qvec = wp.cross(tvec, ab)
                v = wp.dot(dirn, qvec) * inv_det

                if v < 0.0 or u + v > 1.0:
                    continue

                t = wp.dot(ac, qvec) * inv_det

                if t > 1e-6 and t < best_t:
                    best_t = t
                    best_pt = orig + dirn * t

            if best_t < max_d:
                out_hits[i] = wp.uint8(1)
                out_pts[i] = best_pt
                out_dists[i] = best_t

        wp.launch(kernel, dim=N, inputs=[
            ray_origins_wp, ray_directions_wp, vertices, faces,
            wp.int32(n_triangles), wp.float32(max_distance),
            hit_mask, hit_points, hit_distances
        ])

        hit_mask_np = wp.to_numpy(hit_mask).astype(bool)
        hit_points_np = wp.to_numpy(hit_points)
        distances_np = wp.to_numpy(hit_distances)

        hit_points_np[~hit_mask_np] = np.nan
        distances_np[~hit_mask_np] = np.inf

        return hit_mask_np, hit_points_np, distances_np

    def segment_intersects_surface(self, point_start, point_end, use_gpu=None):
        """
        Check if line segment crosses electrode surface.
        Uses GPU (Warp) if available, falls back to CPU (trimesh).

        Args:
            point_start: (N, 3) array of starting positions
            point_end: (N, 3) array of ending positions
            use_gpu: None (auto-detect), True (force GPU), False (force CPU)

        Returns:
            hit_mask: (N,) boolean - True if segment crosses boundary
            hit_points: (N, 3) - intersection points (NaN if no hit)
            hit_fractions: (N,) - position along segment [0 to 1] (NaN if no hit)

        Example:
            > elec = PyElectrode("test")
            > pos_old = np.array([[0, 0, 0], [1, 1, 1]])
            > pos_new = np.array([[0.1, 0.1, 0.1], [1.1, 1.1, 1.1]])
            > hit, pts, fracs = elec.segment_intersects_surface(pos_old, pos_new)
        """

        # Auto-detect if not specified
        if use_gpu is None:
            use_gpu = HAVE_WARP

        if use_gpu and not HAVE_WARP:
            warnings.warn(
                "GPU requested but Warp not available. "
                "Install with: pip install warp-lang. "
                "Falling back to CPU (will be ~10-100x slower).",
                RuntimeWarning
            )
            use_gpu = False

        point_start = np.asarray(point_start, dtype=np.float32)
        point_end = np.asarray(point_end, dtype=np.float32)

        assert point_start.shape == point_end.shape, \
            "point_start and point_end must have same shape"
        assert point_start.ndim == 2 and point_start.shape[1] == 3, \
            "points must be (N, 3) arrays"

        if use_gpu:
            return self._segment_intersects_gpu(point_start, point_end)
        else:
            return self._segment_intersects_cpu(point_start, point_end)

    def _segment_intersects_cpu(self, point_start, point_end):
        """CPU implementation using trimesh ray casting"""

        mesh = self._get_trimesh()

        # Ray directions and distances
        ray_dirs = point_end - point_start
        distances = np.linalg.norm(ray_dirs, axis=1, keepdims=True)

        # Avoid division by zero
        distances = np.maximum(distances, 1e-10)
        ray_dirs = ray_dirs / distances
        distances = distances.flatten()

        # Ray-mesh intersection (single hit per ray)
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=point_start,
            ray_directions=ray_dirs,
            multiple_hits=False
        )

        # Initialize output arrays
        N = len(point_start)
        hit_mask = np.zeros(N, dtype=bool)
        hit_points = np.full((N, 3), np.nan, dtype=np.float32)
        hit_fractions = np.full(N, np.nan, dtype=np.float32)

        # Process each intersection
        for i, loc in enumerate(locations):
            ray_idx = index_ray[i]
            dist_to_intersection = np.linalg.norm(loc - point_start[ray_idx])

            # Only count if intersection is between start and end
            if dist_to_intersection <= distances[ray_idx]:
                hit_mask[ray_idx] = True
                hit_points[ray_idx] = loc.astype(np.float32)
                hit_fractions[ray_idx] = (
                        dist_to_intersection / distances[ray_idx]
                )

        return hit_mask, hit_points, hit_fractions


    def _segment_intersects_gpu(self, point_start, point_end):
        """GPU implementation using NVIDIA Warp"""

        if not HAVE_WARP:
            raise RuntimeError("Warp not available")

        try:
            mesh = self._get_trimesh()

            # Ensure numpy arrays
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            point_start = np.asarray(point_start, dtype=np.float32)
            point_end = np.asarray(point_end, dtype=np.float32)

            # Ray setup
            ray_dirs = point_end - point_start
            distances = np.linalg.norm(ray_dirs, axis=1, keepdims=True)
            distances = np.maximum(distances, 1e-10)
            ray_dirs = ray_dirs / distances
            distances = distances.flatten()

            # Convert to Warp
            vertices_wp = wp.from_numpy(vertices)
            faces_wp = wp.from_numpy(faces.flatten())
            origins_wp = wp.from_numpy(point_start)
            directions_wp = wp.from_numpy(ray_dirs)

            N = len(point_start)
            n_triangles = len(faces)

            # Initialize output arrays with proper types
            hit_mask = wp.zeros(N, dtype=wp.uint8)
            hit_points = wp.zeros(N, dtype=wp.vec3f)
            hit_fractions = wp.zeros(N, dtype=wp.float32)

            @wp.kernel
            def segment_intersect_kernel(
                    origins: wp.array(dtype=wp.vec3f),
                    directions: wp.array(dtype=wp.vec3f),
                    vertices: wp.array(dtype=wp.vec3f),
                    faces: wp.array(dtype=wp.int32),
                    n_tri: wp.int32,
                    seg_distances: wp.array(dtype=wp.float32),
                    hits: wp.array(dtype=wp.uint8),
                    points: wp.array(dtype=wp.vec3f),
                    fractions: wp.array(dtype=wp.float32),
            ):
                seg_id = wp.tid()

                origin = origins[seg_id]
                direction = directions[seg_id]
                max_dist = seg_distances[seg_id]

                closest_t = max_dist
                closest_point = wp.vec3f(0.0, 0.0, 0.0)

                for tri_id in range(n_tri):
                    base = tri_id * 3
                    v0 = vertices[faces[base]]
                    v1 = vertices[faces[base + 1]]
                    v2 = vertices[faces[base + 2]]

                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    h = wp.cross(direction, edge2)
                    a = wp.dot(edge1, h)

                    if wp.abs(a) < 1e-7:
                        continue

                    inv_a = 1.0 / a
                    s = origin - v0
                    u = wp.dot(s, h) * inv_a

                    if u < 0.0 or u > 1.0:
                        continue

                    q = wp.cross(s, edge1)
                    v = wp.dot(direction, q) * inv_a

                    if v < 0.0 or u + v > 1.0:
                        continue

                    t = wp.dot(edge2, q) * inv_a

                    if t > 1e-6 and t < closest_t:
                        closest_t = t
                        closest_point = origin + direction * t

                if closest_t < max_dist:
                    hits[seg_id] = wp.uint8(1)
                    points[seg_id] = closest_point
                    fractions[seg_id] = closest_t / max_dist

            seg_dist_wp = wp.from_numpy(distances.astype(np.float32))

            wp.launch(
                segment_intersect_kernel,
                dim=N,
                inputs=[
                    origins_wp, directions_wp, vertices_wp, faces_wp,
                    wp.int32(n_triangles), seg_dist_wp, hit_mask, hit_points, hit_fractions
                ]
            )

            # Copy to numpy - FIXED: use helper function
            hit_mask_raw = _warp_to_numpy(hit_mask)
            hit_points_raw = _warp_to_numpy(hit_points)
            hit_fractions_raw = _warp_to_numpy(hit_fractions)

            hit_mask_np = hit_mask_raw.astype(bool)
            hit_fractions_np = hit_fractions_raw.astype(np.float32)

            # Convert vec3f structs to (N, 3) float array
            hit_points_np = np.zeros((N, 3), dtype=np.float32)
            for i in range(N):
                hit_points_np[i, 0] = hit_points_raw[i][0]
                hit_points_np[i, 1] = hit_points_raw[i][1]
                hit_points_np[i, 2] = hit_points_raw[i][2]

            hit_points_np[~hit_mask_np] = np.nan
            hit_fractions_np[~hit_mask_np] = np.nan

            return hit_mask_np, hit_points_np, hit_fractions_np

        except Exception as e:
            logger.warning(f"GPU segment intersection failed: {e}. Falling back to CPU.")
            return self._segment_intersects_cpu(point_start, point_end)

    def signed_distance_to_surface(self, points, use_gpu=None):
        """
        Compute signed distance from points to electrode surface.
        Negative = inside, Positive = outside

        Note: Distance calculation is CPU-only. GPU is used for ray intersection
        collision detection, which is the performance bottleneck.

        Args:
            points: (N, 3) array of query points
            use_gpu: Ignored (kept for API compatibility)

        Returns:
            signed_distances: (N,) array, negative inside, positive outside
            nearest_points: (N, 3) closest surface points
        """

        points = np.asarray(points, dtype=np.float32)
        assert points.ndim == 2 and points.shape[1] == 3, \
            "points must be (N, 3) array"

        # Always use CPU for distance calculation
        return self._signed_distance_cpu(points)

    def _signed_distance_cpu(self, points):
        """CPU implementation using trimesh proximity queries"""

        mesh = self._get_trimesh()

        # Get closest points on surface
        closest_pts, distances, _ = trimesh.proximity.closest_point(mesh, points)

        # Determine if inside or outside using ray casting
        directions = np.random.randn(len(points), 3).astype(np.float32)
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

        # Ray casting to infinity for inside/outside test
        _, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=points,
            ray_directions=directions,
            multiple_hits=True
        )

        # Count intersections per ray
        intersection_count = np.bincount(index_ray, minlength=len(points))
        inside = (intersection_count % 2) == 1  # Odd = inside

        # Apply sign
        signed_distances = distances.copy()
        signed_distances[inside] *= -1

        return signed_distances.astype(np.float32), closest_pts.astype(np.float32)


    def show(self, display=None):

        if self._occ_obj is not None:
            display, ais_shape = self._occ_obj.show(display, color=self._color)

            return display, ais_shape

        else:

            return None, None


    def points_inside(self, _points):
        """
        Function that calculates whether the point(s) is/are inside the vane or not.
        Currently this only works with pythonocc-core installed and can be very slow
        for a large number of points.
        :param _points: any shape (N, 3) structure holding the points to check. Can be a list of tuples,
                       a list of lists, a numpy array of points (N, 3)...
                       Alternatively: a single point with three coordinates (list, tuple or numpy array)
        :return: boolean numpy array of True or False depending on whether the points are inside or
                 outside (on the surface is counted as inside!)
        """

        if self._occ_obj is not None:

            return self._occ_obj.points_inside(_points)

        else:

            return 1

    def export(self, filename):

        if self._occ_obj is not None:

            return self._occ_obj.export(filename)

        else:

            return 1

    def update_transformations(self):
        """
        Re-applys rotations and translations to the OCC object
        :return: 0
        """

        if not self._initialized:
            return 1

        self._occ_obj.translation = self._transformation.translation
        self._occ_obj.rotation = self._transformation.rotation
        self._occ_obj.update_transformations()

        return 0


if __name__ == "__main__":
    pass
