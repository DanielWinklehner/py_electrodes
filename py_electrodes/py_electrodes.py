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
# TEMP_DIR = settings["TEMP_DIR"]
OCC_GRADIENT1 = [int(item) for item in settings["OCC_GRADIENT1"].split("]")[0].split("[")[1].split(",")]
OCC_GRADIENT2 = [int(item) for item in settings["OCC_GRADIENT2"].split("]")[0].split("[")[1].split(",")]

# Temporary directory for saving intermittent files
# if os.path.exists(TEMP_DIR):
#     shutil.rmtree(TEMP_DIR)
# os.mkdir(TEMP_DIR)

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
    import kaolin
    HAVE_KAOLIN = True
except ImportError:
    if DEBUG:
        print("Kaolin not available")

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

    def get_bempp_mesh(self, brep_h=0.005):

        if not HAVE_BEMPP and not HAVE_MESHIO:
            print("It looks like we can't find BEMPP or meshio. Aborting!")
            return 1

        # TODO: Can this be expedited using cython or multiple cores? -DW
        if RANK == 0:

            # Initialize empty arrays of the correct shape (3 x n)
            vertices = np.zeros([3, 0])
            elements = np.zeros([3, 0])
            vertex_counter = 0
            domain_counter = 1
            domains = np.zeros([0], int)

            # Domains will just be counted through from 1 to max
            for _id, _electrode in self._electrodes.items():

                # Check whether there is a individual mesh for this electrode already
                # if not: generate it with gmsh
                if _electrode.gmsh_file is None:
                    _electrode.generate_mesh(brep_h=brep_h)

                if HAVE_MESHIO:
                    # Note: This is only the 2D mesh.
                    mesh = meshio.read(_electrode.gmsh_file)
                    # cell_data_tri = mesh.cell_data["triangle"]

                    _vertices = mesh.points.T
                    # _elements = mesh.cells["triangle"].T
                    for cellblock in mesh.cells:
                        if cellblock.type == "triangle":
                            _elements = cellblock.data.T

                    # _domain_ids = np.ones(len(cell_data_tri["gmsh:physical"]), int) * domain_counter
                    # _domain_ids = np.ones(len(mesh.cell_data["gmsh:physical"]), int) * domain_counter
                    # TODO: This seems dangerous for mixed cases where some electrodes have a bempp domain -DW
                    dom_idx = _electrode.bempp_domain if _electrode.bempp_domain is not None else domain_counter
                    _domain_ids = np.ones(_elements.shape[1], int) * dom_idx

                    vertices = np.concatenate((vertices, _vertices), axis=1)
                    elements = np.concatenate((elements, _elements + vertex_counter), axis=1)
                    domains = np.concatenate((domains, _domain_ids), axis=0)

                    _electrode.bempp_domain = dom_idx  # Override if simple enumeration was used

                    vertex_counter += _vertices.shape[1]
                    domain_counter += 1

                # elif HAVE_BEMPP:
                #     print("_electrode.gmsh_file", _electrode.gmsh_file)
                #     mesh = bempp_cl.api.import_grid(_electrode.gmsh_file)
                #
                #     _vertices = mesh.leaf_view.vertices
                #     _elements = mesh.leaf_view.elements
                #     _domain_ids = np.ones(len(mesh.leaf_view.domain_indices), int) * domain_counter
                #
                #     vertices = np.concatenate((vertices, _vertices), axis=1)
                #     elements = np.concatenate((elements, _elements + vertex_counter), axis=1)
                #     domains = np.concatenate((domains, _domain_ids), axis=0)
                #
                #     # set current domain index in electrode object
                #     _electrode.bempp_domain = domain_counter
                #
                #     # Increase the running counters
                #     vertex_counter += _vertices.shape[1]
                #     domain_counter += 1

            self._full_mesh = {"verts": vertices,
                               "elems": elements,
                               "domns": domains}

            # Apply assembly global transformation here
            vertices = self._transformation.apply_to_points(vertices.T).T

            if DEBUG:
                if HAVE_BEMPP:
                    bempp_cl.api.PLOT_BACKEND = "gmsh"
                    BemppGrid(vertices, elements, domains).plot()

        if MPI is not None:
            # Broadcast results to all nodes
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
                                 return_first_hit_only=True, chunk_size=None):
        """
        Batch ray-surface intersection across all electrodes.

        Returns:
            hit_data: dict {
                'hit_mask': (N,) bool - which rays hit something,
                'hit_points': (N, 3) - surface intersection points,
                'distances': (N,) - distance to intersection,
                'electrode_ids': (N,) - which electrode index (-1 if no hit),
                'electrode_index_map': dict mapping int index to electrode UUID,
            }
        """

        N = len(ray_origins)

        if chunk_size is None:
            chunk_size = N

        hit_data = {
            'hit_mask': np.zeros(N, dtype=bool),
            'hit_points': np.full((N, 3), np.nan, dtype=np.float32),
            'distances': np.full(N, np.inf, dtype=np.float32),
            'electrode_ids': np.full(N, -1, dtype=np.int32),
            'electrode_index_map': {},
        }

        # Create mapping of electrode ID (UUID) to integer index
        electrode_id_map = {elec_id: idx for idx, elec_id in enumerate(self.electrodes.keys())}
        hit_data['electrode_index_map'] = {v: k for k, v in electrode_id_map.items()}

        # Process each electrode
        for elec_id, electrode in self._electrodes.items():
            elec_idx = electrode_id_map[elec_id]  # Get integer index

            # Process in chunks
            for i in range(0, N, chunk_size):
                end_idx = min(i + chunk_size, N)
                chunk_origins = ray_origins[i:end_idx]
                chunk_dirs = ray_directions[i:end_idx]

                hit_mask, hit_pts, dists = electrode.ray_surface_intersection(
                    chunk_origins, chunk_dirs, use_gpu=use_gpu
                )

                if return_first_hit_only:
                    # Only update if this is closer than previous hits
                    is_closer = hit_mask & (dists < hit_data['distances'][i:end_idx])

                    hit_data['hit_mask'][i:end_idx][is_closer] = True
                    hit_data['hit_points'][i:end_idx][is_closer] = hit_pts[is_closer]
                    hit_data['distances'][i:end_idx][is_closer] = dists[is_closer]
                    hit_data['electrode_ids'][i:end_idx][is_closer] = elec_idx  # Use integer index
                else:
                    # Record all hits (could have multiple per ray)
                    hit_data['hit_mask'][i:end_idx] |= hit_mask
                    hit_data['hit_points'][i:end_idx][hit_mask] = hit_pts[hit_mask]
                    hit_data['distances'][i:end_idx][hit_mask] = dists[hit_mask]
                    hit_data['electrode_ids'][i:end_idx][hit_mask] = elec_idx  # Use integer index

        return hit_data

    def compute_axis_aligned_surface_intersections(self, mesh_nodes, axes='all',
                                                   use_gpu=None, chunk_size=None):
        """
        Compute ray-surface intersections for mesh nodes in cardinal directions.

        Returns:
            intersections: dict {
                'x+': {'hit_mask': (N,), 'hit_points': (N, 3), 'distances': (N,)},
                'x-': {...},
                'y+': {...},
                'y-': {...},
                'z+': {...},
                'z-': {...},
                'electrode_ids': (N, 6) - electrode index for each direction (-1 if no hit),
                'electrode_index_map': dict mapping int index to electrode UUID,
            }
        """

        if axes == 'all':
            axes = list(AXIS_DIRECTIONS.keys())

        mesh_nodes = np.asarray(mesh_nodes, dtype=np.float32)
        assert mesh_nodes.ndim == 2 and mesh_nodes.shape[1] == 3, \
            "mesh_nodes must be (N, 3) array"

        N = len(mesh_nodes)

        # Create mapping
        electrode_id_map = {elec_id: idx for idx, elec_id in enumerate(self.electrodes.keys())}

        intersections = {
            'electrode_ids': np.full((N, len(axes)), -1, dtype=np.int32),
            'electrode_index_map': {v: k for k, v in electrode_id_map.items()},
        }

        for ax_idx, axis_name in enumerate(axes):
            if axis_name not in AXIS_DIRECTIONS:
                raise ValueError(
                    f"Unknown axis: {axis_name}. Must be one of {list(AXIS_DIRECTIONS.keys())}"
                )

            ray_direction = AXIS_DIRECTIONS[axis_name]

            # Create ray directions for all mesh nodes
            ray_directions = np.tile(ray_direction, (N, 1)).astype(np.float32)

            # Get intersections in this direction
            hit_data = self.ray_surface_intersection(
                mesh_nodes, ray_directions,
                use_gpu=use_gpu,
                return_first_hit_only=True,
                chunk_size=chunk_size
            )

            # Store results
            intersections[axis_name] = {
                'hit_mask': hit_data['hit_mask'],
                'hit_points': hit_data['hit_points'],
                'distances': hit_data['distances'],
            }

            intersections['electrode_ids'][:, ax_idx] = hit_data['electrode_ids']

        return intersections

    def compute_surface_distances(self, mesh_nodes, exclude_electrode_id=None,
                                  use_gpu=None):
        """
        Compute signed distance from mesh nodes to nearest electrode surface.
        For PIC field solvers with conductor boundaries.

        Args:
            mesh_nodes: (N, 3) array of field grid points
            exclude_electrode_id: Optional - don't compute distance to this electrode
            use_gpu: None (auto-detect), True (force GPU), False (force CPU)

        Returns:
            signed_distances: (N,) - negative inside, positive outside
            nearest_electrode_id: (N,) - which electrode is nearest
            nearest_surface_point: (N, 3) - closest point on any surface

        Example:
            > assembly = PyElectrodeAssembly("RFQ")
            > mesh_pts = np.random.randn(10000, 3)
            > dists, elec_ids, surf_pts = assembly.compute_surface_distances(mesh_pts)
        """

        N = len(mesh_nodes)

        # Initialize output arrays
        signed_distances = np.full(N, np.inf, dtype=np.float32)
        nearest_electrode_id = np.full(N, -1, dtype=np.int32)
        nearest_surface_point = np.full((N, 3), np.nan, dtype=np.float32)

        # Process each electrode
        for elec_id, electrode in self._electrodes.items():

            if exclude_electrode_id is not None and elec_id == exclude_electrode_id:
                continue

            dist, pts = electrode.signed_distance_to_surface(mesh_nodes, use_gpu=use_gpu)

            # Update only where this electrode is closer
            closer = np.abs(dist) < np.abs(signed_distances)

            signed_distances[closer] = dist[closer]
            nearest_electrode_id[closer] = elec_id
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
        self._occ_obj = None
        self._bempp_domain = None

        self._transformation = CoordinateTransformation3D()

        self._initialized = False

        if self._geo_str is not None:
            self._originated_from = "geo_str"
            self.generate_from_geo_str(self._geo_str)

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

    def generate_mesh(self, brep_h=0.005):

        if self._orig_file is None:
            print("No geometry loaded yet!")
            return 1

        msh_fn = os.path.join(TEMP_DIR, "{}.msh".format(self._id))
        log_fn = os.path.join(TEMP_DIR, "{}_gmsh.log".format(self._id))

        # For now, we need to save in msh2 format for BEMPP compability
        # gmsh can handle geo, brep and stl the same way. However, brep has no mesh resolution
        # information. STL is already a mesh...

        # Create a string that contains the transformations
        # TODO: may have to add the mesh size in again.
        # TODO: What about the reverse mesh thing?
        # TODO: This is assuming the user has defined a volume in geo string or geo file...
        if self._originated_from == "brep":

            # command = "{} \"{}\" -2 -clmax {} -o \"{}\" -format msh2 -log {}".format(GMSH_EXE,
            #                                                                          self._orig_file,
            #                                                                          brep_h,
            #                                                                          msh_fn,
            #                                                                          log_fn)

            try:
                result = subprocess.run(
                    [GMSH_EXE, self._orig_file, "-2", "-clmax", brep_h, "-o", msh_fn, "-format", "msh2"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                gmsh_success = result.returncode

                if result.returncode != 0:
                    self._debug_message(f"gmsh stderr: {result.stderr}")

            except subprocess.TimeoutExpired:
                self._debug_message("gmsh timed out after 60 seconds")
                gmsh_success = 1

            except Exception as e:
                self._debug_message(f"Error running gmsh: {e}")
                gmsh_success = 1

        elif self._originated_from in ["geo_str", "geo_file"]:

            tx, ty, tz = self._transformation.translation
            v_rot = quaternion.as_rotation_vector(self._transformation.rotation)
            angle = np.sqrt(np.sum(np.dot(v_rot, v_rot)))

            omit_t = omit_r = ""

            if np.abs(tx) + np.abs(ty) + np.abs(tz) < 1 / DECIMALS:
                # no translation... omit in geo file
                omit_t = "//"

            if np.abs(angle) <= 1 / DECIMALS:
                # no rotation... omit in geo file
                omit_r = "//"

            transform_str = """v() = Volume "*";
{}Rotate {{ {{ {}, {}, {} }}, {{ 0, 0, 0 }}, {} }} {{  Volume{{v()}}; }}
{}Translate {{ {}, {}, {} }} {{ Volume{{v()}}; }}
""".format(omit_r, v_rot[0], v_rot[1], v_rot[2], angle, omit_t, tx, ty, tz)

            transform_fn = os.path.join(TEMP_DIR, "{}_trafo.geo".format(self._id))
            with open(transform_fn, "w") as _of:
                _of.write(transform_str)

            # command = "{} \"{}\" \"{}\" -2 -o \"{}\" -format msh2 -log {}".format(GMSH_EXE,
            #                                                                       self._orig_file,
            #                                                                       transform_fn,
            #                                                                       msh_fn,
            #                                                                       log_fn)

            try:

                result = subprocess.run(
                    [GMSH_EXE, self._orig_file, transform_fn, "-2", "-o", msh_fn, "-format", "msh2"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                gmsh_success = result.returncode

                if result.returncode != 0:
                    self._debug_message(f"gmsh stderr: {result.stderr}")

            except subprocess.TimeoutExpired:
                self._debug_message("gmsh timed out after 60 seconds")
                gmsh_success = 1

            except Exception as e:
                self._debug_message(f"Error running gmsh: {e}")
                gmsh_success = 1


        elif self._originated_from == "stl":
            print("Meshing with transformations from stl not yet implemented")
            return 1
        else:
            print("Format not supported for meshing!")
            return 1

        # sys.stdout.flush()
        # gmsh_success = os.system(command)

        if gmsh_success != 0:

            # Catch error messages related to transfinite algorithm (mesh will still be produced)
            with open(log_fn, 'r') as infile:
                lines = infile.readlines()

            ignore_error = True
            for line in lines:
                if "Error" in line and "cannot be meshed using the transfinite algo" not in line:
                    ignore_error = False

            if not ignore_error:
                self._debug_message("Something went wrong with gmsh, log file was saved in {}".format(TEMP_DIR))
                return 1

        self._gmsh_file = msh_fn

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

    def generate_from_file(self, filename=None):
        """
        Loads the electrode object from file. Extension can be .brep, .geo, .stl
        :param filename: input file name.
        :return:
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
                self._generate_from_brep()
            elif ext.lower() == ".geo":
                self._originated_from = "geo_file"
                self._generate_from_geo()
            elif ext.lower() == ".stl":
                self._originated_from = "stl"
                self._generate_from_stl()
            # elif ext.lower() in [".stp", ".step"]:
            #     self._originated_from = "step"
            #     self._generate_from_step()

            return 0

        else:

            return 1

    def _generate_from_brep(self):
        self._debug_message("Generating from brep")

        self._occ_obj = PyOCCElectrode(debug=DEBUG)
        self._occ_obj.translation = self._transformation.translation
        self._occ_obj.rotation = self._transformation.rotation
        error = self._occ_obj.load_from_brep(self._orig_file)

        if error:
            return error
        else:
            self._initialized = True
            return 0

    def _generate_from_geo(self):
        self._debug_message("Generating from geo")

        geo_fn = self._orig_file
        brep_fn = os.path.join(TEMP_DIR, "{}.brep".format(self._id))
        sto_fn = os.path.join(TEMP_DIR, "{}_geo-to-brep.out".format(self._id))
        err_fn = os.path.join(TEMP_DIR, "{}_geo-to-brep.err".format(self._id))

        gmsh_success = 0

        # Call gmsh to transform .geo file to .brep
        # command = "\"{}\" \"{}\" -0 -o \"{}\" -format brep 1>{} 2>{}".format(GMSH_EXE,
        #                                                                  geo_fn,
        #                                                                  brep_fn,
        #                                                                  sto_fn,
        #                                                                  err_fn)
        #
        # self._debug_message("Running", command)
        # gmsh_success += os.system(command)
        #
        # if gmsh_success != 0:
        #     self._debug_message("Something went wrong with gmsh, output and error was saved in {}".format(TEMP_DIR))
        #     return 1

        try:
            result = subprocess.run(
                [GMSH_EXE, geo_fn, "-0", "-o", brep_fn, "-format", "brep"],
                capture_output=True,
                text=True,
                timeout=60
            )

            gmsh_success += result.returncode

            if result.returncode != 0:
                self._debug_message(f"gmsh stderr: {result.stderr}")

        except subprocess.TimeoutExpired:
            self._debug_message("gmsh timed out after 60 seconds")
            gmsh_success += 1

        except Exception as e:
            self._debug_message(f"Error running gmsh: {e}")
            gmsh_success += 1

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

    def _get_trimesh(self):
        """
        Cached trimesh object for collision detection.
        Loads mesh only once, then returns cached version.
        """
        if not hasattr(self, '_trimesh_cache') or self._trimesh_cache is None:
            if not HAVE_TRIMESH:
                raise ImportError(
                    "trimesh required for collision detection. "
                    "Install with: pip install trimesh"
                )

            if self.gmsh_file is None:
                self.generate_mesh()

            mesh_data = meshio.read(self.gmsh_file)
            vertices = mesh_data.points

            triangles = None
            for cell_block in mesh_data.cells:
                if cell_block.type == "triangle":
                    triangles = cell_block.data
                    break

            if triangles is None:
                raise RuntimeError(f"No triangles in mesh for {self.name}")

            self._trimesh_cache = trimesh.Trimesh(
                vertices=vertices,
                faces=triangles,
                process=False
            )

        return self._trimesh_cache

    def ray_surface_intersection(self, ray_origins, ray_directions, use_gpu=None,
                                 max_distance=None):
        """
        Find surface intersection for arbitrary rays.

        Args:
            ray_origins: (N, 3) array of ray starting points
            ray_directions: (N, 3) array of ray directions (should be normalized)
            use_gpu: None (auto-detect), True (force GPU), False (force CPU)
            max_distance: Optional - only return intersections within this distance

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
                ray_origins, ray_directions, max_distance
            )
        else:
            return self._ray_surface_intersection_cpu(
                ray_origins, ray_directions, max_distance
            )

    def _ray_surface_intersection_cpu(self, ray_origins, ray_directions, max_distance=None):
        """CPU implementation using trimesh ray casting"""

        mesh = self._get_trimesh()

        # Ray-mesh intersection
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False  # Single first hit per ray
        )

        # Initialize output arrays
        N = len(ray_origins)
        hit_mask = np.zeros(N, dtype=bool)
        hit_points = np.full((N, 3), np.nan, dtype=np.float32)
        distances = np.full(N, np.inf, dtype=np.float32)

        # Process each intersection
        for i, loc in enumerate(locations):
            ray_idx = index_ray[i]
            dist_to_hit = np.linalg.norm(loc - ray_origins[ray_idx])

            # Check distance limit if specified
            if max_distance is not None and dist_to_hit > max_distance:
                continue

            hit_mask[ray_idx] = True
            hit_points[ray_idx] = loc.astype(np.float32)
            distances[ray_idx] = dist_to_hit

        return hit_mask, hit_points, distances

    def _ray_surface_intersection_gpu(self, ray_origins, ray_directions, max_distance=None):
        """GPU implementation using NVIDIA Warp - simplified version"""

        if not HAVE_WARP:
            raise RuntimeError("Warp not available")

        try:
            mesh = self._get_trimesh()

            # Ensure numpy arrays
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            ray_origins = np.asarray(ray_origins, dtype=np.float32)
            ray_directions = np.asarray(ray_directions, dtype=np.float32)

            # Convert to Warp
            vertices_wp = wp.from_numpy(vertices)
            faces_wp = wp.from_numpy(faces.flatten())
            ray_origins_wp = wp.from_numpy(ray_origins)
            ray_directions_wp = wp.from_numpy(ray_directions)

            N = len(ray_origins)
            n_triangles = len(faces)

            # CORRECT: Initialize output arrays with proper types
            hit_mask = wp.zeros(N, dtype=wp.uint8)
            hit_points = wp.zeros(N, dtype=wp.vec3f)  # Array of vec3f, not (N,3) float32
            hit_distances = wp.zeros(N, dtype=wp.float32)

            max_dist = float(max_distance) if max_distance is not None else 1e30

            @wp.kernel
            def ray_triangle_intersect(
                    origins: wp.array(dtype=wp.vec3f),
                    directions: wp.array(dtype=wp.vec3f),
                    vertices: wp.array(dtype=wp.vec3f),
                    faces: wp.array(dtype=wp.int32),
                    n_tri: wp.int32,
                    max_d: wp.float32,
                    hits: wp.array(dtype=wp.uint8),
                    points: wp.array(dtype=wp.vec3f),
                    distances: wp.array(dtype=wp.float32),
            ):
                ray_id = wp.tid()

                origin = origins[ray_id]
                direction = directions[ray_id]

                closest_t = max_d
                closest_point = wp.vec3f(0.0, 0.0, 0.0)

                # Check all triangles
                for tri_id in range(n_tri):
                    # Get vertex indices
                    base = tri_id * 3
                    v0_idx = faces[base]
                    v1_idx = faces[base + 1]
                    v2_idx = faces[base + 2]

                    # Get vertices
                    v0 = vertices[v0_idx]
                    v1 = vertices[v1_idx]
                    v2 = vertices[v2_idx]

                    # Möller-Trumbore
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

                if closest_t < max_d:
                    hits[ray_id] = wp.uint8(1)
                    points[ray_id] = closest_point
                    distances[ray_id] = closest_t

            # Launch kernel
            wp.launch(
                ray_triangle_intersect,
                dim=N,
                inputs=[
                    ray_origins_wp,
                    ray_directions_wp,
                    vertices_wp,
                    faces_wp,
                    wp.int32(n_triangles),
                    wp.float32(max_dist),
                    hit_mask,
                    hit_points,
                    hit_distances,
                ]
            )

            # Copy to numpy - FIXED: use helper function
            hit_mask_raw = _warp_to_numpy(hit_mask)
            hit_points_raw = _warp_to_numpy(hit_points)
            distances_raw = _warp_to_numpy(hit_distances)

            hit_mask_np = hit_mask_raw.astype(bool)
            distances_np = distances_raw.astype(np.float32)

            # Convert vec3f structs to (N, 3) float array
            hit_points_np = np.zeros((N, 3), dtype=np.float32)
            for i in range(N):
                hit_points_np[i, 0] = hit_points_raw[i][0]
                hit_points_np[i, 1] = hit_points_raw[i][1]
                hit_points_np[i, 2] = hit_points_raw[i][2]

            # Mark misses
            hit_points_np[~hit_mask_np] = np.nan
            distances_np[~hit_mask_np] = np.inf

            return hit_mask_np, hit_points_np, distances_np

        except Exception as e:
            logger.warning(f"GPU ray intersection failed: {e}. Falling back to CPU.")
            return self._ray_surface_intersection_cpu(ray_origins, ray_directions, max_distance)



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

        Args:
            points: (N, 3) array of query points
            use_gpu: None (auto-detect), True (force GPU), False (force CPU)

        Returns:
            signed_distances: (N,) array, negative inside, positive outside
            nearest_points: (N, 3) closest surface points

        Example:
            > elec = PyElectrode("test")
            > points = np.random.randn(1000, 3)
            > dists, surf_pts = elec.signed_distance_to_surface(points)
        """

        # Auto-detect if not specified
        if use_gpu is None:
            use_gpu = HAVE_KAOLIN

        if use_gpu and not HAVE_KAOLIN:
            warnings.warn(
                "GPU requested but Kaolin not available. "
                "Install with: pip install kaolin torch. "
                "Falling back to CPU.",
                RuntimeWarning
            )
            use_gpu = False

        points = np.asarray(points, dtype=np.float32)
        assert points.ndim == 2 and points.shape[1] == 3, \
            "points must be (N, 3) array"

        if use_gpu:
            return self._signed_distance_gpu(points)
        else:
            return self._signed_distance_cpu(points)

    def _signed_distance_cpu(self, points):
        """CPU implementation using trimesh proximity queries"""

        mesh = self._get_trimesh()

        # Get closest points on surface
        closest_pts, distances, _ = trimesh.proximity.closest_point(mesh, points)

        # Determine if inside or outside using ray casting
        # Cast rays in random directions and count intersections
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

    def _signed_distance_gpu(self, points):
        """GPU implementation using Kaolin"""

        if not HAVE_KAOLIN:
            raise RuntimeError("Kaolin not available")

        import torch

        mesh = self._get_trimesh()

        # Convert to torch tensors
        vertices = torch.from_numpy(mesh.vertices).float().cuda()
        faces = torch.from_numpy(mesh.faces).long().cuda()
        points_torch = torch.from_numpy(points).float().cuda()

        try:
            # Use Kaolin's point-to-mesh distance
            distances, closest_indices = kaolin.metrics.pointcloud.point_to_mesh_distance(
                points_torch.unsqueeze(0),
                vertices.unsqueeze(0),
                faces
            )
            distances = distances.squeeze(0).cpu().numpy()
        except Exception as e:
            logger.warning(f"Kaolin distance computation failed: {e}. Using CPU fallback.")
            return self._signed_distance_cpu(points)

        # Get closest surface points
        closest_indices = closest_indices.cpu().numpy().flatten()
        closest_pts = mesh.vertices[closest_indices]

        # Inside/outside test via ray casting (CPU for now)
        directions = np.random.randn(len(points), 3).astype(np.float32)
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

        _, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=points,
            ray_directions=directions,
            multiple_hits=True
        )

        intersection_count = np.bincount(index_ray, minlength=len(points))
        inside = (intersection_count % 2) == 1

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
