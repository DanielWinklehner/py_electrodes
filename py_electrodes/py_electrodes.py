import numpy as np
import sys
import os
import time
import uuid
from .py_electrodes_occ import *
import shutil
import quaternion
from .settings import SettingsHandler
from .tk_filedialog import FileDialog
import copy

__author__ = "Daniel Winklehner"
__doc__ = """Create electrodes using gmsh and pythonocc-core for use in field calculations and particle tracking"""

# --- Set global variables from Settings.txt file--- #
settings = SettingsHandler()
DEBUG = (settings["DEBUG"] == "True")
DECIMALS = float(settings["DECIMALS"])
GMSH_EXE = settings["GMSH_EXE"]
TEMP_DIR = settings["TEMP_DIR"]
OCC_GRADIENT1 = [int(item) for item in settings["OCC_GRADIENT1"].split("]")[0].split("[")[1].split(",")]
OCC_GRADIENT2 = [int(item) for item in settings["OCC_GRADIENT2"].split("]")[0].split("[")[1].split(",")]

# Temporary directory for saving intermittent files
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
os.mkdir(TEMP_DIR)

# Define the axis directions and vane rotations:
X = 0
Y = 1
Z = 2
AXES = {"X": 0, "Y": 1, "Z": 2}
XYZ = range(3)  # All directions as a list

# --- Test if we have OCC and Viewer
HAVE_OCC = False
try:
    from OCC.Display.SimpleGui import init_display
    from OCC.Core.gp import gp_Vec, gp_Quaternion

    HAVE_OCC = True
except ImportError:
    init_display = None
    gp_Vec = gp_Quaternion = None
    if DEBUG:
        print("Something went wrong during OCC import. No OpenCasCade support outside gmsh possible!")

# --- Try importing BEMPP
HAVE_BEMPP = False
try:
    import bempp.api
    from bempp.api.shapes.shapes import __generate_grid_from_geo_string as generate_from_string

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

                if HAVE_BEMPP:
                    mesh = bempp.api.import_grid(_electrode.gmsh_file)

                    _vertices = mesh.leaf_view.vertices
                    _elements = mesh.leaf_view.elements
                    _domain_ids = np.ones(len(mesh.leaf_view.domain_indices), int) * domain_counter

                    vertices = np.concatenate((vertices, _vertices), axis=1)
                    elements = np.concatenate((elements, _elements + vertex_counter), axis=1)
                    domains = np.concatenate((domains, _domain_ids), axis=0)

                    # set current domain index in electrode object
                    _electrode.bempp_domain = domain_counter

                    # Increase the running counters
                    vertex_counter += _vertices.shape[1]
                    domain_counter += 1
                elif HAVE_MESHIO:
                    # Note: This is only the 2D mesh.
                    mesh = meshio.read(_electrode.gmsh_file)
                    cell_data_tri = mesh.cell_data["triangle"]

                    _vertices = mesh.points.T
                    _elements = mesh.cells["triangle"].T
                    _domain_ids = np.ones(len(cell_data_tri["gmsh:physical"]), int) * domain_counter

                    vertices = np.concatenate((vertices, _vertices), axis=1)
                    elements = np.concatenate((elements, _elements + vertex_counter), axis=1)
                    domains = np.concatenate((domains, _domain_ids), axis=0)

                    _electrode.bempp_domain = domain_counter

                    vertex_counter += _vertices.shape[1]
                    domain_counter += 1

            self._full_mesh = {"verts": vertices,
                               "elems": elements,
                               "domns": domains}

            # Apply assembly global transformation here
            vertices = self._transformation.apply_to_points(vertices.T).T

            if DEBUG:
                if HAVE_BEMPP:
                    bempp.api.grid.grid_from_element_data(vertices,
                                                          elements,
                                                          domains).plot()

        if MPI is not None:
            # Broadcast results to all nodes
            self._full_mesh = COMM.bcast(self._full_mesh, root=0)

            COMM.barrier()

        return self._full_mesh

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
        sto_fn = os.path.join(TEMP_DIR, "{}_gmsh.out".format(self._id))
        err_fn = os.path.join(TEMP_DIR, "{}_gmsh.err".format(self._id))

        # For now, we need to save in msh2 format for BEMPP compability
        # gmsh can handle geo, brep and stl the same way. However, brep has no mesh resolution
        # information. STL is already a mesh...

        # Create a string that contains the transformations
        # TODO: may have to add the mesh size in again.
        # TODO: What about the reverse mesh thing?
        # TODO: This is assuming the user has defined a volume in geo string or geo file...
        if self._originated_from == "brep":

            command = "{} \"{}\" -2 -clmax {} -o \"{}\" -format msh2 1>{} 2>{}".format(GMSH_EXE,
                                                                                       self._orig_file,
                                                                                       brep_h,
                                                                                       msh_fn,
                                                                                       sto_fn,
                                                                                       err_fn)
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

            command = "{} \"{}\" \"{}\" -2 -o \"{}\" -format msh2 1>{} 2>{}".format(GMSH_EXE,
                                                                                    self._orig_file,
                                                                                    transform_fn,
                                                                                    msh_fn,
                                                                                    sto_fn,
                                                                                    err_fn)

        elif self._originated_from == "stl":
            print("Meshing with transformations from stl not yet implemented")
            return 1
        else:
            print("Format not supported for meshing!")
            return 1

        sys.stdout.flush()
        gmsh_success = os.system(command)

        if gmsh_success != 0:
            self._debug_message("Something went wrong with gmsh, output and error was saved in {}".format(TEMP_DIR))
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
        command = "{} \"{}\" -0 -o \"{}\" -format brep 1>{} 2>{}".format(GMSH_EXE,
                                                                         geo_fn,
                                                                         brep_fn,
                                                                         sto_fn,
                                                                         err_fn)

        self._debug_message("Running", command)
        gmsh_success += os.system(command)

        if gmsh_success != 0:
            self._debug_message("Something went wrong with gmsh, output and error was saved in {}".format(TEMP_DIR))
            return 1

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

    def show(self, display=None):

        if self._occ_obj is not None:
            display, ais_shape = self._occ_obj.show(display, color=self._color)

        return display, ais_shape

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
