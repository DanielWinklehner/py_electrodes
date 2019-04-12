import numpy as np
import sys
import os
import uuid
from .py_electrodes_occ import *
import shutil
import quaternion


# --- Some global variables --- #
# Display debug messages?
DEBUG = True
# How many decimal places to use for rounding
DECIMALS = 12
# Define the axis directions and vane rotations:
X = 0
Y = 1
Z = 2
AXES = {"X": 0, "Y": 1, "Z": 2}
XYZ = range(3)  # All directions as a list

# Temporary directory for saving intermittent files
TEMP_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "temp")
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
os.mkdir(TEMP_DIR)

# Gmsh path to executable
GMSH_EXE = "gmsh"
# GMSH_EXE = "E:/gmsh4/gmsh.exe"

__author__ = "Daniel Winklehner"
__doc__ = """Create electrodes using gmsh and pythonocc-core for use in field calculations and particle tracking"""

# --- Test if we have OCC and Viewer
HAVE_OCC = False
try:
    from OCC.Display.SimpleGui import init_display
    HAVE_OCC = True
except ImportError:
    init_display = None
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

# For now, everything involving the pymodules with be done on master proc (RANK 0)
if RANK == 0:

    from dans_pymodules import *

    COLORS = MyColors()

else:

    COLORS = None
# ------------------------------------ #


class CoordinateTransformation3D(object):

    def __init__(self):

        self._translation = np.array([0.0, 0.0, 0.0])
        self._rotation = np.quaternion(1.0, 0.0, 0.0, 0.0)

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

        if translation.shape == (3, ):
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
        axis /= np.linalg.norm(axis)
        axis *= angle
        quat = quaternion.from_rotation_vector(axis)

        if absolute:
            self._rotation = quat
        else:
            self._rotation = quat * self._rotation


class PyElectrodeAssembly(object):

    def __init__(self,
                 name="New Assembly"):

        self._name = name
        self._electrodes = {}
        self._full_mesh = None  # BEMPP full mesh

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

        self._debug_message("\n*** Calculating is_inside for {} points ***".format(_points.shape[0]))

        _mask = np.zeros(_points.shape[0], dtype=bool)

        for _id, _electrode in self._electrodes.items():

            self._debug_message("[{}] Working on electrode object {}".format(
                time.strftime('%H:%M:%S', time.gmtime(int(time.time() - _ts))), _electrode.name))

            _mask = _mask | _electrode.points_inside(_points)

        return _mask

    def get_bempp_mesh(self, brep_h=0.005):

        if not HAVE_BEMPP:
            print("It looks like we can't find BEMPP. Aborting!")
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

            self._full_mesh = {"verts": vertices,
                               "elems": elements,
                               "domns": domains}

            if DEBUG:
                bempp.api.grid.grid_from_element_data(vertices,
                                                      elements,
                                                      domains).plot()

        if MPI is not None:
            # Broadcast results to all nodes
            self._full_mesh = COMM.bcast(self._full_mesh, root=0)

            COMM.barrier()

        return self._full_mesh

    def show(self):

        if not HAVE_OCC:
            print("OCC couldn't be loaded, no ViewScreen available!")
            return 1

        display, start_display, _, _ = init_display()
        display.set_bg_gradient_color(175, 210, 255, 255, 255, 255)

        for _id, _electrode in self._electrodes.items():
            if _electrode is not None:
                display.DisplayShape(_electrode._occ_obj._elec, color=_electrode.color, update=False)

        display.FitAll()
        display.Repaint()
        start_display()

        return 0


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

        self._local_to_global_transformation = CoordinateTransformation3D()

        self._initialized = False

        if self._geo_str is not None:
            self._originated_from = "geo_str"
            self.generate_from_geo_str(self._geo_str)

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        assert color in ['RED', 'BLUE', 'GREEN'], "For now, colors are restricted to RED, BLUE, GREEN."
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
    def voltage(self):
        return self._voltage

    @property
    def local_to_global_transformation(self):
        return self._local_to_global_transformation

    @local_to_global_transformation.setter
    def local_to_global_transformation(self, trafo):

        if isinstance(trafo, CoordinateTransformation3D):
            self._local_to_global_transformation = trafo
        else:
            print("Can only set the full transformation as a CoordinateTransformation3D object! "
                  "Consider using set_shift(), set_rotation()")

    def set_translation(self, translation, absolute=True):

        translation = np.asarray(translation)
        if not translation.shape == (3, ):
            print("Shift has to be a 3 x 1 array of dx, dy, dz")
            return 1
        
        self._local_to_global_transformation.set_translation(translation, absolute=absolute)

    def set_rotation_angle_axis(self, angle, axis, absolute=True):

        self._local_to_global_transformation.set_rotation_from_angle_axis(angle, axis, absolute=absolute)

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

        print("\n\nIn generate_mesh: Originated from = {}\n\n".format(self._originated_from))

        msh_fn = os.path.join(TEMP_DIR, "{}.msh".format(self._id))

        # For now, we need to save in msh2 format for BEMPP compability
        # gmsh can handle geo, brep and stl the same way. However, brep has no mesh resolution
        # information. STL is already a mesh...

        # Create a string that contains the transformations
        # TODO: may have to add the mesh size in again.
        # TODO: What about the reverse mesh thing?
        # TODO: This is assuming the user has defined a volume in geo string or geo file...
        if self._originated_from == "brep":

            command = "{} \"{}\" -2 -clmax {} -o \"{}\" -format msh2".format(GMSH_EXE, self._orig_file,
                                                                             brep_h, msh_fn)
        elif self._originated_from in ["geo_str", "geo_file"]:

            tx, ty, tz = self._local_to_global_transformation.translation
            transform_str = """SetFactory("OpenCASCADE");
Geometry.NumSubEdges = 100; // nicer display of curve
Merge "{}";

v() = Volume "*";

Translate {{ {}, {}, {} }} {{ Volume{{v()}}; }}
//Rotate {{ expression-list }} {{  v(); }}
                """.format(self._orig_file, tx, ty, tz)

            transform_fn = os.path.join(TEMP_DIR, "{}_trafo.geo".format(self._id))
            with open(transform_fn, "w") as _of:
                _of.write(transform_str)

            command = "{} \"{}\" -2 -o \"{}\" -format msh2".format(GMSH_EXE, transform_fn, msh_fn)

        elif self._originated_from == "stl":
            print("Meshing with transformations from stl not yet implemented")
            return 1
        else:
            print("Format not supported for meshing!")
            return 1

        if self._debug:
            print("Running", command)
            sys.stdout.flush()

        gmsh_success = os.system(command)

        if gmsh_success != 0:

            self._debug_message("Something went wrong with gmsh, be sure you defined "
                                "the correct path at the beginning of the file!")
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
        self._occ_obj.translation = self._local_to_global_transformation.translation
        self._occ_obj.rotation = self._local_to_global_transformation.rotation
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

        gmsh_success = 0

        # Call gmsh to transform .geo file to .brep
        command = "{} \"{}\" -0 -o \"{}\" -format brep".format(GMSH_EXE, geo_fn, brep_fn)
        if RANK == 0 and DEBUG:
            print("Running", command)
            sys.stdout.flush()
        gmsh_success += os.system(command)

        if gmsh_success != 0:

            self._debug_message("Something went wrong with gmsh, be sure you defined "
                                "the correct path at the beginning of the file!")

            return 1

        self._occ_obj = PyOCCElectrode(debug=DEBUG)
        self._occ_obj.translation = self._local_to_global_transformation.translation
        self._occ_obj.rotation = self._local_to_global_transformation.rotation
        error = self._occ_obj.load_from_brep(brep_fn)

        if error:
            return error
        else:
            self._initialized = True
            return 0

    def _generate_from_stl(self):
        self._debug_message("Generating from stl")

        self._occ_obj = PyOCCElectrode(debug=DEBUG)
        self._occ_obj.translation = self._local_to_global_transformation.translation
        self._occ_obj.rotation = self._local_to_global_transformation.rotation
        error = self._occ_obj.load_from_stl(self._orig_file)

        if error:
            return error
        else:
            self._initialized = True
            return 0

    # def _generate_from_step(self):
    #     self._debug_message("Generating from step")
    #
    #     print("{}: Generating from step not yet implemented!".format(self._name))

    def show(self):

        if self._occ_obj is not None:
            self._occ_obj.show()

        return 0

#     def generate_gmsh_files(self):
#
#         tmp_dir = TEMP_DIR
#
#         if tmp_dir is not None:
#
#             geo_fn = os.path.join(tmp_dir, "{}.geo".format(self.name))
#             msh_fn = os.path.splitext(geo_fn)[0] + ".msh"
#             stl_fn = os.path.splitext(geo_fn)[0] + ".stl"
#             brep_fn = os.path.splitext(geo_fn)[0] + ".brep"
#             refine_fn = os.path.join(tmp_dir, "refine_{}.geo".format(self.name))
#
#             gmsh_success = 0
#
#             with open(geo_fn, "w") as _of:
#                 _of.write(self._geo_str)
#
#             command = "{} \"{}\" -0 -o \"{}\" -format brep".format(GMSH_EXE, geo_fn, brep_fn)
#             if self._debug:
#                 print("Running", command)
#                 sys.stdout.flush()
#             gmsh_success += os.system(command)
#
#             refine_str = """
# Merge "{}";
# Mesh.SecondOrderLinear = 0;
# RefineMesh;
# """.format(msh_fn)
#
#             with open(refine_fn, "w") as _of:
#                 _of.write(refine_str)
#
#             # TODO: Could we use higher order (i.e. curved) meshes? -DW
#             # For now, we need to save in msh2 format for BEMPP compability
#             command = "{} \"{}\" -2 -o \"{}\" -format msh2".format(GMSH_EXE, geo_fn, msh_fn)
#             if self._debug:
#                 print("Running", command)
#                 sys.stdout.flush()
#             gmsh_success += os.system(command)
#
#             # for i in range(self._refine_steps):
#             #     command = "{} \"{}\" -0 -o \"{}\" -format msh2".format(GMSH_EXE, refine_fn, msh_fn)
#             #     if self._debug:
#             #         print("Running", command)
#             #         sys.stdout.flush()
#             #     gmsh_success += os.system(command)
#
#             # --- TODO: For testing: save stl mesh file also
#             command = "{} \"{}\" -0 -o \"{}\" -format stl".format(GMSH_EXE, msh_fn, stl_fn)
#             if self._debug:
#                 print("Running", command)
#                 sys.stdout.flush()
#             gmsh_success += os.system(command)
#             # --- #
#
#             if gmsh_success != 0:  # or not os.path.isfile("shape.stl"):
#                 print("Something went wrong with gmsh, be sure you defined "
#                       "the correct path at the beginning of the file!")
#                 return 1
#
#             # self._mesh_fn = msh_fn
#
#         return 0

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


if __name__ == "__main__":

    pass
