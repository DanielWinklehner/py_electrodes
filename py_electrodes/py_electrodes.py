import numpy as np
import sys
import os
import uuid
from .py_electrodes_occ import *

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
    os.rmdir(TEMP_DIR)
    os.mkdir(TEMP_DIR)

# Gmsh path to executable
GMSH_EXE = "gmsh"
# GMSH_EXE = "E:/gmsh4/gmsh.exe"

__author__ = "Daniel Winklehner"
__doc__ = """Create electrodes using gmsh and pythonocc-core for use in field calculations and particle tracking"""

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


class PyElectrode(object):
    def __init__(self,
                 name,
                 voltage=0,
                 geo_str=None):

        print(TEMP_DIR)

        self._id = uuid.uuid1()
        self._name = name
        self._voltage = voltage
        self._debug = DEBUG

        self._originated_from = ""
        self._orig_file = None
        self._geo_str = geo_str
        self._occ_obj = None

        if self._geo_str is not None:
            self._originated_from = "geo_str"
            self.generate_from_geo_str(self._geo_str)

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def voltage(self):
        return self._voltage

    def generate_from_geo_str(self, geo_str=None):
        if geo_str is not None:
            self._geo_str = geo_str

        if self._geo_str is not None:
            self._originated_from = "geo_str"
            self._orig_file = os.path.join(TEMP_DIR, "{}.geo".format(self._id))
            with open(self._orig_file, 'w') as _of:
                _of.write(self._geo_str)
            self._generate_from_geo()

        return 0

    def generate_from_file(self, filename=None):
        """
        Loads the electrode object from file. Extension can be .brep, .geo, .stl, .stp/.step
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

            assert ext in [".brep", ".geo", ".stl", ".stp", ".step"], \
                "Extension has to be .brep, .geo, .stl, .stp/.step!"

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
            elif ext.lower() in [".stp", ".step"]:
                self._originated_from = "step"
                self._generate_from_step()

            return 0

        else:

            return 1

    def _generate_from_brep(self):
        if DEBUG:
            print("Generating from brep")

    def _generate_from_geo(self):
        if DEBUG:
            print("Generating from geo")

    def _generate_from_stl(self):
        if DEBUG:
            print("Generating from stl")

    def _generate_from_step(self):
        if DEBUG:
            print("Generating from step")

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

    # def generate_occ(self):
    #
    #     if HAVE_OCC:
    #
    #         tmp_dir = self._parent.temp_dir
    #         brep_fn = os.path.join(tmp_dir, "{}.brep".format(self._name))
    #
    #         self._occ_obj = PyOCCElectrode()
    #         self._occ_obj.load_from_brep(brep_fn)
    #         self._occ_obj.partition_z(self._occ_npart)
    #
    #         return 0
    #
    #     else:
    #
    #         print("Couldn't load PythonOCC-Core earlier, cannot create OpenCasCade object!")
    #
    #         return 1

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
