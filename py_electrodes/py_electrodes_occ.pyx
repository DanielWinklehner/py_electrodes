import numpy as np
import quaternion
import os
import sys
cimport numpy as np
cimport cython

DTYPE1 = np.float64
DTYPE2 = np.int
ctypedef np.float64_t DTYPE1_t
ctypedef np.int_t DTYPE2_t

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

# --- Try importing pythonocc-core --- #
HAVE_OCC = False
try:
    from OCC.Extend.DataExchange import read_stl_file, read_step_file
    from OCC.Extend.TopologyUtils import TopologyExplorer
    from OCC.Core.BRepBndLib import brepbndlib_Add as bbox_add
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf, gp_Quaternion
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    from OCC.Core.TopAbs import TopAbs_ON, TopAbs_OUT, TopAbs_IN
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepTools import breptools_Read
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Transform
    from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut
    from OCC.Display.SimpleGui import init_display
    from OCC.Core.TopTools import TopTools_ListOfShape
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    HAVE_OCC = True
except ImportError:
    if DEBUG:
        print("Something went wrong during OCC import. No OpenCasCade support outside gmsh possible!")


class PyOCCElectrode(object):

    def __init__(self, elec=None, tolerance=1.0e-5, debug=False):

        self._debug = debug
        self._bbox_use_mesh = True
        self._elec = elec  # The electrode as OCC object
        self._bbox = None  # Bounding box
        self._socl = None  # Solid classifier
        self._bldr = BRep_Builder()
        # tolerance determines how far the bounding box will be
        # extending beyond the actual electrode limits
        self._tol = tolerance

        self._transformation = gp_Trsf()
        self._translation = gp_Vec(0.0, 0.0, 0.0)  # No translation by default
        self._rotation = gp_Quaternion(0.0, 0.0, 0.0, 1.0)  # No rotation by default

        # The same but as lists in case we split the electrode in
        # sub-electrodes for faster is-inside tests
        self._elec_s = None
        self._bbox_s = None
        self._socl_s = None
        self._n_s = 0

        if elec is not None:
            self._socl = BRepClass3d_SolidClassifier(self._elec)
            self._bbox = self.create_bbox(self._elec, self._tol, self._bbox_use_mesh)
            self._elec_s = [self._elec]
            self._bbox_s = [self._bbox]
            self._socl_s = [self._socl]
            self._n_s = 1

        self._filename = None

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, translation):

        translation = np.asarray(translation)

        if translation.shape == (3, ):
            self._translation = gp_Vec(translation[0],
                                       translation[1],
                                       translation[2])

    @property
    def rotation(self):
        """
        Cave: python quaternion has w as first entry, OCC quaternion as last!
        :return: rotation as a quaternion object.
        """
        rot = np.quaternion(self.rotation[3],
                            self.rotation[0],
                            self.rotation[1],
                            self.rotation[2])

        return rot

    @rotation.setter
    def rotation(self, rotation):
        """
        Cave: python quaternion has w as first entry, OCC quaternion as last!
        """
        assert isinstance(rotation, np.quaternion), "rotation has to be a python/numpy quaternion object!"

        self._rotation = gp_Quaternion(rotation.x, rotation.y, rotation.z, rotation.w)

    def get_bounds(self):
        """
        get the bounds of the full electrode (plus tolerance)
        TODO: Think about subtracting tolerance?
        :return: xmin, xmax, ymin, ymax, zmin, zmax
        """
        return self._bbox.Get()

    def get_bbox(self):
        return self._bbox

    def apply_second_transformation(self, translation=None, rotation=None):
        """
        ONLY USE THIS IF YOU KNOW WHAT YOU ARE DOING!
        :param translation:
        :param rotation:
        :return:
        """
        update = False

        # First apply rotation
        if rotation is None:
            self._transformation.SetRotation(rotation)
            self._elec = BRepBuilderAPI_Transform(self._elec, self._transformation).Shape()
            update = True

        # Then apply translation
        if translation is not None:
            self._transformation.SetTranslation(translation)
            self._elec = BRepBuilderAPI_Transform(self._elec, self._transformation).Shape()
            update = True

        if update:
            self._socl = BRepClass3d_SolidClassifier(self._elec)
            self._bbox = self.create_bbox(self._elec, self._tol, self._bbox_use_mesh)

            self._elec_s = [self._elec]
            self._bbox_s = [self._bbox]
            self._socl_s = [self._socl]
            self._n_s = 1

    def load_from_stl(self, filename=None):
        """
        STL is a mesh format. Segmentation will not work (yet) if the object is loaded as STL!
        :param filename:
        :return:
        """

        if filename is None:
            return 1

        assert os.path.splitext(filename)[1] == ".stl", "File extension does not match 'stl' file!"

        self._filename = filename

        print("Proc {} loading from STL file and generating OCC Object...".format(RANK))
        sys.stdout.flush()

        # Create the OCC solid + bbox
        self._elec = read_stl_file(filename)

        # First apply rotation
        self._transformation.SetRotation(self._rotation)
        self._elec = BRepBuilderAPI_Transform(self._elec, self._transformation).Shape()

        # Then apply translation
        self._transformation.SetTranslation(self._translation)
        self._elec = BRepBuilderAPI_Transform(self._elec, self._transformation).Shape()

        self._socl = BRepClass3d_SolidClassifier(self._elec)
        self._bbox = self.create_bbox(self._elec, self._tol, self._bbox_use_mesh)

        self._elec_s = [self._elec]
        self._bbox_s = [self._bbox]
        self._socl_s = [self._socl]
        self._n_s = 1

        return 0

    def load_from_brep(self, filename=None):
        """
        BREP is the native OpenCascade format for shapes/objects/solids, etc.
        :param filename:
        :return:
        """

        if filename is None:
            return 1

        assert os.path.splitext(filename)[1] == ".brep", "File extension does not match 'brep' file!"

        self._filename = filename

        print("Proc {} loading from BREP file and generating OCC Object...".format(RANK))
        sys.stdout.flush()

        load_shape = TopoDS_Shape()
        breptools_Read(load_shape, filename, self._bldr)

        # The loaded shape is a compound of solid, shell, surfs, lines and points. We select the solid only.
        # TODO: some assertions, right now we put a lot of faith in the user.
        _te = TopologyExplorer(load_shape)

        # Here we make our way down from solids to surfaces to edges
        if len(list(_te.solids())) > 0:
            self._elec = list(_te.solids())[0]  # TODO: What if there are more than 1 solid? -DW
        elif len(list(_te.faces())) > 0:
            self._elec = list(_te.faces())[0]
        elif len(list(_te.edges())) > 0:
            self._elec = list(_te.edges())[0]
        else:
            print("Couldn't load from brep file")
            print("Number of faces: {}".format(_te.number_of_faces()))
            print("Number of vertices: {}".format(_te.number_of_vertices()))
            print("Number of wires: {}".format(_te.number_of_wires()))
            print("Number of edges: {}".format(_te.number_of_edges()))
            print("Number of shells: {}".format(_te.number_of_shells()))
            print("Number of solids: {}".format(_te.number_of_solids()))
            print("Number of compounds: {}".format(_te.number_of_compounds()))
            print("Number of compound solids: {}".format(_te.number_of_comp_solids()))

        # First apply rotation
        self._transformation.SetRotation(self._rotation)
        self._elec = BRepBuilderAPI_Transform(self._elec, self._transformation).Shape()

        # Then apply translation
        self._transformation.SetTranslation(self._translation)
        self._elec = BRepBuilderAPI_Transform(self._elec, self._transformation).Shape()

        self._socl = BRepClass3d_SolidClassifier(self._elec)
        self._bbox = self.create_bbox(self._elec, self._tol, self._bbox_use_mesh)

        self._elec_s = [self._elec]
        self._bbox_s = [self._bbox]
        self._socl_s = [self._socl]
        self._n_s = 1

        return 0

    # def load_from_step(self, filename=None):
    #
    #     if filename is None:
    #         return 1
    #
    #     assert os.path.splitext(filename)[1] in [".stp", ".step"], "File extension does not match 'step' file!"
    #
    #     self._filename = filename
    #
    #     print("Proc {} loading from STEP file and generating OCC Object...".format(RANK))
    #     sys.stdout.flush()
    #
    #     self._elec = read_step_file(filename)
    #     self._socl = BRepClass3d_SolidClassifier(self._elec)
    #     self._bbox = self.create_bbox(self._elec, self._tol)
    #
    #     return 0

    @staticmethod
    def create_bbox(elec, tol, use_mesh=False):

        _bbox = Bnd_Box()
        _bbox.SetGap(tol)

        if use_mesh:
            mesh = BRepMesh_IncrementalMesh()
            mesh.SetParallel(True)
            mesh.SetShape(elec)
            mesh.Perform()

            assert mesh.IsDone()

        bbox_add(elec, _bbox, use_mesh)

        return _bbox

    def partition_z(self, nslices=1):
        """
        Partition the electrode in n pieces along the z axix
        :param nslices:
        :return:
        """
        if RANK == 0:
            print("Partitioning the electrode in {} pieces along z.".format(nslices))

        cdef int n = nslices
        self._n_s = n

        self._elec_s = []
        self._bbox_s = []
        self._socl_s = []

        # We are trusting that the bounding box really is larger than the electrode in every direction
        cdef float xmin = 0
        cdef float ymin = 0
        cdef float zmin = 0
        cdef float xmax = 0
        cdef float ymax = 0
        cdef float zmax = 0
        xmin, ymin, zmin, xmax, ymax, zmax = self.get_bounds()

        cdef float length = zmax - zmin
        cdef float dz = length / n

        remainder_shp = self._elec
        cdef float box_z_s = zmin
        cdef float box_z_e = zmin

        cdef int i = 0
        for i in range(n - 1):

            # _box is the piece to be cut off this iteration
            box_z_s = box_z_e
            box_z_e = box_z_s + dz
            _box_shp = BRepPrimAPI_MakeBox(gp_Pnt(xmin, ymin, box_z_s), gp_Pnt(xmax, ymax, box_z_e)).Shape()

            new_piece = BRepAlgoAPI_Common(remainder_shp, _box_shp)
            assert new_piece.IsDone()
            new_piece_shp = new_piece.Shape()

            self._elec_s.append(new_piece_shp)
            self._bbox_s.append(self.create_bbox(new_piece_shp, self._tol, use_mesh=self._bbox_use_mesh))
            self._socl_s.append(BRepClass3d_SolidClassifier(new_piece_shp))

            remainder = BRepAlgoAPI_Cut(remainder_shp, _box_shp)
            assert remainder.IsDone()
            remainder_shp = remainder.Shape()

        self._elec_s.append(remainder_shp)
        self._bbox_s.append(self.create_bbox(remainder_shp, self._tol, use_mesh=self._bbox_use_mesh))
        self._socl_s.append(BRepClass3d_SolidClassifier(remainder_shp))

        return 0

    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def points_inside(self, np.ndarray[DTYPE1_t, ndim=2] global_pts):
        """
        Function that calculates whether the point(s) are inside the electrode or not.
        Currently this only works with pythonocc-core installed and can be slow
        for a large number of points.
        :param points: numpy ndarray with dtype=float64 of shape (N, 3)
        :return: boolean numpy array of True or False depending on whether the points are inside or
                 outside (on the surface is counted as inside!)

        Note: If we are running on a single processor, fall back to multiprocessing...need to figure out pickle for
        OCC objects for that... :(
        """

        assert self._elec is not None, "No electrode loaded!"
        assert global_pts.dtype == DTYPE1, "lobal_pts must be numpy ndarray of dtype float64"

        if RANK == 0 and self._debug:
            print("Running points_inside test on {} processor(s)".format(SIZE))

        # Divide points array into pieces to be worked on in parallel (if we have MPI/multiprocessing)
        cdef int npts = global_pts.shape[0]
        cdef int pts_per_proc = npts / SIZE
        cdef int n_slices = self._n_s

        cdef int start = RANK * pts_per_proc
        cdef int end = (RANK + 1) * pts_per_proc

        # Highest proc needs to make sure last points are included
        if RANK == SIZE - 1:
            end = npts + 1
        cdef np.ndarray[DTYPE1_t, ndim=2] local_pts = global_pts[start:end]

        cdef int n_loc_pts = local_pts.shape[0]  # Can't be pts_per_proc because of the last proc havng more pts
        cdef np.ndarray[DTYPE2_t, ndim=1] local_mask = np.zeros(n_loc_pts, dtype=DTYPE2)

        if self._debug:
            print("Host {}, proc {} of {}, local_pts =\n".format(HOST, RANK + 1, SIZE), local_pts)

        cdef int i = 0
        cdef int j = 0

        if self._n_s == 1:

            for i in range(n_loc_pts):

                pt = gp_Pnt(local_pts[i, 0],
                            local_pts[i, 1],
                            local_pts[i, 2])

                if not self._bbox.IsOut(pt):

                    self._socl.Perform(pt, self._tol)

                    if self._socl.State() in [TopAbs_ON, TopAbs_IN]:

                        local_mask[i] = 1

        elif self._n_s > 1:

            for i in range(n_loc_pts):

                pt = gp_Pnt(local_pts[i, 0],
                            local_pts[i, 1],
                            local_pts[i, 2])

                if not self._bbox.IsOut(pt):

                    for j in range(n_slices):

                        if not self._bbox_s[j].IsOut(pt):

                            self._socl_s[j].Perform(pt, self._tol)

                            if self._socl_s[j].State() in [TopAbs_ON, TopAbs_IN]:

                                local_mask[i] = 1

        else:
            print("It seems like there is no electrode data loaded, aborting.")
            return None

        if SIZE != 1 and MPI is not None:
            mask = np.concatenate(COMM.allgather(local_mask), axis=0)
        else:
            mask = local_mask

        return np.array(mask, dtype=np.bool)

    def show(self, display=None, color="RED"):

        if display is None:

            display, start_display, _, _ = init_display()

            display.set_bg_gradient_color([175, 210, 255], [255, 255, 255])
            ais_shape = display.DisplayShape(self._elec, color=color, update=True)

            start_display()

        else:

            ais_shape = display.DisplayShape(self._elec, color=color, update=False)

        return display, ais_shape


if __name__ == "__main__":

    pass
