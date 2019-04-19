# py_electrodes
classes and methods for creating electrode objects (solids with a voltage applied) using gmsh and OCC
## Setting up the Anaconda3 environment in Windows
Create an environment using the attached windows spec file:

``conda create --name pycontrolsystem --file spec-file-win.txt``

This configuration includes minGW, libpython, and m2w64-toolchain from msys2. Together, 
they let you compile cython code during installation.

Then install OCE and PythonOCC-Core (https://github.com/tpaviot/pythonocc-core). Unfortunately, we require 
pythonocc-core==0.18.2 which seems not directly supported for windows and python 3 
at the moment. This workaround lets us install it, but may break the conda environment.
Thus it is advisable to do it last, after installing all other packages with conda
(or just do it all on Ubuntu).

#### OCE
Download the latest Windows tarball with the right python version from here: https://anaconda.org/oce/oce/files

Open a Anaconda environment (from Anaconda Navigator), go to the Downloads folder and run
``conda install oce-0.18.3-vc14_3.win64.tar.bz2`` (or whichever file you downloaded).

#### PythonOCC-Core
Download the latest Windows tarball with the right python version from here: https://anaconda.org/tpaviot/pythonocc-core/files

Same process as above: ``conda install pythonocc-core-0.18.2-py36_vc14h24bf2e0_281.tar.bz2`` 
(or whichever file you downloaded).

_In a future release of OCE and PythonOCC-Core, it might work with a simple 
__conda install -c tpaviot -c oce pythonocc-core___

#### quaternions
``pip install numpy-quaternion``

#### dans_pymodules
__TODO: Remove dependency! It requires too many other packages (matplotlib, h5py, ...)__
 
``pip install git+https://github.com/DanielWinklehner/dans_pymodules.git``