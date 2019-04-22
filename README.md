# py_electrodes
classes and methods for creating electrode objects (solids with a voltage applied) using gmsh and OCC

## The Settings.txt file
When the py_electrodes.py script is first loaded, it creates a Settings.txt file in
the same directory that settings.py has been installed to 
(typically in _.../site-packages/py_electrodes/py_electrodes/_). The settings handler 
will look for this Settings.txt file every time the script is loaded. The user can change 
settings in this file. Better would be to copy it to a local directory.

In Windows, this should be:

_$APPDATA\py_electrodes\Settings.txt_ 

(typically this is _C:\Users\<Username>\AppData\Roaming\py_electrodes_). 
The _py_electrodes_ subdirectory has to be created manually.

In Linux it should be:

_$HOME/.local/py_electrodes/Settings.txt_

The settings handler wil look in those directories first and in the package path 
second. 

## Setting up the Anaconda3 environment in Windows
Create an environment using the attached windows spec file:

``conda create --name py_electrodes_env --file spec-file-win.txt``

This configuration includes minGW, libpython, and m2w64-toolchain from msys2. Together, 
they let you compile cython code during installation.

Then install OCE and PythonOCC-Core (https://github.com/tpaviot/pythonocc-core). 
Unfortunately, we require pythonocc-core==0.18.2 which seems not directly supported 
for windows and python 3 at the moment. This workaround lets us install it, 
but may break the conda environment.
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