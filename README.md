# py_electrodes
classes and methods for creating electrode objects (solids with a voltage applied) using gmsh and OCC

## The Settings.txt file
During installation a Settings.txt file is created in
the same directory that settings.py is being installed to 
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

__WARNING: This used to work until 08/22/2019. Now there is a persistent OCC error that 
we haven't figured out yet! Currently, it is not working under Windows. :(__

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

## Setting up the Anaconda3 environment in Ubuntu 18
Create an environment using the attached ubuntu spec file:

``conda create --name py_electrodes_env --file spec-file-win.txt``

It looks like the 3D rendering drivers are missing in a vanilla Ubuntu 18 installation, install them using:

``sudo apt-get install libglu1-mesa``

Note: OCE and Pythonocc-Core are included in anaconda on ubuntu.

## Dark theme
On both Windows and Linux, the qdarkstyle theme can be installed using

``pip install qdarkstyle==2.6.8``

Note: the newest version (2.7) has some bugs (missing frames), so we are sticking with 2.6.8 for now.