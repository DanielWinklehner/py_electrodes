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

## Setting up the Anaconda3 environment

With the new version of OpenCascade (OCC) and 
[pythonocc-core](https://github.com/tpaviot/pythonocc-core) (7.4.0), a simple 
anaconda installation from yml file is possible. The file _environment.yml_ 
can be found in /py_electrodes/documents. 

simply create a new Anaconda3 environment from the Navigator (Environments-->Import)
or from the command line

``conda env create -f environment.yml``

The environment name can be changed in the yml file or with the _-name_ flag.

### Additional notes for Windows

A community edition of MS Visual Studio (2017 or newer) is also needed to compile 
some of the c-extensions for cython. 
Get it for free here: https://visualstudio.microsoft.com/downloads/

### Additional notes for Ubuntu 18 (WSL and local installation)
In most cases the c compilers for cython are included in the Ubuntu installation.
If not, the build-essential package should have what is needed.

``sudo apt-get install build-essential``

It looks like the 3D rendering drivers are missing in a vanilla Ubuntu 18 installation, 
install them using:

``sudo apt-get install libglu1-mesa``
