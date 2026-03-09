# py_electrodes
classes and methods for creating electrode objects (solids with a voltage applied) using 
[gmsh](https://gmsh.info/) 
and [pythonocc-core](https://github.com/tpaviot/pythonocc-core). Including fast methods to find segment/ray - triangle/boundary intersections. 

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

## Setting up the Anaconda3 environment (tested in Windows 11 and WSL2/Ubuntu 22.04)

The simplest setup is using conda (Anaconda3-tested) The file _conda-env.yml_ 
can be found in /PyPATools/documents.

Simply create a new Anaconda3 environment from the Navigator (Environments-->Import)
or from the command line

```bash
conda env create -f conda-env.yml
```

The environment name can be changed in the yml file or with the _--name_ flag.

### Install PyPATools

_Note: there is a co-dependency between PyPATools and py_electrodes. Please install both for full functionality._

Either:

```bash
git clone https://github.com/DanielWinklehner/PyPATools.git
cd PyPATools
pip install -e .

git clone https://github.com/DanielWinklehner/py_electrodes.git
cd py_electrodes
pip install -e .
```

Or directly (if you don't need the source files and examples):

```bash
pip install git+https://github.com/DanielWinklehner/PyPATools.git
pip install git+https://github.com/DanielWinklehner/py_electrodes.git
```

### Additional notes for Windows

A community edition of MS Visual Studio (2017 or newer) is also needed to compile 
some of the c-extensions for cython. 
Get it for free here: https://visualstudio.microsoft.com/downloads/

### Additional notes for Ubuntu 18 (WSL and local installation)
In most cases the c compilers for cython are included in the Ubuntu installation.
If not, the build-essential package should have what is needed.

```bash
sudo apt-get install build-essential
```

It looks like the 3D rendering drivers are missing in a vanilla Ubuntu 18 installation, 
install them using:

```bash
sudo apt-get install libglu1-mesa
````
