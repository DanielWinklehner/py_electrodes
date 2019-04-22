from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(name='py_electrodes',
      version='1.0.0',
      description='This module handles geometry objects using gmsh and OpenCasCade to be used as electrodes',
      url='https://github.com/DanielWinklehner/py_electrodes',
      author='Daniel Winklehner',
      author_email='winklehn@mit.edu',
      license='MIT',
      packages=['py_electrodes'],
      ext_modules=cythonize("py_electrodes/py_electrodes_occ.pyx"),
      include_dirs=[numpy.get_include()],
      zip_safe=False)
