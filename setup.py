from distutils.core import setup
from Cython.Build import cythonize


setup(name='py_electrodes',
      version='0.0.2',
      description='This module handles geometry objects using gmsh and OpenCasCade to be used as electrodes',
      url='https://github.com/DanielWinklehner/py_electrodes',
      author='Daniel Winklehner',
      author_email='winklehn@mit.edu',
      license='MIT',
      packages=['py_electrodes'],
      ext_modules=cythonize("py_electrodes/py_electrodes_occ.pyx"),
      zip_safe=False)
