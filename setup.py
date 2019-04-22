from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(name='py_electrodes',
      version='1.0.0',
      python_requires='>=3',
      description='This module handles geometry objects using gmsh and OpenCasCade to be used as electrodes',
      url='https://github.com/DanielWinklehner/py_electrodes',
      author='Daniel Winklehner',
      author_email='winklehn@mit.edu',
      license='MIT',
      package_dir={'': ''},
      packages=['py_electrodes'],
      package_data={'py_electrodes': ['Examples/*', 'Settings.txt']},
      include_package_data=True,
      ext_modules=cythonize("py_electrodes/py_electrodes_occ.pyx"),
      include_dirs=[numpy.get_include()],
      zip_safe=False,
      )
