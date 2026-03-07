"""
Minimal setup.py for Cython extension compilation.
Works with pyproject.toml via setuptools.build_meta backend.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="py_electrodes.py_electrodes_occ",
        sources=["py_electrodes/py_electrodes_occ.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
    )
]

setup(
    name="py_electrodes",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
        force=False,
    ),
)