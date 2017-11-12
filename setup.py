from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


ext_modules = [
    Extension(
        "*",
        sources=["*.pyx"],
        libraries=["m"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    ext_modules=cythonize(ext_modules)
)
