from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


ext_modules = [
    Extension(
        "*",
        sources=["src/*.pyx"],
        libraries=["m"],
        include_dirs=[numpy.get_include()],
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp']
    )
]

setup(
    ext_modules=cythonize(ext_modules)
)
