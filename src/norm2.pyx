import cython
cimport numpy as np
from libc.math cimport sqrt
from .dot import dot

ctypedef np.float_t DTYPE_t

cpdef double norm2(
        np.ndarray[DTYPE_t, ndim=2] x
) except -1:
    cdef double dot_product
    cdef double norm

    dot_product = dot(x, x)
    norm = sqrt(dot_product)

    return norm
