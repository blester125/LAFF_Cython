import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from LAFF.laff_dot import laff_dot

DTYPE = np.float32
ctypedef np.float_t DTYPE_t

cpdef double laff_norm2(
        np.ndarray[DTYPE_t, ndim=2] x
) except -1:
    cdef double dot
    cdef double norm

    dot = laff_dot(x, x)
    norm = sqrt(dot)

    return norm
