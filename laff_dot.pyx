import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double laff_dot(
        np.ndarray[DTYPE_t, ndim=2] x,
        np.ndarray[DTYPE_t, ndim=2] y
) except -1:
    cdef int m_x = x.shape[0]
    cdef int n_x = x.shape[1]
    cdef int m_y = y.shape[0]
    cdef int n_y = y.shape[1]
    cdef int i
    cdef double dot_product

    if m_x != 1 and n_x != 1:
        raise ValueError("x must be a row or column vector.")
    if m_y != 1 and n_y != 1:
        raise ValueError("y must be a row or column vector.")
    if m_x * n_x != m_y * n_y:
        raise ValueError("x and y must be the same size.")

    dot_product = 0.0

    if n_x == 1:
        if n_y == 1:
            for i in prange(m_x, nogil=True):
                dot_product += y[i, 0] * x[i, 0]
        else:
            for i in prange(m_x, nogil=True):
                dot_product += y[0, i] * x[i, 0]
    else:
        if n_y == 1:
            for i in prange(n_x, nogil=True):
                dot_product += y[i, 0] * x[0, i]
        else:
            for i in prange(n_x, nogil=True):
                dot_product += y[0, i] * x[0, i]

    return dot_product
