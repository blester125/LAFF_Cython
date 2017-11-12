import cython
import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] laff_axpy(
        float alpha,
        np.ndarray[DTYPE_t, ndim=2] x,
        np.ndarray[DTYPE_t, ndim=2] y
):
    cdef int m_x = x.shape[0]
    cdef int n_x = x.shape[1]
    cdef int m_y = y.shape[0]
    cdef int n_y = y.shape[1]
    cdef int i

    if m_x != 1 and n_x != 1:
        raise ValueError("x must be a row or column vector.")
    if m_y != 1 and n_y != 1:
        raise ValueError("y must be a row or column vector.")
    if m_x * n_x != m_y * n_y:
        raise ValueError("x and y must be the same size.")

    if n_x == 1:
        if n_y == 1:
            for i in range(x.shape[0]):
                y[i, 0] = alpha * x[i, 0] + y[i, 0]
        else:
            for i in range(x.shape[0]):
                y[0, i] = alpha * x[i, 0] + y[0, i]
    else:
        if n_y == 1:
            for i in range(x.shape[1]):
                y[i, 0] = alpha * x[0, i] + y[i, 0]
        else:
            for i in range(x.shape[1]):
                y[0, i] = alpha * x[0, i] + y[0, i]

    return y
