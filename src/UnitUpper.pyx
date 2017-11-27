import cython
from cython.parallel import prange
cimport numpy as np

ctypedef np.float_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] UnitUpper(
    np.ndarray[DTYPE_t, ndim=2] A
):
    cdef int m_x = A.shape[0]
    cdef int n_x = A.shape[1]
    cdef int i, j

    for i in prange(m_x, nogil=True):
        for j in prange(n_x):
            if i == j:
                A[i, j] = 1
            elif i > j:
                A[i, j] = 0

    return A
