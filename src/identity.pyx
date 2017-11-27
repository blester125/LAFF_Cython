import cython
from cython.parallel import prange
cimport numpy as np

ctypedef np.float_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] identity(
    np.ndarray[DTYPE_t, ndim=2] A
):
    cdef int m_a = A.shape[0]
    cdef int n_a = A.shape[1]
    cdef int i, j

    if m_a != n_a:
        raise ValueError("A must be a square matrix.")

    for i in prange(m_a, nogil=True):
        for j in prange(n_a):
            if i == j:
                A[i, j] = 1
            else:
                A[i, j] = 0

    return A
