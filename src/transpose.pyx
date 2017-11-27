import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np

ctypedef np.float_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] transpose(
    np.ndarray[DTYPE_t, ndim=2] A
):
    cdef int m_a = A.shape[0]
    cdef int n_a = A.shape[1]
    cdef int i, j
    B = np.ndarray((n_a, m_a), dtype=np.float64)
    cdef DTYPE_t [:, :] B_view = B

    for i in prange(m_a, nogil=True):
        for j in prange(n_a):
            B_view[j, i] = A[i, j]

    return B
