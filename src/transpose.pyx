import cython
from cython.parallel import prange
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:, ::1] transpose(
       double[:, ::1] A
):
    cdef int m_a = A.shape[0]
    cdef int n_a = A.shape[1]
    cdef int i, j

    B = np.ndarray((n_a, m_a), dtype=np.float64)
    cdef double[:, :] B_view = B

    for i in prange(m_a, nogil=True):
        for j in prange(n_a):
            B_view[j, i] = A[i, j]

    return B
