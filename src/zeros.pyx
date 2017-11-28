import cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:, ::1] zeros(
        double[:, ::1] A
):
    cdef int m_x = A.shape[0]
    cdef int n_x = A.shape[1]
    cdef int i, j

    for i in prange(m_x, nogil=True):
        for j in prange(n_x):
            A[i, j] = 0

    return A
