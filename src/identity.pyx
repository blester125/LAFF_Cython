import cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:, ::1] identity(
        double[:, ::1] A
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
