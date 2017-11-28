# !python
# cython: boundscheck=False, wraparound=False

from cython.parallel import prange


cpdef double[:, ::1] symmetrize_from_upper(
        double[:, ::1] A
):
    cdef int m_a = A.shape[0]
    cdef int n_a = A.shape[1]
    cdef int i, j

    if m_a != n_a:
        raise ValueError("A must be a square matrix.")

    for i in prange(m_a, nogil=True):
        for j in prange(i):
            A[i, j] = A[j, i]

    return A


cpdef double[:, ::1] symmetrize_from_lower(
        double[:, ::1] A
):
    cdef int m_a = A.shape[0]
    cdef int n_a = A.shape[1]
    cdef int i, j

    if m_a != n_a:
        raise ValueError("A must be a square matrix.")

    for i in prange(m_a, nogil=True):
        for j in prange(i, n_a):
            A[i, j] = A[j, i]

    return A
