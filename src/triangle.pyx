# !python
# cython: boundscheck=False, wraparound=False

from cython.parallel import prange


cpdef double[:, ::1] LowerTriangle(
        double[:, ::1] A
):
    cdef int m_x = A.shape[0]
    cdef int n_x = A.shape[1]
    cdef int i, j

    if m_x != n_x:
        raise ValueError("A must be a square matrix.")

    for i in prange(m_x, nogil=True):
        for j in prange(n_x):
            if i < j:
                A[i, j] = 0

    return A


cpdef double[:, ::1] StrictlyLower(
        double[:, ::1] A
):
    cdef int m_x = A.shape[0]
    cdef int n_x = A.shape[1]
    cdef int i, j

    if m_x != n_x:
        raise ValueError("A must be a square matrix.")

    for i in prange(m_x, nogil=True):
        for j in prange(n_x):
            if i <= j:
                A[i, j] = 0

    return A


cpdef double[:, ::1] UnitLower(
        double[:, ::1] A
):
    cdef int m_x = A.shape[0]
    cdef int n_x = A.shape[1]
    cdef int i, j

    if m_x != n_x:
        raise ValueError("A must be a square matrix.")

    for i in prange(m_x, nogil=True):
        for j in prange(n_x):
            if i == j:
                A[i, j] = 1
            elif i < j:
                A[i, j] = 0

    return A


cpdef double[:, ::1] UpperTriangle(
        double[:, ::1] A
):
    cdef int m_x = A.shape[0]
    cdef int n_x = A.shape[1]
    cdef int i, j

    if m_x != n_x:
        raise ValueError("A must be a square matrix")

    for i in prange(m_x, nogil=True):
        for j in prange(n_x):
            if i > j:
                A[i, j] = 0

    return A


cpdef double[:, ::1] StrictlyUpper(
        double[:, ::1] A
):
    cdef int m_x = A.shape[0]
    cdef int n_x = A.shape[1]
    cdef int i, j

    if m_x != n_x:
        raise ValueError("A must be a square matrix.")

    for i in prange(m_x, nogil=True):
        for j in prange(n_x):
            if i >= j:
                A[i, j] = 0

    return A


cpdef double[:, ::1] UnitUpper(
        double[:, ::1] A
):
    cdef int m_x = A.shape[0]
    cdef int n_x = A.shape[1]
    cdef int i, j

    if m_x != n_x:
        raise ValueError("A must be a square matrix.")

    for i in prange(m_x, nogil=True):
        for j in prange(n_x):
            if i == j:
                A[i, j] = 1
            elif i > j:
                A[i, j] = 0

    return A
