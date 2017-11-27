# !python
# cython: boundscheck=False, wraparound=False

from cython.parallel import prange
cimport numpy as np

ctypedef np.float_t DTYPE_t


cpdef np.ndarray[DTYPE_t, ndim=2] LowerTriangle(
    np.ndarray[DTYPE_t, ndim=2] A
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


cpdef np.ndarray[DTYPE_t, ndim=2] StrictlyLower(
    np.ndarray[DTYPE_t, ndim=2] A
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


cpdef np.ndarray[DTYPE_t, ndim=2] UnitLower(
    np.ndarray[DTYPE_t, ndim=2] A
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


cpdef np.ndarray[DTYPE_t, ndim=2] UpperTriangle(
    np.ndarray[DTYPE_t, ndim=2] A
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


cpdef np.ndarray[DTYPE_t, ndim=2] StrictlyUpper(
    np.ndarray[DTYPE_t, ndim=2] A
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


cpdef np.ndarray[DTYPE_t, ndim=2] UnitUpper(
    np.ndarray[DTYPE_t, ndim=2] A
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
