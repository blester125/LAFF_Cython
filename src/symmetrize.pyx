# !python
# cython: boundscheck=False, wraparound=False

from cython.parallel import prange
cimport numpy as np

ctypedef np.float_t DTYPE_t


cpdef np.ndarray[DTYPE_t, ndim=2] symmetrize_from_upper(
    np.ndarray[DTYPE_t, ndim=2] A
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


cpdef np.ndarray[DTYPE_t, ndim=2] symmetrize_from_lower(
    np.ndarray[DTYPE_t, ndim=2] A
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
