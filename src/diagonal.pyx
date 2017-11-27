import cython
from cython.parallel import prange
cimport numpy as np

ctypedef np.float_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] diagonal(
    np.ndarray[DTYPE_t, ndim=2] A,
    np.ndarray[DTYPE_t, ndim=2] x
):
    cdef int m_a = A.shape[0]
    cdef int n_a = A.shape[1]
    cdef int m_x = x.shape[0]
    cdef int n_x = x.shape[1]
    cdef int i, j

    if m_x != 1 and n_x != 1:
        raise ValueError("x must be a vector.")
    if m_a != n_a:
        raise ValueError("A must be a square Matrix.")
    if n_x == 1:
        if m_a != m_x:
            raise ValueError("x must be the same size as A")
    else:
        if m_a != n_x:
            raise ValueError("x must be the same size as A")

    if n_x == 1:
        for i in prange(m_a, nogil=True):
            for j in prange(n_a):
                if i == j:
                    A[i, j] = x[i, 0]
                else:
                    A[i, j] = 0
    else:
        for i in prange(m_a, nogil=True):
            for j in prange(n_a):
                if i == j:
                    A[i, j] = x[0, i]
                else:
                    A[i, j] = 0

    return A
