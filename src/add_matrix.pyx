import cython
from cython.parallel import prange
cimport numpy as np

ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] add_matrix(
    np.ndarray[DTYPE_t, ndim=2] A,
    np.ndarray[DTYPE_t, ndim=2] B
):
    cdef int m_a = A.shape[0]
    cdef int n_a = A.shape[1]
    cdef int m_b = B.shape[0]
    cdef int n_b = B.shape[1]
    cdef int i, j

    if m_a != m_b or n_a != n_b:
        raise ValueError("A and B must be the same shape got A: [{}, {}] and B: [{}, {}}]".format(m_a, n_a, m_b, n_b))

    for i in prange(m_a, nogil=True):
        for j in prange(n_a):
            A[i, j] += B[i, j]

    return A
