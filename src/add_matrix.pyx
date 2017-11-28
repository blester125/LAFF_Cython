import cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:, ::1] add_matrix(
        double[:, ::1] A,
        double[:, ::1] B
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
