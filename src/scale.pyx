# !python
# cython: boundscheck=False, wraparound=False

from cython.parallel import prange


cpdef double[:, ::1] scale(
        float alpha,
        double[:, ::1] x
):
    cdef int m_x = x.shape[0]
    cdef int n_x = x.shape[1]
    cdef int i

    if m_x != 1 and n_x != 1:
        raise ValueError("x must be a row or column vector.")

    if n_x == 1:
        for i in prange(m_x, nogil=True):
            x[i, 0] = x[i, 0] * alpha
    else:
        for i in prange(n_x, nogil=True):
            x[0, i] = x[0, i] * alpha

    return x


cpdef double[:, ::1] scale_matrix(
        float alpha,
        double[:, ::1] A
):
    cdef int m_a = A.shape[0]
    cdef int n_a = A.shape[1]
    cdef int i, j

    for i in prange(m_a, nogil=True):
        for j in prange(n_a):
            A[i, j] = alpha * A[i, j]

    return A
