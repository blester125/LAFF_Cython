import cython
from cython.parallel import prange
import numpy as np
from .dot cimport dot


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:, ::1] m_v_mult(
        double[:, ::1] A,
        double[:, ::1] x
):
    cdef int m_a = A.shape[0]
    cdef int n_a = A.shape[1]
    cdef int m_x = x.shape[0]
    cdef int n_x = x.shape[1]
    cdef int i

    if n_a != m_x:
        raise ValueError("Inner columns are not equal. Got shapes {} and {}".format(n_a, m_x))

    out = np.ndarray((m_a, n_x), dtype=np.float64)
    cdef double[:, ::1] out_view = out

    for i in range(m_a):
        out[i, 0] = dot(A[None, i, ::1].copy(), x)

    return out
