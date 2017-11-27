import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
from .dot cimport dot

ctypedef np.float_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] m_v_mult(
    np.ndarray[DTYPE_t, ndim=2] A,
    np.ndarray[DTYPE_t, ndim=2] x
):
    cdef int m_a = A.shape[0]
    cdef int n_a = A.shape[1]
    cdef int m_x = x.shape[0]
    cdef int n_x = x.shape[1]
    cdef int i

    if n_a != m_x:
        raise ValueError("Inner columns are not equal. Got shapes {} and {}".format(n_a, m_x))

    out = np.ndarray((m_a, n_x), dtype=np.float64)
    cdef DTYPE_t [:, :] out_view = out

    for i in range(m_a):
        out[i, 0] = dot(A[np.newaxis, i, :], x)

    return out
