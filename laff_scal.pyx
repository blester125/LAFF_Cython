import cython
from cython.parallel import prange
cimport numpy as np

ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def laff_scal(
        float alpha,
        np.ndarray[DTYPE_t, ndim=2] x
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
