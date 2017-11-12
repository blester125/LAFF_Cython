import cython
import numpy as np
cimport numpy as np

DTYPE = np.float32
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
        for i in range(m_x):
            x[i, 0] = x[i, 0] * alpha
    else:
        for i in range(n_x):
            x[0, i] = x[0, i] * alpha

    return x
