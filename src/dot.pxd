cimport numpy as np

ctypedef np.float_t DTYPE_t

cpdef double dot(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=2] y) except -1
