import cython
from libc.math cimport sqrt
from .dot cimport dot


cpdef double norm2(
        double[:, ::1] x
) except -1:
    cdef double dot_product
    cdef double norm

    dot_product = dot(x, x)
    norm = sqrt(dot_product)

    return norm
