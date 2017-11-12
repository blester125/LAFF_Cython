# LAFF Assignments in python/cython
##

build with `python setup.py build_ext --inplace`

test with `python -m unittest discover -p 'test_*.py'`

Note: Currently Cython uses a depreciated numpy API which will throw a warning during compliation. This is being worked on by the cython developers and can be saftly ignored.
