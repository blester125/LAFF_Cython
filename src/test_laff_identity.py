import unittest
import numpy as np
from .identity import identity


class IdentityTest(unittest.TestCase):

    def setUp(self):
        self.matrix_size = np.random.randint(5, 20)
        self.A = np.random.randn(self.matrix_size, self.matrix_size)

    def test_identity(self):
        a = identity(self.A)
        assert np.sum(a) == self.matrix_size
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if i == j:
                    assert a[i, j] == 1
                else:
                    assert a[i, j] == 0

    def test_shape(self):
        a = identity(self.A)
        assert a.shape == (self.matrix_size, self.matrix_size)

    def test_error_on_non_square(self):
        diff = 0
        while diff == 0:
            diff = np.random.randint(-4, 5)
        input_matrix = np.random.randn(self.matrix_size, self.matrix_size + diff)
        self.assertRaises(
            Exception, identity, input_matrix
        )


if __name__ == "__main__":
    unittest.main()
