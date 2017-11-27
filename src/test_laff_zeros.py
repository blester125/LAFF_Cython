import unittest
import numpy as np
from .zeros import zeros


class LAFFZerosTest(unittest.TestCase):

    def setUp(self):
        self.rows = np.random.randint(1, 20)
        self.cols = np.random.randint(1, 20)
        self.A = np.random.randn(self.rows, self.cols)

    def test_zeros(self):
        a = zeros(self.A)
        assert np.sum(a) == 0
        np.testing.assert_allclose(a, np.zeros([self.rows, self.cols]))

    def test_shape(self):
        a = zeros(self.A)
        assert a.shape == (self.rows, self.cols)

if __name__ == "__main__":
    unittest.main()
