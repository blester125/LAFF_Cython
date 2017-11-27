import unittest
import numpy as np
from .ones import ones


class LAFFOnesTest(unittest.TestCase):

    def setUp(self):
        self.rows = np.random.randint(1, 20)
        self.cols = np.random.randint(1, 20)
        self.A = np.random.randn(self.rows, self.cols)

    def test_ones(self):
        a = ones(self.A)
        assert np.sum(a) == self.rows * self.cols
        np.testing.assert_allclose(a, np.ones([self.rows, self.cols]))

    def test_shape(self):
        a = ones(self.A)
        assert a.shape == (self.rows, self.cols)

if __name__ == "__main__":
    unittest.main()
