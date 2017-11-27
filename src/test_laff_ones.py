import unittest
import numpy as np
from .ones import ones


class LAFFOnesTest(unittest.TestCase):

    def setUp(self):
        self.number_or_rows = np.random.randint(1, 20)
        self.number_or_cols = np.random.randint(1, 20)
        self.A = np.random.randn(self.number_or_rows, self.number_or_cols)

    def test_ones(self):
        a = ones(self.A)
        assert np.sum(a) == self.number_or_rows * self.number_or_cols
        np.testing.assert_allclose(a, np.ones([self.number_or_rows, self.number_or_cols]))


if __name__ == "__main__":
    unittest.main()
