import unittest
import numpy as np
from .zeros import zeros


class LAFFOnesTest(unittest.TestCase):

    def setUp(self):
        self.number_or_rows = np.random.randint(1, 20)
        self.number_or_cols = np.random.randint(1, 20)
        self.A = np.random.randn(self.number_or_rows, self.number_or_cols)

    def test_ones(self):
        a = zeros(self.A)
        assert np.sum(a) == 0
        np.testing.assert_allclose(a, np.zeros([self.number_or_rows, self.number_or_cols]))


if __name__ == "__main__":
    unittest.main()
