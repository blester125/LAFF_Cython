import unittest
import numpy as np
from .transpose import transpose


class TransposeTest(unittest.TestCase):

    def setUp(self):
        self.rows = np.random.randint(1, 20)
        self.cols = np.random.randint(1, 20)
        self.A = np.random.randn(self.rows, self.cols)
        self.gold_matrix = np.copy(self.A)

    def test_1(self):
        a = transpose(self.A)
        np.testing.assert_allclose(a, self.gold_matrix.T)

if __name__ == "__main__":
    unittest.main()
