import unittest
import numpy as np
from .add_matrix import add_matrix


class LaffAddMatrixTest(unittest.TestCase):

    def setUp(self):
        self.rows = np.random.randint(5, 20)
        self.cols = np.random.randint(5, 20)
        self.A = np.random.randn(self.rows, self.cols)
        self.gold_a = np.copy(self.A)
        self.B = np.random.randn(self.rows, self.cols)
        self.gold_b = np.copy(self.B)

    def test_add(self):
        C = add_matrix(self.A, self.B)
        np.testing.assert_allclose(C, self.gold_a + self.gold_b)

    def test_shape_preserved(self):
        C = add_matrix(self.A, self.B)
        assert C.shape == self.gold_a.shape

    def test_on_bad_shape(self):
        diff = 0
        while diff == 0:
            diff = np.random.randint(-4, 5)
        input_matrix = np.random.randn(self.rows, self.cols + diff)
        self.assertRaises(
            Exception, add_matrix, self.A, input_matrix
        )
