import unittest
import numpy as np
from .scale import scale, scale_matrix


class LaffScaleTest(unittest.TestCase):

    def setUp(self):
        real_length = np.random.randint(1, 20)
        self.x = np.random.uniform(0, 10, real_length)
        self.x = np.reshape(self.x, [1, real_length])
        self.alpha = float(np.random.randint(-5, 6))

    def test_column_scale(self):
        gold = self.alpha * self.x
        np.testing.assert_allclose(scale(self.alpha, self.x), gold)

    def test_row_scale(self):
        gold = self.alpha * self.x.T
        np.testing.assert_allclose(scale(self.alpha, self.x.T), gold)

    def test_bad_alpha(self):
        self.assertRaises(Exception, scale, np.array([3, 4]), self.x)

    def test_bad_x(self):
        bad_x = np.random.randn(2, 2)
        self.assertRaises(Exception, scale, self.alpha, bad_x)


class LaffScaleMatrixTest(unittest.TestCase):

    def setUp(self):
        self.rows = np.random.randint(5, 20)
        self.cols = np.random.randint(5, 20)
        self.A = np.random.randn(self.rows, self.cols)
        self.gold_matrix = np.copy(self.A)
        self.alpha = np.random.rand()

    def test_scale(self):
        scaled = scale_matrix(self.alpha, self.A)
        np.testing.assert_allclose(scaled, self.alpha * self.gold_matrix)

    def test_shape_preserved(self):
        scaled = scale_matrix(self.alpha, self.A)
        assert scaled.shape == self.gold_matrix.shape

    def test_bad_alpha(self):
        bad_alpha = np.array([3, 4])
        self.assertRaises(Exception, scale_matrix, bad_alpha, self.A)

if __name__ == "__main__":
    unittest.main()
