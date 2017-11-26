import unittest
import numpy as np
from LAFF.axpy import axpy


class LaffAXPYTest(unittest.TestCase):

    def setUp(self):
        real_length = np.random.randint(1, 20)
        self.x = np.random.uniform(0, 10, real_length)
        self.x = np.reshape(self.x, [1, real_length])
        self.y = np.random.uniform(0, 10, real_length)
        self.y = np.reshape(self.y, [1, real_length])
        z_diff = 0
        while z_diff == 0 or real_length + z_diff < 0:
            z_diff = np.random.randint(-5, 6)
        self.z = np.random.uniform(0, 10, real_length + z_diff)
        self.z = np.reshape(self.z, [1, real_length + z_diff])
        self.alpha = float(np.random.randint(-5, 6))

    def test_column_column_axpy(self):
        gold = self.alpha * self.x + self.y
        result = axpy(self.alpha, self.x, self.y)
        np.testing.assert_allclose(result, gold)

    def test_column_row_axpy(self):
        gold = self.alpha * self.x + self.y
        result = axpy(self.alpha, self.x, self.y.T)
        np.testing.assert_allclose(result, gold.T)

    def test_row_column_axpy(self):
        gold = self.alpha * self.x + self.y
        result = axpy(self.alpha, self.x.T, self.y)
        np.testing.assert_allclose(result, gold)

    def test_row_row_axpy(self):
        gold = self.alpha * self.x.T + self.y.T
        result = axpy(self.alpha, self.x.T, self.y.T)
        np.testing.assert_allclose(result, gold)

    def test_bad_alpha(self):
        self.assertRaises(
            Exception, axpy, np.array([3, 4]), self.x, self.y
        )

    def test_column_column_axpy_wrong_size(self):
        self.assertRaises(
            Exception, axpy, self.alpha, self.x, self.z
        )

    def test_column_row_axpy_wrong_size(self):
        self.assertRaises(
            Exception, axpy, self.alpha, self.x, self.z.T
        )

    def test_row_column_axpy_wrong_size(self):
        self.assertRaises(
            Exception, axpy, self.alpha, self.x.T, self.z
        )

    def test_row_row_axpy_wrong_size(self):
        self.assertRaises(
            Exception, axpy, self.alpha, self.x.T, self.z.T
        )


if __name__ == "__main__":
    unittest.main()
