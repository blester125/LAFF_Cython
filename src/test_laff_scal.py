import unittest
import numpy as np
from .scal import scal


class LaffScalTest(unittest.TestCase):

    def setUp(self):
        real_length = np.random.randint(1, 20)
        self.x = np.random.uniform(0, 10, real_length)
        self.x = np.reshape(self.x, [1, real_length])
        self.alpha = float(np.random.randint(-5, 6))

    def test_column_scale(self):
        gold = self.alpha * self.x
        np.testing.assert_allclose(scal(self.alpha, self.x), gold)

    def test_row_scale(self):
        gold = self.alpha * self.x.T
        np.testing.assert_allclose(scal(self.alpha, self.x.T), gold)

    def test_bad_alpha(self):
        self.assertRaises(Exception, scal, np.array([3, 4]), self.x)

    def test_bad_x(self):
        bad_x = np.random.randn(2, 2)
        self.assertRaises(Exception, scal, self.alpha, bad_x)


if __name__ == "__main__":
    unittest.main()
