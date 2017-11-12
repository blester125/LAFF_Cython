import unittest
import numpy as np
from LAFF.laff_norm2 import laff_norm2


class LaffNorm2Test(unittest.TestCase):

    def setUp(self):
        real_length = np.random.randint(1, 20)
        self.x = np.random.uniform(0, 10, real_length)
        self.x = np.reshape(self.x, [1, real_length])
        self.alpha = np.random.randint(-5, 6)
        self.gold = np.linalg.norm(np.squeeze(self.x))

    def test_column_scale(self):
        np.testing.assert_allclose(laff_norm2(self.x), self.gold)

    def test_row_scale(self):
        np.testing.assert_allclose(laff_norm2(self.x.T), self.gold)

    def test_bad_x(self):
        bad_x = np.random.randn(2, 2)
        self.assertRaises(ValueError, laff_norm2, bad_x)


if __name__ == "__main__":
    unittest.main()
