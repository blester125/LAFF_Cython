import unittest
import numpy as np
from .dot import dot


class LaffDotTest(unittest.TestCase):

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
        self.gold = np.dot(np.squeeze(self.x), np.squeeze(self.y))

    def test_column_column_copy(self):
        np.testing.assert_allclose(dot(self.x, self.y), self.gold)

    def test_column_row_copy(self):
        np.testing.assert_allclose(dot(self.x, self.y.T), self.gold)

    def test_row_column_copy(self):
        np.testing.assert_allclose(dot(self.x.T, self.y), self.gold)

    def test_row_row_copy(self):
        np.testing.assert_allclose(dot(self.x.T, self.y.T), self.gold)

    def test_column_column_worong_size(self):
        self.assertRaises(ValueError, dot, self.x, self.z)

    def test_column_row_worong_size(self):
        self.assertRaises(ValueError, dot, self.x, self.z.T)

    def test_row_column_worong_size(self):
        self.assertRaises(ValueError, dot, self.x.T, self.z)

    def test_row_row_worong_size(self):
        self.assertRaises(ValueError, dot, self.x.T, self.z.T)


if __name__ == "__main__":
    unittest.main()
