import unittest
import numpy as np
from .diagonal import diagonal


class DiagonalTest(unittest.TestCase):

    def setUp(self):
        self.matrix_size = np.random.randint(5, 20)
        self.A = np.random.randn(self.matrix_size, self.matrix_size)
        self.x = np.random.randn(1, self.matrix_size)

    def test_diagonal_with_row(self):
        a = diagonal(self.A, self.x)
        np.testing.assert_allclose(np.sum(a), np.sum(self.x))
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if i == j:
                    assert a[i, j] == self.x.squeeze()[i]
                else:
                    assert a[i, j] == 0

    def test_diagonal_with_col(self):
        x = np.reshape(self.x, [-1, 1])
        a = diagonal(self.A, x)
        np.testing.assert_allclose(np.sum(a), np.sum(self.x))
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if i == j:
                    assert a[i, j] == self.x.squeeze()[i]
                else:
                    assert a[i, j] == 0

    def test_diagonal_size_error(self):
        x = np.random.randn(1, self.matrix_size + 1)
        self.assertRaises(
            Exception, diagonal, self.A, x
        )

    def test_diagonal_not_square(self):
        rows = np.random.randint(5, 20)
        cols = rows
        while cols == rows:
            cols = np.random.randint(5, 20)
        A = np.random.randn(rows, cols)
        self.assertRaises(
            Exception, diagonal, A, self.x
        )

    def test_diagonal_not_vector(self):
        x = np.random.randn(np.random.randint(2, 10), self.matrix_size)
        self.assertRaises(
            Exception, diagonal, self.A, x
        )


if __name__ == "__main__":
    unittest.main()
