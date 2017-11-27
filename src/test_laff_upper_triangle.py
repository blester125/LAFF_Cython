import unittest
import numpy as np
from .triangle import UpperTriangle, StrictlyUpper, UnitUpper


class UpperTriangleTest(unittest.TestCase):

    def setUp(self):
        self.rows = np.random.randint(5, 20)
        self.cols = self.rows
        self.A = np.random.randn(self.rows, self.cols)
        self.gold_matrix = np.copy(self.A)

    def test_triangle_is_made(self):
        triangle = UpperTriangle(self.A)
        for i in range(triangle.shape[0]):
            for j in range(triangle.shape[0]):
                if i <= j:
                    assert triangle[i, j] == self.gold_matrix[i, j]
                else:
                    assert triangle[i, j] == 0

    def test_shape_preserved(self):
        triangle = UpperTriangle(self.A)
        assert triangle.shape == self.gold_matrix.shape

    def test_error_on_non_square(self):
        diff = 0
        while diff == 0:
            diff = np.random.randint(-4, 5)
        input_matrix = np.random.rand(self.rows, self.cols+diff)
        self.assertRaises(
            Exception, UpperTriangle, input_matrix
        )


class StrictlyUpperTest(unittest.TestCase):

    def setUp(self):
        self.rows = np.random.randint(5, 20)
        self.cols = self.rows
        self.A = np.random.randn(self.rows, self.cols)
        self.gold_matrix = np.copy(self.A)

    def test_strict_triangle_is_made(self):
        triangle = StrictlyUpper(self.A)
        for i in range(triangle.shape[0]):
            for j in range(triangle.shape[0]):
                if i < j:
                    assert triangle[i, j] == self.gold_matrix[i, j]
                else:
                    assert triangle[i, j] == 0

    def test_shape_preserved(self):
        triangle = StrictlyUpper(self.A)
        assert triangle.shape == self.gold_matrix.shape

    def test_error_on_non_square(self):
        diff = 0
        while diff == 0:
            diff = np.random.randint(-4, 5)
        input_matrix = np.random.rand(self.rows, self.cols+diff)
        self.assertRaises(
            Exception, StrictlyUpper, input_matrix
        )


class UnitUpperTest(unittest.TestCase):

    def setUp(self):
        self.rows = np.random.randint(5, 20)
        self.cols = self.rows
        self.A = np.random.randn(self.rows, self.cols)
        self.gold_matrix = np.copy(self.A)

    def test_unit_triangle_is_made(self):
        triangle = UnitUpper(self.A)
        for i in range(triangle.shape[0]):
            for j in range(triangle.shape[0]):
                if i < j:
                    assert triangle[i, j] == self.gold_matrix[i, j]
                elif i == j:
                    assert triangle[i, j] == 1
                else:
                    assert triangle[i, j] == 0

    def test_shape_preserved(self):
        triangle = UnitUpper(self.A)
        assert triangle.shape == self.gold_matrix.shape

    def test_error_on_non_square(self):
        diff = 0
        while diff == 0:
            diff = np.random.randint(-4, 5)
        input_matrix = np.random.rand(self.rows, self.cols+diff)
        self.assertRaises(
            Exception, UnitUpper, input_matrix
        )

if __name__ == "__main__":
    unittest.main()
