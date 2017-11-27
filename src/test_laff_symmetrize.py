import unittest
import numpy as np
from .symmetrize import symmetrize_from_upper, symmetrize_from_lower


class SymmetrizeFromUpperTest(unittest.TestCase):

    def setUp(self):
        self.size = np.random.randint(5, 20)
        self.A = np.random.randint(5, 20, size=[self.size, self.size]).astype(np.float64)
        self.gold_matrix = np.copy(self.A)

    def test_symmetric(self):
        a = symmetrize_from_upper(self.A)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                assert a[i, j] == a[j, i]

    def test_right_values(self):
        a = symmetrize_from_upper(self.A)
        for i in range(self.gold_matrix.shape[0]):
            for j in range(self.gold_matrix.shape[1]):
                if i > j:
                    assert a[i, j] == self.gold_matrix[j, i]

    def test_right_shape(self):
        a = symmetrize_from_upper(self.A)
        assert a.shape == self.gold_matrix.shape


    def test_error_on_non_square(self):
        diff = 0
        while diff == 0:
            diff = np.random.randint(-4, 5)
        input_matrix = np.random.randn(self.size, self.size + diff)
        self.assertRaises(
            Exception, symmetrize_from_upper, input_matrix
        )


class SymmetrizeFromLowerTest(unittest.TestCase):

    def setUp(self):
        self.size = np.random.randint(5, 20)
        self.A = np.random.randint(5, 20, size=[self.size, self.size]).astype(np.float64)
        self.gold_matrix = np.copy(self.A)

    def test_symmetric(self):
        a = symmetrize_from_lower(self.A)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                assert a[i, j] == a[j, i]

    def test_right_values(self):
        a = symmetrize_from_lower(self.A)
        for i in range(self.gold_matrix.shape[0]):
            for j in range(self.gold_matrix.shape[1]):
                if i < j:
                    assert a[i, j] == self.gold_matrix[j, i]

    def test_right_shape(self):
        a = symmetrize_from_lower(self.A)
        assert a.shape == self.gold_matrix.shape


    def test_error_on_non_square(self):
        diff = 0
        while diff == 0:
            diff = np.random.randint(-4, 5)
        input_matrix = np.random.randn(self.size, self.size + diff)
        self.assertRaises(
            Exception, symmetrize_from_lower, input_matrix
        )

if __name__ == "__main__":
    unittest.main()
