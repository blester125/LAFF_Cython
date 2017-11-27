import unittest
import numpy as np
from .m_v_mult import m_v_mult


class MVMultTest(unittest.TestCase):

    def setUp(self):
        self.rows = np.random.randint(5, 20)
        self.cols = np.random.randint(5, 20)
        self.A = np.random.randn(self.rows, self.cols)
        self.x = np.random.randn(self.cols, 1)

    def test_mult(self):
        res = m_v_mult(self.A, self.x)
        np.testing.assert_allclose(res, np.dot(self.A, self.x))

    def test_shape(self):
        res = m_v_mult(self.A, self.x)
        assert res.shape == (self.A.shape[0], self.x.shape[1])

    def test_error_on_bad_vector_shape(self):
        diff = 0
        while diff == 0:
            diff = np.random.randint(-4, 5)
        input_vector = np.random.randn(self.cols + diff, 1)
        self.assertRaises(
            Exception, m_v_mult, self.A, input_vector
        )

    def test_error_on_bad_matrix_shape(self):
        diff = 0
        while diff == 0:
            diff = np.random.randint(-4, 5)
        input_matrix = np.random.randn(self.rows, self.cols + diff)
        self.assertRaises(
            Exception, m_v_mult, input_matrix, self.x
        )


if __name__ == "__main__":
    unittest.main()
