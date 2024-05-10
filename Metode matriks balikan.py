import unittest
import numpy as np

def inverse_matrix_method(A, B):
    A_inv = np.linalg.inv(A)
    X = np.dot(A_inv, B)
    return X

class TestInverseMatrixMethod(unittest.TestCase):
    def test_solution(self):
        A = np.array([[2, 1],
                      [3, -2]])
        B = np.array([[5],
                      [7]])
        expected_solution = np.array([[1.],
                                      [2.]])
        
        actual_solution = inverse_matrix_method(A, B)
        np.testing.assert_array_almost_equal(actual_solution, expected_solution)

if __name__ == '__main__':
    unittest.main()
