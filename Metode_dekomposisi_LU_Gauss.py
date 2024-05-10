import unittest
import numpy as np

#kode sumber
def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i, i] = 1

    for k in range(n):
        U[k, k:] = A[k, k:] - L[k, :k] @ U[:k, k:]
        L[(k+1):, k] = (A[(k+1):, k] - L[(k+1):, :] @ U[:, k]) / U[k, k]

    return L, U

def solve_lu_gauss(A, b):
    L, U = lu_decomposition(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(U, y)
    return x

#kode testing
class TestLUGauss(unittest.TestCase):
    def test_solve_lu_gauss(self):
        A = np.array([[2, 1, -1], [4, 1, 0], [-2, 2, 1]])
        b = np.array([1, 4, 5])
        expected_solution = np.array([1, 2, 3])
        solution = solve_lu_gauss(A, b)
        np.testing.assert_almost_equal(solution, expected_solution)

if __name__ == "__main__":
    unittest.main()
