import unittest
import numpy as np

#kode sumber
def crout_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for j in range(n):
        U[j, j] = 1
        for i in range(j, n):
            sum_ = sum(L[i, k] * U[k, j] for k in range(i))
            L[i, j] = A[i, j] - sum_
        for i in range(j + 1, n):
            sum_ = sum(L[j, k] * U[k, i] for k in range(j))
            U[j, i] = (A[j, i] - sum_) / L[j, j]

    return L, U

def solve_crout(A, b):
    L, U = crout_decomposition(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(U, y)
    return x

#kode testing
class TestCrout(unittest.TestCase):
    def test_solve_crout(self):
        A = np.array([[2, 1, -1], [4, 1, 0], [-2, 2, 1]])
        b = np.array([1, 4, 5])
        expected_solution = np.array([1, 2, 3])
        solution = solve_crout(A, b)
        np.testing.assert_almost_equal(solution, expected_solution)

if __name__ == "__main__":
    unittest.main()
