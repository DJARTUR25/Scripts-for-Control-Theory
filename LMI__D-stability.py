import numpy as np
import cvxpy as cp
import warnings

def continuous_system_d_stability_test():
    # Матрица системы
    A = np.array([[-4.2386, -0.2026,  0.7193],
                  [ 2.6649, -2.8342,  0.0175],
                  [ 0.0344,  0.0005, -3.1772]])

    # Параметры LMI-области
    L1 = 4
    M1 = 1
    L2 = np.array([[-9,  3],
                   [ 3, -1]])
    M2 = np.array([[0, 1],
                   [0, 0]])

    nx = A.shape[0]

    # Переменная LMI
    X = cp.Variable((nx, nx), symmetric=True)

    term1_1 = np.kron(L1, X)
    term1_2 = np.kron(M1, A @ X)
    term1_3 = np.kron(M1, X @ A.T)

    term2_1 = np.kron(L2, X)
    term2_2 = np.kron(M2, A @ X)
    term2_3 = np.kron(M2.T, X @ A.T)

    eps = 0.001
    constraints = [
        term1_1 + term1_2 + term1_3 <= -eps * np.eye(2 * nx),
        term2_1 + term2_2 + term2_3 <= np.zeros((2 * nx, 2 * nx)),
        X >> 0
    ]

    objective = cp.Minimize(0)
    problem = cp.Problem(objective, constraints)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem.solve(solver=cp.SCS, verbose=False)

    print("X =\n", X.value)
    print("Собственные числа A:\n", np.linalg.eigvals(A))

if __name__ == "__main__":
    continuous_system_d_stability_test()