import numpy as np
import cvxpy as cp
import warnings

def stabilizing_control_design_with_factor():
    # Матрицы системы
    A = np.array([[ 1, -1],
                  [ 1,  1]])
    B = np.array([[0],
                  [1]])

    alpha = 1.0

    nx = A.shape[0]
    nu = B.shape[1]

    # Переменные LMI
    Y = cp.Variable((nx, nx), symmetric=True)
    Z = cp.Variable((nu, nx))

    # LMI: Y*A' + A*Y + Z'*B' + B*Z + 2*alpha*Y < -0.01*I
    eps = 0.01
    lhs = Y @ A.T + A @ Y + Z.T @ B.T + B @ Z + 2 * alpha * Y
    constraints = [
        lhs <= -eps * np.eye(nx),
        Y >> 0
    ]

    objective = cp.Minimize(0)
    problem = cp.Problem(objective, constraints)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem.solve(solver=cp.SCS, verbose=False)

    if Y.value is None:
        print("LMI неразрешима")
        return

    Y_val = Y.value
    Z_val = Z.value
    G = Z_val @ np.linalg.inv(Y_val)

    print("Y =\n", Y_val)
    print("G =\n", G)
    print("Собственные числа A + B*G:\n", np.linalg.eigvals(A + B @ G))

if __name__ == "__main__":
    stabilizing_control_design_with_factor()