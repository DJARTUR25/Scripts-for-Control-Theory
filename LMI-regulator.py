import numpy as np
import cvxpy as cp

def stabilizing_control_design():
    # матрицы системы
    A = np.array([[0.5, 0, 0],
                  [0, -2, 10],
                  [0, 1, -2]])
    B = np.array([[1, 0],
                  [-2, 2],
                  [0, 1]])
    print("B =\n", B)

    # размерности
    nx = A.shape[0]
    nu = B.shape[1]

    # переменные LMI
    Y = cp.Variable((nx, nx), symmetric=True)
    Z = cp.Variable((nu, nx))

    # ограничения
    constraints = []
    lhs = Y @ A.T + A @ Y + Z.T @ B.T + B @ Z
    constraints.append(lhs <= -0.01 * np.eye(nx))
    constraints.append(Y >> 0)

    # цель (тривиальная, ищем только допустимое решение)
    objective = cp.Minimize(0)

    # решение
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    Y_val = Y.value
    Z_val = Z.value
    print("Y =\n", Y_val)

    G = Z_val @ np.linalg.inv(Y_val)
    print("G =\n", G)

    eigvals = np.linalg.eigvals(A + B @ G)
    print("Собственные числа A + B*G:\n", eigvals)

if __name__ == "__main__":
    stabilizing_control_design()