import numpy as np
import cvxpy as cp
import warnings

def d_stabilizing_control_design():
    # Матрицы системы
    A = np.array([[-4.2386, -0.2026,  0.7193],
                  [ 2.6649, -2.8342,  0.0175],
                  [ 0.0344,  0.0005, -3.1772]])
    B = np.array([[0],
                  [0],
                  [1]])

    # Параметры LMI-области
    L1 = 4
    M1 = 1
    L2 = np.array([[-9,  3],
                   [ 3, -1]])
    M2 = np.array([[0, 1],
                   [0, 0]])

    nx = A.shape[0]
    nu = B.shape[1]

    X = cp.Variable((nx, nx), symmetric=True)
    Z = cp.Variable((nu, nx))

    A_cl = A @ X + B @ Z

    term1_1 = np.kron(L1, X)
    term1_2 = np.kron(M1, A_cl)
    term1_3 = np.kron(M1, A_cl.T)   # M1' = M1
    eps = 0.001
    constr1 = term1_1 + term1_2 + term1_3 <= -eps * np.eye(2 * nx)

    term2_1 = np.kron(L2, X)
    term2_2 = np.kron(M2, A_cl)
    term2_3 = np.kron(M2.T, A_cl.T)
    constr2 = term2_1 + term2_2 + term2_3 <= np.zeros((2 * nx, 2 * nx))

    constraints = [constr1, constr2, X >> 0]

    objective = cp.Minimize(0)
    problem = cp.Problem(objective, constraints)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem.solve(solver=cp.SCS, verbose=False)

    if X.value is None:
        print("LMI неразрешима")
        return

    X_val = X.value
    Z_val = Z.value
    G = Z_val @ np.linalg.inv(X_val)

    print("X =\n", X_val)
    print("Z =\n", Z_val)
    print("G =\n", G)
    print("Собственные числа A + B*G:\n", np.linalg.eigvals(A + B @ G))

if __name__ == "__main__":
    d_stabilizing_control_design()