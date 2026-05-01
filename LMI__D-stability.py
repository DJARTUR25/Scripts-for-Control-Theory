import numpy as np
import cvxpy as cp
import warnings

def d_stabilizing_control_design():
    # Матрицы системы
    A = np.array([[-1.,  2.,  0.],
                  [ 0.,  0.5, 1.],
                  [ 1.,  0., -2.]])
    B = np.array([[0],
                  [1],
                  [1]])

    L1 = np.array([[-2,  1],
                   [ 1, -2]])
    M1 = np.array([[0, 1],
                   [0, 0]])

    L2 = 2
    M2 = 1 

    nx = A.shape[0]
    nu = B.shape[1]

    Y = cp.Variable((nx, nx), symmetric=True)
    Z = cp.Variable((nu, nx))

    A_cl = A @ Y + B @ Z

    L11, L12 = L1[0,0], L1[0,1]
    L21, L22 = L1[1,0], L1[1,1]
    top_L1 = cp.hstack([L11 * Y, L12 * Y])
    bottom_L1 = cp.hstack([L21 * Y, L22 * Y])
    kron_L1_Y = cp.vstack([top_L1, bottom_L1])

    M11, M12 = M1[0,0], M1[0,1]
    M21, M22 = M1[1,0], M1[1,1]
    top_M1 = cp.hstack([M11 * A_cl, M12 * A_cl])
    bottom_M1 = cp.hstack([M21 * A_cl, M22 * A_cl])
    kron_M1_Acl = cp.vstack([top_M1, bottom_M1])

    M1T = M1.T
    M1T11, M1T12 = M1T[0,0], M1T[0,1]
    M1T21, M1T22 = M1T[1,0], M1T[1,1]
    Acl_T = A_cl.T
    top_M1T = cp.hstack([M1T11 * Acl_T, M1T12 * Acl_T])
    bottom_M1T = cp.hstack([M1T21 * Acl_T, M1T22 * Acl_T])
    kron_M1T_AclT = cp.vstack([top_M1T, bottom_M1T])

    term1 = kron_L1_Y + kron_M1_Acl + kron_M1T_AclT
    eps = 1e-6
    constr1 = term1 <= -eps * np.eye(2 * nx) 

    term2 = L2 * Y + M2 * A_cl + M2 * A_cl.T
    constr2 = term2 <= -eps * np.eye(nx)

    constraints = [constr1, constr2, Y >> 0]

    objective = cp.Minimize(0)
    problem = cp.Problem(objective, constraints)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem.solve(solver=cp.SCS, verbose=False, max_iters=5000)

    if Y.value is None:
        print("LMI неразрешима")
        return

    Y_val = Y.value
    Z_val = Z.value
    G = Z_val @ np.linalg.inv(Y_val)

    print("Матрица обратной связи G =")
    print(G)
    print("\nСобственные числа замкнутой системы A + B*G:")
    eigvals = np.linalg.eigvals(A + B @ G)
    print(eigvals)

    # Проверка принадлежности области D
    print("\nПроверка принадлежности области D:")
    for i, lam in enumerate(eigvals):
        x = np.real(lam)
        y = np.imag(lam)
        in_circle = (x+1)**2 + y**2 < 4
        in_halfplane = x < -1
        print(f"λ{i+1} = {lam:.5f}: в круге? {in_circle}, в полуплоскости? {in_halfplane}")

if __name__ == "__main__":
    d_stabilizing_control_design()