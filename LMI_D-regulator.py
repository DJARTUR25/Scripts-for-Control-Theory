import numpy as np
import cvxpy as cp

def d_stabilizing_task8a():
    A = np.array([[-2.5, 0],
                  [0,   1]])
    B = np.array([[1],
                  [1]])
    L = np.array([[-1, 2],
                  [2, -1]])
    M = np.array([[0, 0.75],
                  [0.25, 0]])

    nx = 2
    X = cp.Variable((nx, nx), symmetric=True)
    Z = cp.Variable((1, nx))
    Acl = A @ X + B @ Z

    # Кронекеровы произведения вручную
    # L ⊗ X
    L11, L12 = L[0,0], L[0,1]
    L21, L22 = L[1,0], L[1,1]
    kron_L = cp.vstack([cp.hstack([L11*X, L12*X]),
                        cp.hstack([L21*X, L22*X])])
    # M ⊗ Acl
    M11, M12 = M[0,0], M[0,1]
    M21, M22 = M[1,0], M[1,1]
    kron_M = cp.vstack([cp.hstack([M11*Acl, M12*Acl]),
                        cp.hstack([M21*Acl, M22*Acl])])
    # M^T ⊗ Acl^T
    MT = M.T
    MT11, MT12 = MT[0,0], MT[0,1]
    MT21, MT22 = MT[1,0], MT[1,1]
    AclT = Acl.T
    kron_MT = cp.vstack([cp.hstack([MT11*AclT, MT12*AclT]),
                         cp.hstack([MT21*AclT, MT22*AclT])])

    F = kron_L + kron_M + kron_MT
    constr = F <= -1e-6 * np.eye(4)
    prob = cp.Problem(cp.Minimize(0), [constr, X >> 0])
    prob.solve(solver=cp.CVXOPT, verbose=False, abstol=1e-8, reltol=1e-8)

    if X.value is not None:
        G = Z.value @ np.linalg.inv(X.value)
        print("G =", G)
        print("eig(A+BG) =", np.linalg.eigvals(A + B @ G))
    else:
        print("No solution")

if __name__ == "__main__":
    d_stabilizing_task8a()