import numpy as np
import cvxpy as cp

A = np.array([[-16, -19, 6],
              [17, 20, -7],
              [15, 18, -6]])
B = np.array([[-1], [1], [1]])
C = np.array([[6, 5, 1]])

def lmi_feasible(gamma):
    n = A.shape[0]
    X = cp.Variable((n, n), symmetric=True)
    top_left = A.T @ X + X @ A
    top_mid = X @ B
    top_right = C.T
    mid_left = B.T @ X
    mid_mid = -gamma * np.eye(B.shape[1])
    mid_right = np.zeros((B.shape[1], C.shape[0]))
    bottom_left = C
    bottom_mid = np.zeros((C.shape[0], B.shape[1]))
    bottom_right = -gamma * np.eye(C.shape[0])

    top = cp.hstack([top_left, top_mid, top_right])
    mid = cp.hstack([mid_left, mid_mid, mid_right])
    bottom = cp.hstack([bottom_left, bottom_mid, bottom_right])
    LMI = cp.vstack([top, mid, bottom])

    constraints = [LMI << -1e-7 * np.eye(LMI.shape[0]), X >> 1e-7 * np.eye(n)]
    prob = cp.Problem(cp.Minimize(0), constraints)
    try:
        prob.solve(solver=cp.CVXOPT, verbose=False, abstol=1e-9, reltol=1e-9)
    except:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=20000, eps=1e-9)
    return prob.status == 'optimal'

a = 1e-3
b = 100.0
eps = 1e-6

print(' |-----|-------------|-------------|-------------|')
print(' | шаг |      a      |      b      |      c      |')

for it in range(1, 101):
    c = (a + b) / 2
    print(f' | {it:3d} | {a:.5e} | {b:.5e} | {c:.5e} |')
    if lmi_feasible(c):
        b = c
    else:
        a = c
    if (b - a) < eps:
        break

gamma_star = c
delta_max = 1.0 / gamma_star
print(f'\nγ* = {gamma_star:.8f}')
print(f'δ_max = {delta_max:.8f}')