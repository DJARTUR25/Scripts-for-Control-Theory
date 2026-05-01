import numpy as np
from scipy.linalg import solve_continuous_are

A = np.array([[-1., 0.],
              [2., -2.]])
B = np.array([[0.],
              [0.]])
C = np.array([[0., 1.]])

a = 1e-3
b = 100.0
eps = 1e-10

print(' |-----------------------------------------------|')
print(' |        ВЫЧИСЛЕНИЕ КОМПЛЕКСНОГО РАДИУСА        |')
print(' |             РОБАСТНОЙ УСТОЙЧИВОСТИ            |')
print(' |-----|-------------|-------------|-------------|')
print(' | шаг |      a      |      b      |      c      |')

for it in range(1, 101):
    c = 0.5 * (a + b)
    print(' |-----|-------------|-------------|-------------|')
    print(f' | {it:3d} | {a:.5e} | {b:.5e} | {c:.5e} |')

    Q = C.T @ C
    R = - (c**2) * np.eye(B.shape[1])

    try:
        X = solve_continuous_are(A, B, Q, R)
        b = c
    except np.linalg.LinAlgError:
        a = c

    if abs(b - a) < eps:
        break

gamma_star = c
delta_max = 1.0 / gamma_star

print(' |-----|-------------|-------------|-------------|\n')
print(f'γ* = {gamma_star:.10f}')
print(f'Комплексный радиус робастной устойчивости δ_max = {delta_max:.10f}')