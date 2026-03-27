import numpy as np
from scipy.linalg import solve_discrete_are



def dare_driver():
    A = np.array([1.0])
    B = np.array([1.0])
    Q = np.array([1.0])

    nx = A.shape[0]
    nu = B.shape[0]

    R = np.eye(nu)

    X = solve_discrete_are(A, B, Q, R)

    u = np.linalg.inv(R + B.T @ X @ B) @ (B.T @ X @ A)      

    print("Решение дискретного уравнения Риккати X:")
    print(X)
    print("\nКоэффициенты регулятора u:")
    print(u)

if __name__ == "__main__":
    dare_driver()