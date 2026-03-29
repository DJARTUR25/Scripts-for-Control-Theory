import numpy as np
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt

# Решение дискретного уравнения Риккати

def solve_discrete(A, B, Q, R, x0, N=30):
# параметры: A, B, Q, R - матрицы системы и функционала; x0 - начальное состояние; N - количество шагов моделирования

    X = solve_discrete_are(A, B, Q, R)  # решение дискретного уравнения Риккати через пакет scipy.linalg
    # решается дискретное алгебраическое уравнение Риккати: Q + A^T*X*A - A^T*X*B*[(B^T*X*B + R)^(-1)]*B^T*X*A - X = 0; X - искомая матрица решения Риккати

    if R.shape[0] == 1:
        U = (1.0 / (R[0, 0] + B.T @ X @ B)[0, 0]) * B.T @ X @ A       # вычисление коэффициентов управления для случая, если R - число
        # U = 1 / (B^T * X * B + R) * B^T * X * A
    else:
        U = np.linalg.inv(R + B.T @ X @ B) @ (B.T @ X @ A)      # вычисление коэффициентов управления для случая, если R - матрица
        # U = (B^T * X * B + R)^-1 * B^T * X * A

    A_cl = A - (B @ U)   # матрица замкнутой системы
    print (A_cl)


    n = A.shape[0]  # размерность состояний
    m = B.shape[1]  # размерность входов управления

    x_seq = np.zeros((n, N + 1))    # матрица траекторий состояний
    u_seq = np.zeros((m, N))        # матрица траекторий управления

    x_seq[:, 0] = x0.flatten()  # начальное состояние
    J = 0.0         # переменная для накопления значения функционала

    # вычисление оптимального управления на каждом шаге
    for k in range(N):
        u_k = -U @ x_seq[:, k]  # U_k = - U * x_k
        u_seq[:, k] = u_k   # следующее значение управления
        x_seq[:, k + 1] = A @ x_seq[:, k] + B @ u_k # x_k+1 = A*x_k + B*u_k
        J += x_seq[:, k] @ Q @ x_seq[:, k] + u_k @ R @ u_k  # J = J + x_k^T * Q * x_k + u_k^T * R * u_k - накопление функционала

    return X, U, J, x_seq, u_seq

# построение графиков траекторий и оптимального управления
def plot_results(k_vals, x_seq, u_seq):
    n = x_seq.shape[0]
    m = u_seq.shape[0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # Графики траекторий состояний
    for i in range(n):
        axes[0].plot(k_vals, x_seq[i, :], 'o-', linewidth=2, markersize=4, label=f'x{i+1}[k]')
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('x[k]')
    axes[0].set_title('траектории состояний')

    # График оптимального управления
    if m == 1:
        axes[1].plot(k_vals[:-1], u_seq[0, :], 'ro-', linewidth=2, markersize=4)
    else:
        for i in range(m):
            axes[1].plot(k_vals[:-1], u_seq[i, :], 'o-', linewidth=2, markersize=4, label=f'u{i+1}[k]')
        axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('u[k]')
    axes[1].set_title('Оптимальное управление')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # ввод матриц


    A = np.array([[0.0, 1.0],
                  [-1.0, 1.0]])
    
    B = np.array([[0.0],
                  [1.0]])
    
    Q = np.array([[1.0, 0.0],
                  [0.0, 0.0]])
    
    R = np.array([[1.0]])

    x0 = np.array([[0.1],
                   [0.1]])  # начальное состояние

    N = 30 

    X, U, J, x_seq, u_seq = solve_discrete(A, B, Q, R, x0, N)

    # результаты вычислений: решение уравнения Рикатти X, матрица коэффициентов управления K, значение минимума функционала J
    print("X =")
    print(X)
    print("\nU =")
    print(U)
    print(f"\nJ = {J:.6f}")
    print(f"u* = {-U[0,0]:.4f} * x1,k + {-U[0,1]:.4f} * x2,k")

    # Построение графиков
    plot_results(np.arange(N + 1), x_seq, u_seq)