import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import sympy as sp
import matplotlib.pyplot as plt

# Решение непрерывного уравнения Риккати

def solve_continuous(A, B, Q, R, x0, TIME=10.0, n_points=1001):
# параметры: A, B, Q, R - матрицы системы и функционала; x0 - начальное состояние; TIME - время моделирования; n_points - число точек для моделирования
    
    X = solve_continuous_are(A, B, Q, R) # решение непрерывного уравнения Риккати через пакет scipy.linalg
    # решается непрерывное алгебраическое уравнение Риккати: Q + A^T*X + X*A - X*B*R^-1*B^T*X = 0; X - искомая матрица решения Риккати

    if R.shape[0] == 1:
        U = (1.0 / R[0, 0]) * B.T @ X       # вычисление коэффициентов управления для случая, если R - число
    else:
        U = np.linalg.inv(R) @ B.T @ X      # вычисление коэффициентов управления для случая, если R - матрица

    A_cl = A - (B @ U)   # матрица замкнутой системы

    t_eval = np.linspace(0, TIME, n_points) # создание временного вектора для моделирования оптимального управления и траекторий

    def system(t, x):       # функция для решения системы методом solve_ivp; возвращает dx/dt = A_cl @ x
        return A_cl @ x

    sol = solve_ivp(system, [0, TIME], x0.flatten(), t_eval=t_eval, method='RK45',
                    rtol=1e-6, atol=1e-6)           # решение системы ЧМ Рунге-Кутта 4-5 порядка
    x_t = sol.y.T       # траектория состояний системы при оптимальном управлении
    u_t = np.array([-U @ x for x in x_t]) # траектория оптимального управления; u = -U * x; U - коэффициенты управления, найденные ранее

    # вычисление функции функционала F = (x^T*Q*x + u^T*R*u)
    if Q.shape[0] == 1:
        integrand = Q[0, 0] * np.sum(x_t**2, axis=1) + R[0, 0] * np.sum(u_t**2, axis=1) # случай, если Q, R - числа
    else:
        integrand = np.sum(x_t @ Q @ x_t.T, axis=1) + np.sum(u_t @ R @ u_t.T, axis=1)   # если Q, R - матрицы

    J = np.trapezoid(integrand, t_eval)     # численное интегрирование для получения значения функционала 

    return X, U, J, x_t, u_t, t_eval   


def plot_results(t, x, u):      # функция построения графиков 
    n = x.shape[1]
    m = u.shape[1] if u.ndim > 1 else 1 # число входов управления; 1, если u - вектор; иначе - число столбцов в матрице u

    fig, axes = plt.subplots(2, 1, figsize=(10, 6)) 

    # построение траекторий состояний
    for i in range(n):
        axes[0].plot(t, x[:, i], linewidth=2, label=f'x{i+1}(t)')
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('x(t)')
    axes[0].set_title('States')

    # график оптимального управления замкнутой системы
    if m == 1:
        axes[1].plot(t, u, 'r-', linewidth=2)
    else:
        for i in range(m):
            axes[1].plot(t, u[:, i], linewidth=2, label=f'u{i+1}(t)')
        axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('u(t)')
    axes[1].set_title('Control input')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Ввод матриц

    A = np.array([[0.0, 1.0],
                  [0.0, 0.0]])
    
    B = np.array([[0.0],
                  [1.0]])
    
    Q = np.array([[1.0, 0.0],
                  [0.0, 2.0]])
    
    R = np.array([[1.0]])

    x0 = np.array([[0.4],
                   [0.4]])  # начальное приближение

    X, U, J, x_t, u_t, t_eval = solve_continuous(A, B, Q, R, x0) # решение непрерывного алгебраического уравнения Рикатти
    
    # результаты вычислений: решение уравнения Рикатти Х, вектор оптимального управления U, значение минимума функционала J
    print("X =")
    print(X)
    print("\nU =")
    print(U)
    print(f"\nJ = {J:.6f}")

    # построение графиков
    plot_results(t_eval, x_t, u_t)