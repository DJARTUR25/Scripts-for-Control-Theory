import numpy as np
import matplotlib.pyplot as plt

def ols_estimation_2():
    # Пользовательские данные
    x_vals = [0.6931, 1.0986, 1.3863, 1.6094, 1.7918]
    y_vals = [0.4055, 1.0986, 1.5041, 1.9459, 3.1401]

    xlst = np.array(x_vals)
    ylst = np.array(y_vals)
    n = len(xlst)

    # Матрица регрессоров [x^2, x, 1]
    CM = np.zeros((n, 2))
    for k in range(n):
        CM[k, :] = [xlst[k], 1]

    # Инициализация на первых трёх точках
    H0 = CM[:2].T @ CM[:2]                 # 3x3
    x0 = np.linalg.inv(H0) @ CM[:2].T @ ylst[:2]   # начальные параметры
    alst = np.zeros((2, n))
    alst[:, 2] = x0                         # после третьей точки (индекс 2)

    # Рекуррентное обновление
    for k in range(2, n):
        H0 = H0 + np.outer(CM[k, :], CM[k, :])
        x0 = x0 + np.linalg.inv(H0) @ CM[k, :] * (ylst[k] - CM[k, :] @ x0)
        alst[:, k] = x0
        print(f"k = {k}: a = {x0[0]:.4f}, b = {x0[1]:.4f}")

    # График сходимости параметров
    idx = np.arange(2, n)   # с третьей точки
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(idx, alst[0, idx], 'k', linewidth=2)
    plt.xlabel('k')
    plt.ylabel('a_k')
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(idx, alst[1, idx], 'k', linewidth=2)
    plt.xlabel('k')
    plt.ylabel('b_k')
    plt.grid()

    plt.tight_layout()

    # Итоговая оценка параметров (последний столбец)
    final_params = alst[:, -1]
    print("Итоговые параметры (a, b, c):", final_params)

    # Аппроксимация
    zlst = CM @ final_params

    plt.figure()
    plt.plot(xlst, zlst, 'k', linewidth=2, label='Аппроксимация')
    plt.plot(xlst, ylst, '*r', label='Данные')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ols_estimation_2()