import numpy as np
import matplotlib.pyplot as plt

def R_LQM():     
                    
    x_vals = [1., 1.8, 2.2, 2.6, 3., 3.8, 4.2]   # факторы
    y_vals = [0.7, 0.67, 0.62, 0.51, 0.45, 0.32, 0.28]   # отклики (измерения, наблюдения)

    xlst = np.array(x_vals)             # собираем данные в массив
    xlog = np.log(xlst)                 # массив логарифмов ln(x)
    ylst = np.array(y_vals)
    ylog = np.log(ylst)                 # массив логарифмов ln(y)
    n = len(ylst)                       # запоминаем число измерений

    RM = np.zeros((n, 3))               # n строк, 3 столбца: [1, ln(x), x]
    for k in range(n):
        RM[k, :] = [1, xlog[k], xlst[k]]

    # Инициализация RLS с ковариационной матрицей
    alpha = 1e3                         # большое число
    P = alpha * np.eye(3)               # начальная ковариационная матрица
    theta = np.zeros(3)                 # начальные параметры: [A, b, c] (A = ln a)
    alst = np.zeros((3, n))             # история параметров для графиков сходимости

    # Вывод на шаге k=0 (до обработки данных)
    print(" k = 0 ")
    print(f"a = {np.exp(theta[0]):.4f}, b = {theta[1]:.4f}, c = {theta[2]:.4f}")
    print("-" * 50)

    # Обработка точек последовательно
    for i in range(n):
        phi = RM[i, :]                 # вектор регрессоров
        z = ylog[i]                    # измеренное значение

        # Коэффициент усиления
        K = P @ phi / (1.0 + phi @ P @ phi)   # для λ=1 (обычный МНК)

        # Обновление параметров
        theta = theta + K * (z - phi @ theta)

        # Обновление ковариационной матрицы
        P = P - np.outer(K, phi) @ P

        # Сохраняем параметры для графиков
        alst[:, i] = theta

        # Вывод на нужных шагах (k = количество обработанных точек)
        if i+1 in [2, 5, 7]:           # шаги 2, 5, 7
            print(f"k = {i+1}")
            print(f"a = {np.exp(theta[0]):.4f}, b = {theta[1]:.4f}, c = {theta[2]:.4f}")
            print("-" * 50)

    # Графики сходимости параметров
    idx = np.arange(1, n+1)            # номера шагов от 1 до n
    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(idx, np.exp(alst[0, :]), 'k', linewidth=2)
    plt.xlabel('k')
    plt.ylabel('a_k')
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(idx, alst[1, :], 'k', linewidth=2)
    plt.xlabel('k')
    plt.ylabel('b_k')
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(idx, alst[2, :], 'k', linewidth=2)
    plt.xlabel('k')
    plt.ylabel('c_k')
    plt.grid()

    plt.tight_layout()

    # вывод в консоль итоговых параметров МНК-кривой
    final_params = theta
    print("Итоговые параметры (a, b, c):", np.exp(final_params[0]), final_params[1], final_params[2])

    # Аппроксимация, построение МНК-кривой и пометка исходных точек звездами
    zlst = RM @ final_params
    plt.figure()
    plt.plot(xlst, np.exp(zlst), 'k', linewidth=2, label='Аппроксимация')
    plt.plot(xlst, ylst, '*r', label='Данные')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.show()

# запуск скрипта
if __name__ == "__main__":
    R_LQM()