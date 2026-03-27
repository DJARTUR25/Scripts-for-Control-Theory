import numpy as np
import matplotlib.pyplot as plt

def R_LQM():     
                    
    x_vals = [0.6931, 1.0986, 1.3863, 1.6094, 1.7918]   # факторы
    y_vals = [0.4055, 1.0986, 1.5041, 1.9459, 2.1401]   # отклики (измерения, наблюдения)

    xlst = np.array(x_vals)             # собираем данные в массив
    ylst = np.array(y_vals)
    n = len(ylst)                       # запоминаем число измерений

    RM = np.zeros((n, 2))               # заполнена нулями; n строк, 2 столбца, если МНК-прямая; для МНК-параболы 3 столбца
    for k in range(n):
        RM[k, :] = [xlst[k], 1]         # первый столбец заполняется факторами; для МНК-параболы до xlist[k] вставить xlist**2[k]

    H0 = RM[:2].transpose() @ RM[:2]                                # начальная матрица; считаем как C0^T * C0
    x0 = np.linalg.inv(H0) @ RM[:2].transpose() @ ylst[:2]          # начальные параметры; если МНК-прямая, то берем 2 штуки; для параболы - 3 и т.д.
    alst = np.zeros((2, n))             # alist сохраняет историю оценок параметров МНК-кривой; размерность 2 на n; используются для построения графиков сходимости
    alst[:, 2] = x0
    print(f"a = {x0[0]:.4f}, b = {x0[1]:.4f}")  # вывод начальных параметров
    print(f"H{k} = {H0}")                       # вывод начальной матрицы H
    print("-" * 50)

    for k in range(2, n):
        H0 = H0 + np.outer(RM[k, :].transpose(), RM[k, :])          # H_k = H_k-1 + (C_k)^T * C_k
        # print (H0) # для вывода итераций
        
        x0 = x0 + np.linalg.inv(H0) @ RM[k, :].transpose() * (ylst[k] - RM[k, :].transpose() @ x0)  # вычисление новых параметров
        alst[:, k] = x0     # сохранение параметров для графиков сходимости
        print(f"k = {k}")
        print(f"a = {x0[0]:.4f}, b = {x0[1]:.4f}")  # вывод текущей оценки параметров
        print(f"H{k} = {H0}")                       # вывод текущей матрицы H
        print("-" * 50)



    # Графики сходимости параметров
    idx = np.arange(2, n)
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

    # для МНК-параболы, график сходимости свободного коэффициента; 
    # для больших размерностей МНК-кривой изменить первую тройку в subplot до нужной размерности

    # plt.subplot(3, 1, 3)
    # plt.plot(idx, alst[1, idx], 'k', linewidth=2)
    # plt.xlabel('k')
    # plt.ylabel('с_k')
    # plt.grid()

    plt.tight_layout()

    # вывод в консоль итоговых параметров МНК-кривой
    final_params = alst[:, -1]
    print("Итоговые параметры (a, b, c):", final_params)

    # Аппроксимация, построение МНК-кривой и пометка исходных точек звездами
    zlst = RM @ final_params

    plt.figure()
    plt.plot(xlst, zlst, 'k', linewidth=2, label='Аппроксимация')
    plt.plot(xlst, ylst, '*r', label='Данные')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.show()

# запуск скрипта
if __name__ == "__main__":
    R_LQM()