import numpy as np
import matplotlib.pyplot as plt

def LQM():      # метод наименьших квадратов 

    x_vals = [0.6931, 1.0986, 1.3863, 1.6094, 1.7918]   # факторы
    y_vals = [0.4055, 1.0986, 1.5041, 1.9459, 3.1401]   # отклики (измерения, наблюдения)

    xlst = np.array(x_vals)     # собираем данные в массив
    ylst = np.array(y_vals)
    n = len(xlst)               # запоминаем число измерений

    # матрица регрессоров
    RM = np.zeros((n, 2))       # заполнена нулями; n строк, 2 столбца, если МНК-прямая; для МНК-параболы 3 столбца
    for k in range(n):
        RM[k, :] = [xlst[k], 1] # первый столбец заполняется факторами; для МНК-параболы до xlist[k] вставить xlist**2[k]

    CtC = np.dot(RM.transpose(), RM)         # C^T * C
    print (CtC) # для отладки
    CM = np.linalg.inv(CtC)                  # (C^T * C)^-1
    print (CM) # для отладки
    CtCC = np.dot(CM, RM.transpose())        # (C^T * C)^-1 * C^T
    print (CtCC) # для отладки
    params = np.dot(CtCC, ylst)              # (C^T * C)^-1 * C^T * y   - искомые параметры
 
    zlst = np.dot(RM, params)       # значения для построения МНК-кривой и аппроксимации

    print("Оценённые параметры:")
    print(f"a = {params[0]:.4f}")
    print(f"b = {params[1]:.4f}")
    # print(f"c = {params[2]:.4f}") # раскомментировать, если МНК-парабола; добавить строки до нужной размерности

    # построение графика МНК-кривой, аппроксимирующей наблюдения
    plt.figure()
    plt.plot(xlst, zlst, 'k', linewidth=2.0, label='МНК-кривая')
    plt.plot(xlst, ylst, '*r', label='Экспериментальные точки')
    plt.xlabel('x', fontsize=12, fontweight='bold')
    plt.ylabel('y', fontsize=12, fontweight='bold')
    plt.grid(True)
    plt.legend()
    plt.show()

# старт
if __name__ == "__main__":
    LQM()