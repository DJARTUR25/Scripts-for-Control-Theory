import numpy as np
import matplotlib.pyplot as plt

def ols_estimation_1():
    xlst = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,
                     0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ylst = np.array([10.1267, 9.57937, 6.4242, 5.68413,
                     4.35654, 3.15887, 2.5492, 1.85007,
                     3.27934, 1.99238, 2.9698, 2.93076, 4.47339])

    count = len(xlst)

    # Построение матрицы регрессоров
    CM = np.zeros((count, 3))
    for k in range(count):
        CM[k, :] = [xlst[k]**2, xlst[k], 1]

    params, residuals, rank, s = np.linalg.lstsq(CM, ylst, rcond=None)

    zlst = CM @ params

    print("Оценённые параметры (a, b, c) для полинома ax² + bx + c:")
    print(f"a = {params[0]:.6f}")
    print(f"b = {params[1]:.6f}")
    print(f"c = {params[2]:.6f}")

    plt.figure()
    plt.plot(xlst, zlst, 'k', linewidth=2.0, label='МНК-аппроксимация')
    plt.plot(xlst, ylst, '*r', label='Экспериментальные точки')
    plt.xlabel('x', fontsize=12, fontweight='bold')
    plt.ylabel('y', fontsize=12, fontweight='bold')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ols_estimation_1()