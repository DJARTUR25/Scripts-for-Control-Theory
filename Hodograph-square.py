import numpy as np              
import matplotlib.pyplot as plt 

# скрипт для построения годографа Цыпкина-Поляка и квадратов робастной устойчивости для исследуемого полинома
def robust_stability_square():
    
    # ВВОД 
    ncffs = [433.5, 667.25, 502.72, 251.25, 80.25, 14, 1]   # коэф-ты номин.полинома (а^0); от старшей степени к младшей
    bcffs = [43.35, 33.36, 25.137, 15.075, 5.6175, 1.4, 0.1]    # коэффициенты граничного полинома (альфы)
    polydim = len(ncffs) # количество коэффициентов


    # формирование полинома U(w)
    u0cffs = np.zeros(polydim)      # числитель U(w), коэффициенты u_0(w)
    uacffs = np.zeros(polydim)      # знаменатель U(w), коэффициенты u_a(w)
    for k in range(polydim, 0, -2): # записывает коэф-ты от четных степеней; 
        idx = k - 1
        u0cffs[idx] = ((-1) ** ((polydim - k) // 2)) * ncffs[idx]   # присваиваем их с чередующимся знаком
        uacffs[idx] = bcffs[idx]


    # формирование полинома V(w)
    v0cffs = np.zeros(polydim)      # числитель V(w), коэффициенты v_0(w)
    vacffs = np.zeros(polydim)      # знаменатель V(w), коэффициенты v_a(w)
    for k in range(polydim - 1, 0, -2):
        idx = k 
        v0cffs[idx] = ((-1) ** ((polydim - 1 - k) // 2)) * ncffs[k - 1] # с чередующимися знаками от нечетных степеней
        vacffs[idx] = bcffs[k - 1]


    # вычисление точек годографа Цыпкина-Поляка
    count = 5001 # количество точек для построения годографа; чем больше, тем точнее годограф и дольше подсчет
    omega_ticks = np.linspace(0, 100, count)    # такты частоты, от 0 до 100 (чтобы не строилось от 0 до inf)
    xy_ticks = np.zeros((2, count))     # массив для хранения координат точек годографа, соотв. (Re, Im)

    for k in range(count):
        w = omega_ticks[k]          # текущая частота
        den_u = np.polyval(uacffs, w)   # через np.polyval вычисляем значение u_a(w) для данного w (знаменатель U(w))
        den_v = np.polyval(vacffs, w)   # через np.polyval вычисляем значение v_a(w) для данного w (знаменатель V(w))
        xy_ticks[0, k] = np.polyval(u0cffs, w) / den_u if den_u != 0 else np.nan    # считаем u_0(w)/u_a(w), если u_a не ноль; иначе NaN
        xy_ticks[1, k] = np.polyval(v0cffs, w) / den_v if den_v != 0 else np.nan    # аналогично для v_0(w)/v_a(w)


    # поиск точек касания с квадратом робастной устойчивости
    wp = np.convolve(u0cffs, vacffs) - np.convolve(v0cffs, uacffs)  # вспомогательные полиномы
    wm = np.convolve(u0cffs, vacffs) + np.convolve(v0cffs, uacffs)  # np.convolve делает свертку
    
    rts = np.roots(wp).tolist() + np.roots(wm).tolist() # корни обоих полиномов, могут быть комплексными
    rts = np.array(rts) # в массив для обработки
    rts = rts[np.abs(np.imag(rts)) < 1e-10]     # оставляем те, чья мнимая часть близка к 0
    rts = rts[np.real(rts) > 0]                 # оставляем только те, что лежат в правой полуплоскости
    rts = np.real(rts)                          # оставляем только вещественную часть

    # подсчет радиусов робастной устойчивости для найденных точек касания 
    deltas = []         
    for w in rts:           # для каждой найденной частоты касания считаем радиус робастной устойчивости (от начала координат до грани квадрата)
        deltas.append(np.abs(np.polyval(u0cffs, w) / np.polyval(uacffs, w)))    # считаем как x(w) = |u_0(w)/u_a(w)|
    deltas = np.array(sorted(deltas))          # сортируем по возрастанию

    # построение графиков
    strs = [f'{d:.5g}' for d in deltas] # легенда
    my_col = plt.cm.jet(np.linspace(0, 1, len(deltas)))[::-1] # цветовая гамма

    plt.figure(figsize=(7, 7))
    plt.plot(xy_ticks[0, :], xy_ticks[1, :], 'b', linewidth=2.0, label='Годограф')
    for i, delta in enumerate(deltas):
        vtx = np.array([[ delta, -delta, -delta,  delta,  delta],       # квадрат со стороной 2*delta
                        [ delta,  delta, -delta, -delta,  delta]])
        plt.plot(vtx[0, :], vtx[1, :], color=my_col[i], linewidth=2.0, label=f'{strs[i]}')

    # plt.grid(True)
    plt.axis('equal')   # одинаковый масштаб по осям
    plt.legend()
    plt.title('Годограф и квадраты робастной устойчивости')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.show()
    
    # вывод результатов
    print("Найденные радиусы (δ):")
    for i, d in enumerate(deltas):
        print(f"  {strs[i]}")
    print(f"\nδ* = {min(deltas):.5f}")
    
    a_0_star = ncffs[0] / bcffs[0] # коэффициент a* для сравнения с δ_max
    a_n_star = ncffs[-1] / bcffs[-1] # коэффициент a_n для сравнения с δ_max
    print(f"Коэффициент a* = {a_0_star:.5f}")
    print(f"Коэффициент a_n = {a_n_star:.5f}")

    delta_star = min(deltas)
    answer = min(delta_star, a_0_star, a_n_star) # наибольший радиус робастной устойчивости
    print(f"\nОтвет: δ_max = {answer:.5f}")

# запуск скрипта
if __name__ == '__main__':
    robust_stability_square()