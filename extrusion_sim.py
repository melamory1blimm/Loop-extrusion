# This is a sample Python script.
import math

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from celluloid import Camera
import scipy
#модель 0 - односторонняя экструзия, случайным образом выбрано направление;
# модель 1 - односторонняя экструзия, фиксированное направление
# модель 2 - двусторонняя экструзия
model = 0

dt = 0.1
v = 1
tau = 20
n_all = 5000 #общая длина выборки
n_ch = 100 #одновренменно наблюдается белков
t_full = round(n_all*tau/n_ch)
print("Общее время расчета",t_full)


scale_ar = []
g_ar_mean = []

ks_ar = []
p_ar = []
g = []
sum_ar_mean = []

gamma_range = [5]

def do_it(gamma, gamma_b, alpha, Plots = False):
    mean_ar = []
    camera = Camera(plt.figure())
    nums = 0
    A0 = 0
    A10 = 0
    A01 = 0
    A11 = 0
    dist = tau * v / gamma
    scale = dist * n_ch
    #времена посадки когезина
    t_start = numpy.random.uniform(low=0, high = t_full, size=n_all)
    t_life = numpy.random.exponential(scale=tau, size=n_all)
    t_end = t_start + t_life
    #расположения когезинов
    pos = numpy.random.uniform(low=0, high = scale, size=n_all)
    if model==0:
        dir = numpy.random.randint(low=0, high = 2, size=n_all)
    elif model==1:
        dir = np.zeros(n_all)
    elif model==2:
        dir = 2*np.ones(n_all)
    l_right = np.zeros(n_all)
    l_left = np.zeros(n_all)

    #расстояние между препятствиями
    d_obst = v*tau / gamma_b
    # количество препятствий, наблюдаемых одновременно
    n_b = scale / d_obst
    # общее количество препятствий, наблюдаемых за время симуляции
    n_b_all = scale / d_obst * t_full / (tau / alpha)

    # времена посадки препятствий
    t_start_b = numpy.random.uniform(low=0, high=t_full, size=max(n_b_all,n_b))
    t_life_b = numpy.random.exponential(scale=tau/ alpha, size=max(n_b_all,n_b))
    t_end_b = t_start_b + t_life_b
    # расположения препятствий
    pos = numpy.random.uniform(low=0, high=scale, size=max(n_b_all,n_b))


    dl = dt*v

    stat_sum = []
    length_all = []
    corr = []
    length_right = []
    length_left = []
    g_fin_ar = []
    for t in np.arange(0,t_full,dt):
#       print(t)
        # извлекаем список актуальных белков
        ind = [i for i in range(n_all) if t_start[i] <= t and t_end[i] > t]
        num = len(ind)
        # извлекаем список актуальных положений
        pos_t = pos[ind]
        # извлекаем список длин справа и слева
        l_right_t = l_right[ind]
        l_left_t = l_left[ind]

        # извлекаем список направлений
        dir_t = dir[ind]
        # объединяем массив индексов и положений и сортируем список положений
        reform = np.reshape(np.dstack((ind, pos_t, l_right_t, l_left_t, dir_t)), (num, 5))
        res = reform[np.argsort(reform[:, 1])]

        for i in range(num):
            if (res[i,4]==0) or (res[i,4]==2):
                if i == num - 1:
                    res[i,2] = res[i, 2] + dl
                else:
                    for j in range(1,num-i):
                        if (res[i + j, 1] - res[i + j, 3] > res[i, 1] + res[i, 2] - 2*dl
                              and res[i + j, 1] - res[i + j, 3] < res[i, 1] + res[i, 2] + 2*dl) or \
                                (res[i + j, 1] + res[i + j, 2] > res[i, 1] + res[i, 2] - 2 * dl
                             and res[i + j, 1] + res[i + j, 2] < res[i, 1] + res[i, 2] + 2 * dl):
                            break
                        elif j==num-i-1:
                            res[i, 2] = res[i, 2] + dl


            if (res[i, 4] == 1) or (res[i,4]==2):
                if i == 0:
                    res[i, 3] = res[i, 3] + dl
                else:
                    for j in range(1, i+1):
                        if (res[i - j, 1] + res[i - j, 2] < res[i, 1] - res[i, 3] + 2 * dl and \
                                res[i - j, 1] + res[i - j, 2] > res[i, 1] - res[i, 3] - 2 * dl) or \
                                (res[i - j, 1] - res[i - j, 3] < res[i, 1] - res[i, 3] + 2 * dl and \
                                res[i - j, 1] - res[i - j, 3] > res[i, 1] - res[i, 3] - 2 * dl):
                            break
                        elif j == i:
                            res[i, 3] = res[i, 3] + dl

#                print(res[i,1],res[i,2],res[i,3])

#                plt.scatter(res[:, 1], np.zeros(num), s=100,color = 'red')
#                lines=[[res[j, 1]-res[j, 3],res[j, 1]+res[j, 2]] for j in range(num)]
#                zeros = []
#                for j in range(num):
#                    plt.plot([res[j, 1]-res[j, 3],res[j, 1]+res[j, 2]],[0,0])
#                plt.text(10,0.03, t)
#                plt.xlim((0,scale))
#                camera.snap()
        if t>t_full/2:
            length_all.append(res[:, 2]+res[:, 3])
            length_right.append(res[:, 2])
            length_left.append(res[:, 3])
            corr.append(res[:, 2] * res[:, 3])

        l_right = np.zeros(n_all)
        l_left = np.zeros(n_all)
        l_right[list(map(round, res[:, 0]))] = res[:, 2]
        l_left[list(map(round, res[:, 0]))] = res[:, 3]

    #размещение данных по массивах
    length_hist = [x for lista in length_right for x in lista if x>dl]
#        anim = camera.animate()
#        plt.show()
#        length_left = [x for lista in length_left for x in lista if x > dl]
#        length_right = [x for lista in length_right for x in lista if x > dl]
#        corr = [x for lista in corr for x in lista]
#        g_hist = [x for lista in g_fin_ar for x in lista if x>dl]
#        g_ar_mean.append(np.mean(g_hist)/dist)
#        sum_ar_mean.append(np.mean(stat_sum))
    print(gamma,scale,np.mean(length_hist)/(tau*v))
    mean_ar.append(np.mean(length_hist)/(tau*v))
    #        Построение гистограммы данных
    plt.hist(length_hist, bins=60, density=True, alpha=0.6, color='g', label='Данные')
#        loc,scale1=scipy.stats.expon.fit(length_left,floc=0)
#        loc, scale2 = scipy.stats.expon.fit(length_right, floc=0)
#        loc, scale3 = scipy.stats.expon.fit(length_hist, floc=0)
#        print(gamma)
    plt.xlabel('Length of loop, $\tilde \lambda$')
    plt.ylabel('PDF')
    if model == 0:
        plt.yscale('log')
    # Построение теоретической кривой экспоненциального распределения
#        x = np.linspace(0, np.max(g_hist), 1000)
#        pdf_fitted = scipy.stats.expon.pdf(x, 0, scale)
#        plt.plot(x, pdf_fitted, 'r-', label='Экспоненциальное распределение')
#        plt.title(r'$\gamma$='+str(gamma))
    plt.show()
    return mean_ar

for i in range(5):
    print(do_it())
