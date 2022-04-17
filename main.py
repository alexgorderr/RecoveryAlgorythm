import concurrent.futures
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import time

# число узлов крупной сетки
K = 25
# шаг
h = 1

# инициализация узлов u(i) крупной сетки на отрезке [0, K-1]
eps = np.finfo(float).eps
u = np.linspace(eps, (K - 1) * h, K)
# print(h, u[1]-u[0])

# инициализация значений функции phi
phi = np.sin(u) / u
# plt.scatter(u, phi)
# plt.show()

# число узлов мелкой сетки между двумя узлами крупной сетки
N = 3
N_ = N + 1

# число узлов мелкой сетки
M = N_ * (K - 1) + 1
M_ = N * (K - 1) + K
# print(M, M_)

# массив размером K значений phi
F = phi.copy()
# print(K, len(F))

# массив размера N_ значений ядра В.А. Стеклова без сглаживания
x = np.linspace(eps, h * N / N_, N_)
# global psi2
psi2 = [1 - abs(t) if abs(t) <= 1 else 0 for t in x]
# print(psi2)
# plt.plot(x, psi2)
# plt.show()

# массив размера 2*N_ значений ядра В.А. Стеклова со сглаживанием
x = np.linspace(eps, h * (2*N_-1)/(2*N_), 2 * N_)
# global psi4
psi4 = []
for t in x:
    v = abs(t)
    if v <= 1:
        r = 3 * v ** 3 - 6 * v ** 2 + 4
    elif 1 < v <= 2:
        r = - v ** 3 + 6 * t ** 2 - 12 * v + 8
    else:
        r = 0
    psi4.append(r / 6)


# print(psi4)
# plt.plot(x, psi4)
# plt.show()


def G_0(m):
    a1 = F[int(np.ceil(m / N_))]
    b1 = psi2[m % N_]
    v = a1 * b1
    if M - m > N_:
        a2 = F[int(np.ceil(m / N_)) + 1]
        b2 = psi2[N_ - (m % N_) - 1]
        v += a2 * b2
    return v


def G_2(m):
    v = 0
    if m > N_:
        a1 = F[int(np.ceil(m / N_)) - 1]
        b1 = psi4[N_ + (m % N_)]
        v += a1 * b1
    # \/ всегда выполняется \/
    a2 = F[int(np.ceil(m / N_))]
    b2 = psi4[m % N_]
    v += a2 * b2
    if M - m > N_:
        a3 = F[int(np.ceil(m / N_)) + 1]
        b3 = psi4[N_ - (m % N_)]
        v += a3 * b3
    if M - m > 2 * N_:
        a4 = F[int(np.ceil(m / N_)) + 2]
        b4 = psi4[2 * N_ - (m % N_) - 1]
        v += a4 * b4
    return v


def main():
    pool = mp.Pool(mp.cpu_count()-2)

    # массив размером M значение аппроксимирующей функции в узлах мелкой сетки
    x = np.linspace(np.finfo(float).eps, (K - 1) * h, M)
    # plt.scatter(x, [0.5 for _ in range(M)])
    # plt.show()

    G0 = []
    # рассчет G0 в цикле
    start = time.perf_counter()
    for m in range(M):
        # print(m, M)
        a1 = F[int(np.ceil(m / N_))]
        b1 = psi2[m % N_]
        v = a1 * b1
        if M - m > N_:
            a2 = F[int(np.ceil(m / N_)) + 1]
            b2 = psi2[N_ - (m % N_) - 1]
            v += a2 * b2
        G0.append(v)
    end = time.perf_counter()
    print(f'Ended in {end - start} seconds in a loop')

    # рассчет G0 асинхронно
    start = time.perf_counter()
    results = pool.map(G_0, [m for m in range(M)])
    end = time.perf_counter()
    print(f'Ended in {end - start} seconds asynchronously')

    # вывод исходных и восстановленных данных
    plt.scatter(u, phi, label='Исходные данные')
    plt.plot(x, G0, alpha=0.5, label='В цикле')
    plt.plot(x, results, alpha=0.5, label='Асинхронно')
    plt.title('G0 - без сглаживания')
    plt.legend()
    plt.show()

    G2 = []
    # рассчет G2 в цикле
    start = time.perf_counter()
    for m in range(M):
        # print(m, M)
        v = 0
        if m > N_:
            a1 = F[int(np.ceil(m / N_)) - 1]
            b1 = psi4[N_ + (m % N_)]
            v += a1 * b1
        # \/ всегда выполняется \/
        a2 = F[int(np.ceil(m / N_))]
        b2 = psi4[m % N_]
        v += a2 * b2
        if M - m > N_:
            a3 = F[int(np.ceil(m / N_)) + 1]
            b3 = psi4[N_ - (m % N_)]
            v += a3 * b3
        if M - m > 2 * N_:
            a4 = F[int(np.ceil(m / N_)) + 2]
            b4 = psi4[2 * N_ - (m % N_) - 1]
            v += a4 * b4
        G2.append(v)
    end = time.perf_counter()
    print(f'Ended in {end - start} seconds in a loop')

    # рассчет G2 в асинхронно
    start = time.perf_counter()
    results = pool.map(G_2, [m for m in range(M)])
    end = time.perf_counter()
    print(f'Ended in {end - start} seconds asynchronously')

    # вывод исходных и восстановленных данных
    plt.scatter(u, phi, label='Исходные данные')
    plt.plot(x, G2, alpha=0.5, label='В цикле')
    plt.plot(x, results, alpha=0.5, label='Асинхронно')
    plt.title('G2 - со сглаживанием')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
