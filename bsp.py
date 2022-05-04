import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import time


class BSP:
    def __init__(self, K=10, h=1, N=4, r=0, p=2):
        # число узлов крупной сетки
        self.K = K

        # шаг
        self.h = h

        # число узлов мелкой сетки между двумя узлами крупной сетки
        self.N = N
        self.N_ = N + 1

        # коэффициент сглаживания
        self.r = r

        # число узлов мелкой сетки
        self.M = self.N_ * (self.K - 1) + 1
        # M_ = N * (K - 1) + K

        # число потоков
        self.p = p

    def get_psi(self) -> list:
        match self.r:
            case 0:
                # массив размера N_ значений ядра В.А. Стеклова без сглаживания
                x = np.linspace(0, self.N / self.N_, self.N_)
                psi2 = [1 - abs(t) for t in x]
                return psi2
            case 2:
                # массив размера 2N_ значений ядра В.А. Стеклова без сглаживания
                x = np.linspace(0, (self.N + self.N_) / self.N_, 2 * self.N_)
                psi4 = []
                for t in x:
                    v = abs(t)
                    if v <= 1:
                        r = 3 * v ** 3 - 6 * v ** 2 + 4
                    elif 1 < v <= 2:
                        r = - v ** 3 + 6 * t ** 2 - 12 * v + 8
                    psi4.append(r / 6)
                return psi4
            case other:
                raise ValueError('Некорректный коэффицент сглаживания')

    def get_precomputation(self) -> list:
        self.P2 = []
        match self.r:
            case 0:
                for k in range(self.K):
                    Ps = []
                    for n in range(self.N_):
                        P = self.phi[k] * self.psi[n]
                        Ps.append(P)
                    self.P2.append(Ps)
            case 2:
                for k in range(self.K):
                    Ps = []
                    for n in range(2 * self.N_):
                        P = self.phi[k] * self.psi[n]
                        Ps.append(P)
                    self.P2.append(Ps)
        return self.P2

    def fit(self):
        eps = np.finfo(float).eps
        # инициализация узлов u(i) крупной сетки на отрезке [0, K-1]
        self.u = np.linspace(eps, (self.K - 1) * self.h, self.K)
        # инициализация значений функции phi
        self.phi = np.sin(self.u) / self.u

        # инициальзация z-координаты
        self.z = np.linspace(0, 1, self.M)

        # массив размером K значений phi
        self.F = self.phi.copy()

        # инициализация массива ядер В.А. Стеклова
        self.psi = self.get_psi()

    def improved_compute_by_map(self):
        self.x = np.linspace(0, (self.K - 1) * self.h, self.M)
        ms = list(np.array_split(range(0, self.M), self.p))

        pool = mp.Pool(self.p)

        self.P2 = self.get_precomputation()

        self.time_start = time.perf_counter()
        self.G = []

        match self.r:
            case 0:
                results = pool.map(self.improved_compute_G0, ms)
            case 2:
                results = pool.map(self.improved_compute_G2, ms)
        self.time_end = time.perf_counter()
        for i in results:
            self.G.extend(i)

    def improved_compute_G0(self, ms):
        vs = []
        for m in ms:
            k = int(m / self.N_)
            n = m % self.N_
            v = self.P2[k][n]
            if self.M - m > self.N_:
                v += self.P2[k + 1][self.N_ - n - 1]
            vs.append(v)
        return vs

    def improved_compute_G2(self, ms):
        vs = []
        for m in ms:
            k = int(m / self.N_)
            n = m % self.N_
            v = 0
            if m > self.N_:
                v += self.P2[k-1][self.N_ + n]
            v += self.P2[k][n]
            if self.M - m > self.N_:
                v += self.P2[k + 1][self.N_ - n - 1]
            if self.M - m > 2*self.N_:
                v += self.P2[k+2][2*self.N_ - n - 1]
            vs.append(v)
        return vs

    def naive_compute_by_ppe(self):
        # массив размером M значение аппроксимирующей функции в узлах мелкой сетки
        self.x = np.linspace(0, (self.K - 1) * self.h, self.M)
        ms = list(np.array_split(range(0, self.M), self.p))

        K_p = self.K / self.p
        M_p = self.N * (K_p - 1) + K_p

        self.time_start = time.perf_counter()
        self.G = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.p) as executor:
            match self.r:
                case 0:
                    results = executor.map(self.compute_G0, ms)
                case 2:
                    results = executor.map(self.compute_G2, ms)
        self.time_end = time.perf_counter()
        for i in results:
            self.G.extend(i)

    def naive_compute_by_map(self):
        # массив размером M значение аппроксимирующей функции в узлах мелкой сетки
        self.x = np.linspace(0, (self.K - 1) * self.h, self.M)
        ms = list(np.array_split(range(0, self.M), self.p))

        K_p = self.K / self.p
        M_p = self.N * (K_p - 1) + K_p

        pool = mp.Pool(self.p)
        self.time_start = time.perf_counter()
        results = []
        self.G = []

        match self.r:
            case 0:
                results = pool.map(self.compute_G0, ms)
            case 2:
                results = pool.map(self.compute_G2, ms)
        pool.close()
        pool.join()

        self.time_end = time.perf_counter()
        for i in results:
            self.G.extend(i)

    def naive_compute_by_apply(self):
        # массив размером M значение аппроксимирующей функции в узлах мелкой сетки
        self.x = np.linspace(0, (self.K - 1) * self.h, self.M)
        ms = list(np.array_split(range(0, self.M), self.p))

        K_p = self.K / self.p
        M_p = self.N * (K_p-1) + K_p

        pool = mp.Pool(self.p)
        self.time_start = time.perf_counter()
        results = []
        self.G = []
        match r:
            case 0:
                for i in ms:
                    results.append(pool.apply(self.compute_G0, args=(i,)))
            case 2:
                for i in ms:
                    results.append(pool.apply(self.compute_G2, args=(i,)))
        pool.close()
        pool.join()
        self.time_end = time.perf_counter()
        for i in results:
            self.G.extend(i)

    def compute_G0(self, ms):
        vs = []
        for m in ms:
            phi_1 = self.F[int(m / self.N_)]
            psi_1 = self.psi[m % self.N_]
            v = phi_1 * psi_1
            if self.M - m > self.N_:
                phi_2 = self.F[int(m / self.N_) + 1]
                psi_2 = self.psi[self.N_ - (m % self.N_) - 1]
                v += phi_2 * psi_2
            vs.append(v)
        return vs

    def compute_G2(self, ms):
        vs = []
        for m in ms:
            v = 0
            if m > self.N_:
                phi_1 = self.F[int(m / self.N_) - 1]
                psi_1 = self.psi[self.N_ + (m % self.N_)]
                v += phi_1 * psi_1
            # \/ всегда выполняется \/
            phi_2 = self.F[int(m / self.N_)]
            psi_2 = self.psi[m % self.N_]
            v += phi_2 * psi_2
            if self.M - m > self.N_:
                phi_3 = self.F[int(m / self.N_) + 1]
                psi_3 = self.psi[self.N_ - (m % self.N_)]
                v += phi_3 * psi_3
            if self.M - m > 2 * self.N_:
                phi_4 = self.F[int(m / self.N_) + 2]
                psi_4 = self.psi[2 * self.N_ - (m % self.N_) - 1]
                v += phi_4 * psi_4
            vs.append(v)
        return vs

    def display_results_2D(self):
        # вывод исходных и восстановленных данных
        plt.scatter(self.u, self.phi, label='Исходные данные')
        plt.plot(self.x, self.G, label='Восстановленные данные')
        plt.title('G0 - без сглаживания' if self.r == 0 else 'G2 - со сглаживанием')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    def display_results_3D(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        z = np.linspace(0, 1, self.K)
        ax.scatter(self.u, self.phi, z, label='Исходные данные')
        ax.plot(self.x, self.G, self.z, label='Восстановленные данные')
        ax.set_title('G0 - без сглаживания' if self.r == 0 else 'G2 - со сглаживанием')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.legend()
        plt.show()

    def display_psi(self):
        match self.r:
            case 0:
                x = np.linspace(0, self.N / self.N_, self.N_)
                plt.plot(-x[::-1], self.psi[::-1], 'b--')
                plt.plot(x, self.psi, 'b')
            case 2:
                x = np.linspace(0, (self.N + self.N_) / self.N_, 2 * self.N_)
                plt.plot(-x[::-1], self.psi[::-1], 'b--')
                plt.plot(x, self.psi, 'b')
            case other:
                raise ValueError('Некорректный коэффицент сглаживания')
        plt.title('psi2' if self.r == 0 else 'psi4')
        plt.show()

    def computation_time(self):
        print(f'Время работы составило {self.time_end - self.time_start:.5f} секунд')
