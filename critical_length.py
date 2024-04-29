import matplotlib.ticker
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft


def Kuramoto_Sivashinsky(u0, u1, N, dx, L, nu, dt):
    k = fft.fftfreq(N, dx) * L
    u1_hat = fft.fft(u1)
    u1_hat2 = fft.fft(u1 * u1)
    u0_hat2 = fft.fft(u0 * u0)
    f = (2 * np.pi / L * k) ** 2 - nu * (2 * np.pi / L * k) ** 4
    u2_hat = (1 + dt / 2 * f) / (1 - dt / 2 * f) * u1_hat - 1j * np.pi / L * k * dt * (
            3 / 2 * u1_hat2 - u0_hat2 / 2) / (1 - dt / 2 * f)

    return fft.ifft(u2_hat).real


def computations(L_l):
    j = 0
    A = np.zeros_like(L_l, dtype=float)
    for L in L_l:
        x = np.linspace(0, L, N, False)
        dx = x[1]
        t = np.arange(0, t_max + dt, dt)
        u0 = np.cos(2 * np.pi * x / L) + 0.1 * np.cos(4 * np.pi * x / L)
        u1 = u0
        u2 = np.zeros_like(u0)

        for i in range(len(t) - 1):
            u2 = Kuramoto_Sivashinsky(u0, u1, N, dx, L, nu, dt)
            u0 = u1
            u1 = u2

        A[j] = np.sqrt(np.trapz(u2 * u2, dx=dx) / L)
        j += 1
    return A


# INPUT
nu = 2
N = 1024
dt = 1 / 16
t_max = 100
L_l = np.zeros(26)
L_l[:20] = np.arange(1, 21, 1)
L_l[20:] = np.arange(25,55,5)

A = computations(L_l)

plt.plot(L_l, A)
#plt.loglog()
plt.show()
