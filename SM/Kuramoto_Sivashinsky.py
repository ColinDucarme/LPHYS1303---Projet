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


# INPUT
nu = 1
N = 1024
L = 35
dt = 1/16
t_max = 1000

# COMPUTATIONS
x = np.linspace(0, L, N, False)
dx = x[1]
t = np.arange(0, t_max + dt, dt)
[xx, tt] = np.meshgrid(x, t)
uu = np.empty((len(t), len(x)))
u0 = np.cos(2 * np.pi * x / L) + 0.1 * np.cos(4 * np.pi * x / L)
u1 = u0
uu[0] = u0

for i in range(len(t) - 1):
    u2 = Kuramoto_Sivashinsky(u0, u1, N, dx, L, nu, dt)
    u0 = u1
    u1 = u2
    uu[i + 1] = u2

# PLOT
plt.title("Kuramoto-Sivashinsky - L={}".format(L))
plt.xlabel("x")
plt.ylabel("t")
c = np.max(np.max(np.abs(uu)))
img = plt.contourf(xx, tt, uu, np.linspace(-c, c, 101), cmap=matplotlib.colormaps["jet"])
plt.colorbar(img)
plt.show()
