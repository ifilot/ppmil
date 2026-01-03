from ppmil.math.gamma_numba import Fgamma
import numpy as np
import matplotlib.pyplot as plt
import math

# build table
T_s = 1e-6
T_l = 30.0
N = 200

T_grid = np.logspace(np.log10(T_s), np.log10(T_l), N)

# tabulate scaled function G(T) = sqrt(T) * F0(T)
G_grid = np.array([math.sqrt(T) * Fgamma(0, T) for T in T_grid])

def F0_interp(T):
    if T < T_s:
        return 1.0 - T/3.0 + T*T/10.0

    if T > T_l:
        return math.sqrt(math.pi)/(2*math.sqrt(T))

    # interpolate G(T)
    i = np.searchsorted(T_grid, T) - 1
    i = max(0, min(i, N-2))

    # linear in logT-space
    t0, t1 = T_grid[i], T_grid[i+1]
    g0, g1 = G_grid[i], G_grid[i+1]

    w = (math.log(T) - math.log(t0)) / (math.log(t1) - math.log(t0))
    G = (1.0 - w)*g0 + w*g1

    return G / math.sqrt(T)

T = np.logspace(-5, 5, 100)
plt.semilogx(T, [Fgamma(0, t) for t in T], '--', label=r"Exact")
plt.semilogx(T, [F0_interp(t) for t in T], 'o', alpha=0.5, label=r"Interpolated")
plt.legend()
plt.show()