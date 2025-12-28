import numpy as np
from scipy.special import factorial

from ..util.gto import GTO
from .electron_repulsion_engine import ElectronRepulsionEngine
from ..math.math import binomial_prefactor, gaussian_product_center
from ..math.gamma_numba import Fgamma

class HuzinagaElectronRepulsionEngine(ElectronRepulsionEngine):

    def __init__(self):
        super().__init__()

    def repulsion_primitive(self, gto1:GTO, gto2:GTO, gto3:GTO, gto4:GTO):
        """
        Calculate two-electron integral for four gtos
        """
        return self._repulsion(gto1, gto2, gto3, gto4)

    def _repulsion(self, gto1, gto2, gto3, gto4):
        rab2 = np.sum(np.power(gto1.p - gto2.p,2))
        rcd2 = np.sum(np.power(gto3.p - gto4.p,2))

        p = gaussian_product_center(gto1.alpha, gto1.p, gto2.alpha, gto2.p)
        q = gaussian_product_center(gto3.alpha, gto3.p, gto4.alpha, gto4.p)

        rpq2 = np.sum(np.power(p-q,2))

        gamma1 = gto1.alpha + gto2.alpha
        gamma2 = gto3.alpha + gto4.alpha
        delta = 0.25 * (1.0 / gamma1 + 1.0 / gamma2)

        b = []
        for i in range(0,3):
            b.append(self._B_array(gto1.o[i], gto2.o[i], gto3.o[i], gto4.o[i],
                                    p[i], gto1.p[i], gto2.p[i], q[i], gto3.p[i], gto4.p[i],
                                    gamma1, gamma2, delta))

        # pre-calculate Fgamma values
        nu_max = np.sum(gto1.o) + np.sum(gto2.o) + np.sum(gto3.o) + np.sum(gto4.o)
        fg = np.array([Fgamma(nu, 0.25 * rpq2 / delta) for nu in range(nu_max+1)])

        s = 0.0
        for i in range(gto1.o[0] + gto2.o[0] + gto3.o[0] + gto4.o[0]+1):
            for j in range(gto1.o[1] + gto2.o[1] + gto3.o[1] + gto4.o[1]+1):
                for k in range(gto1.o[2] + gto2.o[2] + gto3.o[2] + gto4.o[2]+1):
                    s += b[0][i] * b[1][j] * b[2][k] * fg[i+j+k]

        return 2.0 * np.power(np.pi, 2.5) / (gamma1 * gamma2 * np.sqrt(gamma1+gamma2)) * \
               np.exp(-gto1.alpha * gto2.alpha * rab2 / gamma1) * \
               np.exp(-gto3.alpha * gto4.alpha * rcd2 / gamma2) * s
    
    def _B_array(self, l1, l2, l3, l4, p, a, b, q, c, d, g1, g2, delta):

        imax = l1 + l2 + l3 + l4 + 1
        arr_b = np.zeros(imax)

        for i1 in range(l1+l2+1):
            for i2 in range(l3+l4+1):
                for r1 in range(i1//2+1):
                    for r2 in range(i2//2+1):
                        for u in range((i1+i2)//2 - r1 - r2 + 1):
                            i = i1 + i2 - 2 * (r1 + r2) - u
                            arr_b[i] += self._B_term(i1, i2, r1, r2, u,
                                                      l1, l2, l3, l4,
                                                      p, a, b, q, c, d,
                                                      g1, g2, delta)

        return arr_b

    def _B_term(self, i1, i2, r1, r2, u, l1, l2, l3, l4, 
                 px, ax, bx, qx, cx, dx, gamma1, gamma2, delta):
        return self._fB(i1, l1, l2, px, ax, bx, r1, gamma1) * \
               np.power(-1, i2) * self._fB(i2, l3, l4, qx, cx, dx, r2, gamma2) * \
               np.power(-1, u) * self._fact_ratio2(i1 + i2 - 2 * (r1 + r2), u) * \
               np.power(qx - px, i1 + i2 - 2 * (r1 + r2) - 2 * u) / \
               np.power(delta, i1+i2 - 2 * (r1 + r2) - u)

    def _fB(self, i, l1, l2, p, a, b, r, g):
        return binomial_prefactor(i, l1, l2, p-a, p-b) * self._B0(i, r, g)

    def _B0(self, i, r, g):
        return self._fact_ratio2(i,r) * np.power(4 * g, r-i)

    def _fact_ratio2(self, a, b):
        return self._fact[a] / self._fact[b] / self._fact[a - 2*b]