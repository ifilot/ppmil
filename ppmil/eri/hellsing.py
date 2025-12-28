import numpy as np
from scipy.special import factorial

from ..util.gto import GTO
from .electron_repulsion_engine import ElectronRepulsionEngine
from ..math.math import gaussian_product_center
from ..math.gamma_numba import Fgamma

class HellsingElectronRepulsionEngine(ElectronRepulsionEngine):

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

        a1 = gto1.alpha
        a2 = gto2.alpha
        a3 = gto3.alpha
        a4 = gto4.alpha
        gamma1 = a1 + a2
        gamma2 = a3 + a4
        eta = gamma1 * gamma2 / (gamma1 + gamma2)

        # collect coefficients
        # b[x][0] -> constants
        # b[x][1] -> mu_x, my_y, mu_z
        # b[x][2] -> u, v, w
        b = []
        for i in range(0,3):
            b.append(self._B_array(gto1.o[i], gto2.o[i], gto3.o[i], gto4.o[i],
                                   a1, a2, a3, a4,
                                   gto1.p[i], gto2.p[i], gto3.p[i], gto4.p[i],
                                   p[i], q[i], gamma1, gamma2))

        # pre-calculate Fgamma values
        nu_max = np.sum(gto1.o) + np.sum(gto2.o) + np.sum(gto3.o) + np.sum(gto4.o)
        fg = np.array([Fgamma(nu, eta * rpq2) for nu in range(nu_max+1)])

        s = 0.0
        for i in range(len(b[0])):
            for j in range(len(b[1])):
                for k in range(len(b[2])):
                    nu = b[0][i][1] + b[1][j][1] + b[2][k][1] - (b[0][i][2] + b[1][j][2] + b[2][k][2])
                    s += b[0][i][0] * b[1][j][0] * b[2][k][0] * fg[nu]

        return 2.0 * np.power(np.pi, 2.5) / (gamma1 * gamma2 * np.sqrt(gamma1+gamma2)) * \
               np.exp(-gto1.alpha * gto2.alpha * rab2 / gamma1) * \
               np.exp(-gto3.alpha * gto4.alpha * rcd2 / gamma2) * s
    
    def _B_array(self, l1, l2, l3, l4, 
                       a1, a2, a3, a4,
                       ax, bx, cx, dx, 
                       px, qx, g1, g2):

        pre1 = np.power(-1, l1+l2) * self._fact[l1] * self._fact[l2] / np.power(g1, l1 + l2)
        pre2 = self._fact[l3] * self._fact[l4] / np.power(g2, l3 + l4)
        eta = g1 * g2 / (g1 + g2)
        arr = []

        for i1 in range(l1//2+1):
            for i2 in range(l2//2+1):
                for o1 in range(l1 - 2*i1 + 1):
                    for o2 in range(l2 - 2*i2 + 1):
                        for r1 in range((o1 + o2)//2 + 1):
                            t11 = np.power(-1, o2+r1) * \
                                  self._fact[o1 + o2] / \
                                  np.power(4, i1 + i2 + r1) / \
                                  self._fact[i1] / \
                                  self._fact[i2] / \
                                  self._fact[o1] / \
                                  self._fact[o2] / \
                                  self._fact[r1]
                            t12 = np.power(a1, o2-i1-r1) * \
                                  np.power(a2, o1-i2-r1) * \
                                  np.power(g1, 2*(i1+i2)+r1) * \
                                  np.power(ax-bx, o1 + o2 - 2*r1) / \
                                  self._fact[l1 - 2*i1 - o1] / \
                                  self._fact[l2 - 2*i2 - o2] / \
                                  self._fact[o1 + o2 - 2*r1]
                            for i3 in range(l3//2+1):
                                for i4 in range(l4//2+1):
                                    for o3 in range(l3 - 2*i3 + 1):
                                        for o4 in range(l4 - 2*i4 + 1):
                                            for r2 in range((o3 + o4)//2 + 1):
                                                t21 = np.power(-1, o3+r2) * \
                                                      self._fact[o3 + o4] / \
                                                      np.power(4, i3 + i4 + r2) / \
                                                      self._fact[i3] / \
                                                      self._fact[i4] / \
                                                      self._fact[o3] / \
                                                      self._fact[o4] / \
                                                      self._fact[r2]
                                                t22 = np.power(a3, o4-i3-r2) * \
                                                      np.power(a4, o3-i4-r2) * \
                                                      np.power(g2, 2*(i3+i4)+r2) * \
                                                      np.power(cx-dx, o3 + o4 - 2*r2) / \
                                                      self._fact[l3 - 2*i3 - o3] / \
                                                      self._fact[l4 - 2*i4 - o4] / \
                                                      self._fact[o3 + o4 - 2*r2]
                                                mu = l1 + l2 + l3 + l4 - 2 * (i1 + i2 + i3 + i4) - (o1 + o2 + o3 + o4)
                                                for u in range(mu//2+1):
                                                    t3 = np.power(-1, u) * \
                                                         self._fact[mu] * \
                                                         np.power(eta, mu-u) * \
                                                         np.power(px - qx, mu-2*u) / \
                                                         np.power(4, u) / \
                                                         self._fact[u] / \
                                                         self._fact[mu - 2*u]
                                                    

                                                    arr.append((pre1 * pre2 * t11 * t12 * t21 * t22 * t3, mu, u))

        return arr