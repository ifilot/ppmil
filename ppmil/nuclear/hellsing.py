import numpy as np
from scipy.special import factorial

from ..util.gto import GTO
from .nuclear_engine import NuclearEngine
from ..math.math import gaussian_product_center
from ..math.gamma import Fgamma

class HellsingNuclearEngine(NuclearEngine):

    def nuclear_primitive(self, gto1:GTO, gto2:GTO, nucleus):
        """
        Calculate nuclear attraction integral for two GTOs
        """
        return self._nuclear(gto1.p, gto1.o, gto1.alpha,
                             gto2.p, gto2.o, gto2.alpha,
                             nucleus)

    def _nuclear(self, a, o1, alpha1, b, o2, alpha2, c):
        """
        Calculate nuclear term
        
        a:      position of gto1
        o1:     orders of gto1
        alpha1: exponent of gto1
        b:      position of gto2
        o2:     orders of gto2
        alpha2: exponent of gto2
        c:      position of nucleus
        """
        gamma = alpha1 + alpha2
        p = gaussian_product_center(alpha1, a, alpha2, b)
        rab2 = np.sum(np.power(a-b,2))
        rcp2 = np.sum(np.power(c-p,2))
        
        ax, mx = self._A_array(o1[0], o2[0], alpha1, alpha2, a[0]-b[0], gamma, p[0] - c[0])
        ay, my = self._A_array(o1[1], o2[1], alpha1, alpha2, a[1]-b[1], gamma, p[1] - c[1])
        az, mz = self._A_array(o1[2], o2[2], alpha1, alpha2, a[2]-b[2], gamma, p[2] - c[2])
        
        s = 0.0
        for i in range(len(ax)):
            for j in range(len(ay)):
                for k in range(len(az)):
                    nu = mx[i][0] + my[j][0] + mz[k][0] - (mx[i][1] + my[j][1] + mz[k][1])
                    s += ax[i] * ay[j] * az[k] * Fgamma(nu,gamma*rcp2)
       
        return -2.0 * np.pi / gamma * np.exp(-alpha1*alpha2*rab2/gamma) * s

    def _A_array(self, l1, l2, alpha1, alpha2, x, gamma, pcx):
        arr = []
        mu_u = []
        
        pre = np.power(-1, l1+l2) * factorial(l1) * factorial(l2)

        for i1 in range(l1//2+1):
            for i2 in range(l2//2+1):
                for o1 in range(l1 - 2*i1+1):
                    for o2 in range(l2 - 2*i2+1):
                        for r in range((o1+o2)//2+1):
                            t1 = np.power(-1, o2+r) * \
                                 factorial(o1 + o2) / \
                                 np.power(4, i1+i2+r) / \
                                 factorial(i1) / \
                                 factorial(i2) / \
                                 factorial(i1) / \
                                 factorial(i2) / \
                                 factorial(r)
                            t2 = np.power(alpha1, o2-i1-r) * \
                                 np.power(alpha2, o1-i2-r) * \
                                 np.power(x, o1+o2-2*r) / \
                                 factorial(l1 -2*i1 - o1) / \
                                 factorial(l2 -2*i2 - o2) / \
                                 factorial(o1 + o2 - 2*r)
                            
                            mu_x = l1 + l2 - 2*(i1+i2) - (o1+o2)
                            for u in range(mu_x//2+1):
                                t3 = np.power(-1, u) * \
                                     factorial(mu_x) * \
                                     np.power(pcx, mu_x - 2*u) / \
                                     np.power(4, u) / \
                                     factorial(u) / \
                                     factorial(mu_x - 2*u) / \
                                     np.power(gamma, o1 + o2 - r + u)

                                arr.append(t1 * t2 * t3)
                                mu_u.append((mu_x, u))

        return pre * np.asarray(arr), mu_u