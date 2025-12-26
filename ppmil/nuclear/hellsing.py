import numpy as np
from scipy.special import factorial

from ..util.gto import GTO
from .nuclear_engine import NuclearEngine
from ..math.math import gaussian_product_center
from ..math.gamma import Fgamma

class HellsingNuclearEngine(NuclearEngine):

    def __init__(self, use_kernel=False):
        self.__use_kernel = use_kernel
        if self.__use_kernel:
            self._compute_kernel()

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
        
        if self.__use_kernel:
            ax, mx = self._A_array_kernel(o1[0], o2[0], alpha1, alpha2, a[0]-b[0], gamma, p[0] - c[0])
            ay, my = self._A_array_kernel(o1[1], o2[1], alpha1, alpha2, a[1]-b[1], gamma, p[1] - c[1])
            az, mz = self._A_array_kernel(o1[2], o2[2], alpha1, alpha2, a[2]-b[2], gamma, p[2] - c[2])
        else:
            ax, mx = self._A_array(o1[0], o2[0], alpha1, alpha2, a[0]-b[0], gamma, p[0] - c[0])
            ay, my = self._A_array(o1[1], o2[1], alpha1, alpha2, a[1]-b[1], gamma, p[1] - c[1])
            az, mz = self._A_array(o1[2], o2[2], alpha1, alpha2, a[2]-b[2], gamma, p[2] - c[2])

        # pre-calculate nu values
        nu_max =  mx[0][0] + my[0][0] + mz[0][0] - (mx[0][1] + my[0][1] + mz[0][1])
        fg = np.array([Fgamma(nu, gamma*rcp2) for nu in range(nu_max+1)])

        s = 0
        for i in range(len(ax)):
            for j in range(len(ay)):
                for k in range(len(az)):
                    nu = mx[i][0] + my[j][0] + mz[k][0] - (mx[i][1] + my[j][1] + mz[k][1])
                    s += ax[i] * ay[j] * az[k] * fg[nu]
        
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
                                 factorial(o1) / \
                                 factorial(o2) / \
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
    
    def _A_array_kernel(self, l1, l2, alpha1, alpha2, x, gamma, pcx):
        terms = self._kernel[l1][l2]
        arr = np.empty(len(terms[0]))

        for i,(s,a1,a2,gam,xp,pcxp,_) in enumerate(zip(*terms)):
            arr[i] = (s * alpha1**a1 * alpha2**a2 / gamma**gam * x**xp * pcx**pcxp)

        return np.asarray(arr), terms[-1]
    
    def _compute_kernel(self, lmax=3):
        self._kernel = [[None for _ in range(lmax+1)] for _ in range(lmax+1)]

        self._fact = np.array(
            [factorial(i) for i in range(lmax*3)],
            dtype=np.float64
        )

        for i in range(lmax+1):
            for j in range(lmax+1):
                self._kernel[i][j] = self._calculate_coefficients(i, j)

    def _calculate_coefficients(self, l1, l2):
        scalar_terms = []
        alpha1_terms = []
        alpha2_terms = []
        gamma_terms = []
        x_terms = []
        pcx_terms = []
        mu_u = []

        pre = np.power(-1, l1+l2) * self._fact[l1] * self._fact[l2]

        for i1 in range(l1//2+1):
            for i2 in range(l2//2+1):
                for o1 in range(l1 - 2*i1+1):
                    for o2 in range(l2 - 2*i2+1):
                        for r in range((o1+o2)//2+1):
                            t1 = np.power(-1, o2+r) * \
                                 self._fact[o1 + o2] / \
                                 np.power(4, i1+i2+r) / \
                                 self._fact[i1] / \
                                 self._fact[i2] / \
                                 self._fact[o1] / \
                                 self._fact[o2] / \
                                 self._fact[r]
                            t2 = self._fact[l1-2*i1 - o1] * \
                                 self._fact[l2-2*i2 - o2] * \
                                 self._fact[o1 + o2 - 2*r]
                            
                            mu_x = l1 + l2 - 2*(i1+i2) - (o1+o2)
                            for u in range(mu_x//2+1):
                                t3 = np.power(-1, u) * \
                                     self._fact[mu_x] / \
                                     np.power(4, u) / \
                                     self._fact[u] / \
                                     self._fact[mu_x- 2*u]
                                
                                scalar_terms.append(pre * t1 / t2 * t3)
                                alpha1_terms.append(o2-i1-r)
                                alpha2_terms.append(o1-i2-r)
                                x_terms.append(o1+o2-2*r)
                                gamma_terms.append(o1 + o2 - r + u)
                                pcx_terms.append(mu_x - 2*u)
                                mu_u.append((mu_x, u))
        
        return (np.asarray(scalar_terms, dtype=np.float64), 
                np.asarray(alpha1_terms, dtype=np.int64), 
                np.asarray(alpha2_terms, dtype=np.int64), 
                np.asarray(gamma_terms, dtype=np.int64), 
                np.asarray(x_terms, dtype=np.int64),
                np.asarray(pcx_terms, dtype=np.int64),
                mu_u)