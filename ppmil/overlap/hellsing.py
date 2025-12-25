import numpy as np
from scipy.special import factorial
from numba import njit

from ..util.gto import GTO
from .overlap_engine import OverlapEngine

class HellsingOverlapEngine(OverlapEngine):
    
    def __init__(self, use_kernel=False):
        self.__use_kernel = use_kernel
        if self.__use_kernel:
            self._compute_kernel()

    def overlap_primitive(self, gto1:GTO, gto2:GTO):
        """
        Calculate overlap integral of two GTOs
        """
        return self.overlap_3d(gto1.p, gto2.p, 
               gto1.alpha, gto2.alpha, 
               gto1.o, gto2.o)

    def overlap_3d(self, p1, p2, alpha1, alpha2, o1, o2):
        """
        Calculate three-dimensional overlap integral
        """
        rab2 = np.sum(np.power(p1-p2,2))
        gamma = alpha1 + alpha2

        pre = np.power(np.pi / gamma, 1.5) * \
              np.exp(-alpha1 * alpha2 * rab2 / gamma)
        
        if self.__use_kernel:
            wx = self._overlap_1d_kernel(o1[0], o2[0], alpha1, alpha2, p1[0] - p2[0], gamma)
            wy = self._overlap_1d_kernel(o1[1], o2[1], alpha1, alpha2, p1[1] - p2[1], gamma)
            wz = self._overlap_1d_kernel(o1[2], o2[2], alpha1, alpha2, p1[2] - p2[2], gamma)
        else:
            wx = self._overlap_1d(o1[0], o2[0], alpha1, alpha2, p1[0] - p2[0], gamma)
            wy = self._overlap_1d(o1[1], o2[1], alpha1, alpha2, p1[1] - p2[1], gamma)
            wz = self._overlap_1d(o1[2], o2[2], alpha1, alpha2, p1[2] - p2[2], gamma)
        
        return pre * wx * wy * wz
    
    def _overlap_1d(self, l1, l2, alpha1, alpha2, x, gamma):
        """
        Calculate the one-dimensional component of the overlap integral
        """
        pre = factorial(l1) * factorial(l2) / \
              np.power(gamma, l1 + l2) / \
              np.power(-1, l1)

        sm = 0 # sum term
        for i1 in range(0, 1+l1//2):
            for i2 in range(0, 1+l2//2):
                j = l1 + l2 - 2*i1 - 2*i2
                for m in range(0, 1+j//2):
                    t1 = np.power(-1, m) * \
                         factorial(j) * \
                         np.power(alpha1, l2 - i1 - 2*i2 - m) * \
                         np.power(alpha2, l1 - 2*i1 - i2 - m) / \
                         np.power(4, i1 + i2 + m) / \
                         factorial(i1) / \
                         factorial(i2) / \
                         factorial(m)
                    t2 = np.power(gamma, 2*(i1+i2)+m) * \
                         np.power(x, j-2*m) / \
                         factorial(l1 - 2*i1) / \
                         factorial(l2 - 2*i2) / \
                         factorial(j - 2*m)
                    
                    sm += t1 * t2
            
        return pre * sm

    def _compute_kernel(self, lmax=3):
        self._kernel = [[None for _ in range(lmax+1)] for _ in range(lmax+1)]

        for i in range(lmax+1):
            for j in range(lmax+1):
                s, a1, a2, g, xp = self._calculate_coefficients(i, j)

                self._kernel[i][j] = (
                    np.asarray(s,  dtype=np.float64),
                    np.asarray(a1, dtype=np.int64),
                    np.asarray(a2, dtype=np.int64),
                    np.asarray(g,  dtype=np.int64),
                    np.asarray(xp, dtype=np.int64),
                )
        
        self._fact = np.array(
            [factorial(i) for i in range(lmax*3)],
            dtype=np.float64
        )

    def _calculate_coefficients(self, l1, l2):
        scalar_terms = []
        alpha1_terms = []
        alpha2_terms = []
        gamma_terms = []
        x_terms = []
        for i1 in range(0, 1+l1//2):
                for i2 in range(0, 1+l2//2):
                    j = l1 + l2 - 2*i1 - 2*i2
                    for m in range(0, 1+j//2):
                        s = np.power(4.0, i1 + i2 + m) * \
                            factorial(i1) * \
                            factorial(i2) * \
                            factorial(m) * \
                            factorial(l1 - 2*i1) * \
                            factorial(l2 - 2*i2) * \
                            factorial(j - 2*m)
                        s = factorial(j) * np.power(-1, m) / s
                        scalar_terms.append(s)
                        alpha1_terms.append(l2 - i1 - 2*i2 - m)
                        alpha2_terms.append(l1 - 2*i1 - i2 - m)
                        gamma_terms.append(2 * (i1+i2) + m)
                        x_terms.append(j-2*m)
        
        return (np.asarray(scalar_terms, dtype=np.float64), 
                np.asarray(alpha1_terms, dtype=np.int64), 
                np.asarray(alpha2_terms, dtype=np.int64), 
                np.asarray(gamma_terms, dtype=np.int64), 
                np.asarray(x_terms, dtype=np.int64))
    
    def _overlap_1d_kernel(self, l1, l2, alpha1, alpha2, x, gamma):
        if l1 < 0 or l2 < 0:
            return 0.0

        pre = self._fact[l1] * self._fact[l2] / \
              np.power(gamma, l1 + l2)

        if l1 == l2 == 0:
            return pre

        if l1 & 1:
            pre *= -1

        terms = self._kernel[l1][l2]
        if len(terms[0]) == 1:
            return pre * terms[0][0] * alpha1**terms[1][0] * alpha2**terms[2][0] * gamma**terms[3][0] * x**terms[4][0]
        # note: no more speed-up observed for in-lining the len(terms[0]) == 2 term    

        sm = 0.0
        for (s, a1, a2, g, xp) in zip(*terms):
            t = s * alpha1**a1 * alpha2**a2 * gamma**g * x**xp
            sm += t
        
        return pre * sm
    
    def _overlap_1d_kernel_numba(self, l1, l2, alpha1, alpha2, x, gamma):
        s, a1, a2, g, xp = self._kernel[l1][l2]
        return HellsingOverlapEngine._overlap_1d_numba(
            l1, l2,
            alpha1, alpha2, x, gamma,
            s, a1, a2, g, xp,
            self._fact
        )

    @staticmethod
    @njit(cache=True)
    def _overlap_1d_numba(
        l1, l2,
        alpha1, alpha2, x, gamma,
        s, a1, a2, g, xp,
        fact
    ):
        pre = fact[l1] * fact[l2] / gamma ** (l1 + l2)
        if l1 & 1:
            pre = -pre

        if l1 == 0 and l2 == 0:
            return pre

        sm = 0.0
        for i in range(len(s)):
            sm += (
                s[i]
                * alpha1 ** a1[i]
                * alpha2 ** a2[i]
                * gamma  ** g[i]
                * x      ** xp[i]
            )

        return pre * sm