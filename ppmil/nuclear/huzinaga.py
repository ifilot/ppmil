import numpy as np
from scipy.special import factorial

from ..util.gto import GTO
from .nuclear_engine import NuclearEngine
from ..math.math import binomial_prefactor, gaussian_product_center
from ..math.gamma import Fgamma

class HuzinagaNuclearEngine(NuclearEngine):

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
        
        ax = self.__A_array(o1[0], o2[0], p[0] - a[0], p[0] - b[0], p[0] - c[0], gamma)
        ay = self.__A_array(o1[1], o2[1], p[1] - a[1], p[1] - b[1], p[1] - c[1], gamma)
        az = self.__A_array(o1[2], o2[2], p[2] - a[2], p[2] - b[2], p[2] - c[2], gamma)
        
        s = 0.0
        
        for i in range(o1[0] + o2[0] + 1):
            for j in range(o1[1] + o2[1] + 1):
                for k in range(o1[2] + o2[2] + 1):
                    s += ax[i] * ay[j] * az[k] * Fgamma(i+j+k,rcp2*gamma)
       
        return -2.0 * np.pi / gamma * np.exp(-alpha1*alpha2*rab2/gamma) * s
    
    def __A_array(self, l1, l2, pa, pb, cp, g):
        imax = l1 + l2 +1
        arr = np.zeros(imax)
        
        for i in range(imax):
            for r in range(i//2+1):
                for u in range((i-2*r)//2+1):
                    iI = i - 2*r - u
                    arr[iI] += self.__A_term(i, r, u, l1, l2, pa, pb, cp, g)
        
        return arr
                    
    def __A_term(self, i, r, u, l1, l2, pax, pbx, cpx, gamma):
        return (-1)**i * binomial_prefactor(i, l1, l2, pax, pbx) * \
               (-1)**u * factorial(i) * np.power(cpx,i - 2*r - 2*u) * \
               np.power(0.25/gamma,r+u) / factorial(r) / factorial(u) / \
               factorial(i - 2*r - 2*u)
