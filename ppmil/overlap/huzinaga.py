import numpy as np
from scipy.special import factorial2

from .oe import OverlapEngine
from ..math.math import gaussian_product_center, binomial_prefactor

class HuzinagaOverlapEngine(OverlapEngine):
    
    def overlap_primitive(self, gto1, gto2):
        """
        Calculate overlap integral of two GTOs
        """
        return HuzinagaOverlapEngine.overlap_3d(gto1.p, gto2.p, 
               gto1.alpha, gto2.alpha, 
               gto1.o, gto2.o)

    @staticmethod
    def overlap_3d(p1, p2, alpha1, alpha2, o1, o2):
        """
        Calculate three-dimensional overlap integral
        """
        rab2 = np.sum(np.power(p1-p2,2))
        gamma = alpha1 + alpha2
        p = gaussian_product_center(alpha1, p1, alpha2, p2)

        pre = np.power(np.pi / gamma, 1.5) * \
              np.exp(-alpha1 * alpha2 * rab2 / gamma)
        
        wx = HuzinagaOverlapEngine.overlap_1d(o1[0], o2[0], p[0] - p1[0], p[0] - p2[0], gamma)
        wy = HuzinagaOverlapEngine.overlap_1d(o1[1], o2[1], p[1] - p1[1], p[1] - p2[1], gamma)
        wz = HuzinagaOverlapEngine.overlap_1d(o1[2], o2[2], p[2] - p1[2], p[2] - p2[2], gamma)
        
        return pre * wx * wy * wz
    
    @staticmethod
    def overlap_1d(l1, l2, x1, x2, gamma):
        """
        Calculate the one-dimensional component of the overlap integral
        """
        sm = 0
        
        for i in range(0, 1 + (l1 + l2) // 2):
            sm += binomial_prefactor(2*i, l1, l2, x1, x2) * \
                  (1 if i == 0 else factorial2(2 * i - 1)) / \
                  np.power(2 * gamma, i)
            
        return sm