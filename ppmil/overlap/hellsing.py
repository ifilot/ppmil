import numpy as np
from scipy.special import factorial

from .oe import OverlapEngine
from ..math.math import gaussian_product_center

class HellsingOverlapEngine(OverlapEngine):
    
    def overlap_primitive(self, gto1, gto2):
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
        eta = alpha1 * alpha2 / gamma
        p = gaussian_product_center(alpha1, p1, alpha2, p2)

        pre = np.power(np.pi / gamma, 1.5) * \
              np.exp(-alpha1 * alpha2 * rab2 / gamma)
        
        wx = self.overlap_1d(o1[0], o2[0], alpha1, alpha2, p1[0] - p2[0], gamma)
        wy = self.overlap_1d(o1[1], o2[1], alpha1, alpha2, p1[1] - p2[1], gamma)
        wz = self.overlap_1d(o1[2], o2[2], alpha1, alpha2, p1[2] - p2[2], gamma)
        
        return pre * wx * wy * wz
    
    def overlap_1d(self, l1, l2, alpha1, alpha2, x, gamma):
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