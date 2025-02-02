import numpy as np
import scipy
from .cgf import CGF
from .gto import GTO
from .gamma import Fgamma
import importlib.util
import warnings
from scipy.special import factorial, factorial2, comb

class PPMIL:
    def __init__(self):
        pylebedev_spec = importlib.util.find_spec('pylebedev')
        if pylebedev_spec is None:
            warnings.warn(
                "Some functionality of PPMIL depends on PyLebedev. "
                "Please install PyLebedev. See: https://ppmil.imc-tue.nl/installation.html."
            )
    
    #
    # CGF INTEGRALS
    #
    
    def overlap(self, cgf1, cgf2):
        """
        Calculate overlap integral between two contracted Gaussian functions
        """
        # verify that variables are CGFS
        if not isinstance(cgf1, CGF):
            raise TypeError('Argument cgf1 must be of CGF type')
        if not isinstance(cgf2, CGF):
            raise TypeError('Argument cgf2 must be of CGF type')
        
        s = 0.0
        for gto1 in cgf1.gtos:
            for gto2 in cgf2.gtos:
                 t = gto1.c * gto2.c * \
                     gto1.norm * gto2.norm * \
                     self.__overlap_3d(gto1.p, gto2.p, 
                                       gto1.alpha, gto2.alpha,
                                       gto1.o, gto2.o)
                 s += t
        return s
    
    def kinetic(self, cgf1, cgf2):
        """
        Calculate kinetic integral between two contracted Gaussian functions
        """
        # verify that variables are CGFs
        if not isinstance(cgf1, CGF):
            raise TypeError('Argument cgf1 must be of CGF type')
        if not isinstance(cgf2, CGF):
            raise TypeError('Argument cgf2 must be of CGF type')
        
        s = 0.0
        for gto1 in cgf1.gtos:
            for gto2 in cgf2.gtos:
                 t = gto1.c * gto2.c * \
                     self.kinetic_gto(gto1, gto2)
                 s += t
        return s
    
    def dipole(self, cgf1, cgf2, cc, cref):
        """
        Calculate 1D-dipole integral between two contracted Gaussian functions
        
        cc:   cartesian direction (0-2)
        cref: reference position (scalar)
        """
        # verify that variables are CGFS
        if not isinstance(cgf1, CGF):
            raise TypeError('Argument cgf1 must be of CGF type')
        if not isinstance(cgf2, CGF):
            raise TypeError('Argument cgf2 must be of CGF type')

        d = 0.0
        for gto1 in cgf1.gtos:
            for gto2 in cgf2.gtos:
                 t = gto1.c * gto2.c * \
                     gto1.norm * gto2.norm * \
                     self.__dipole(gto1.p, gto2.p, 
                                   gto1.alpha, gto2.alpha,
                                   gto1.o, gto2.o,
                                   cc, cref)
                 #print(gto1.c, gto2.c, gto1.alpha, gto2.alpha,t)
                 d += t
        return d
    
    def nuclear(self, cgf1, cgf2, nucleus, charge):
        """
        Calculate 1D-dipole integral between two contracted Gaussian functions
        
        cgf1: Contracted Gaussian Function 1
        cgf2: Contracted Gaussian Function 2
        nucleus: nucleus position
        charge: nucleus charge
        """
        # verify that variables are CGFS
        if not isinstance(cgf1, CGF):
            raise TypeError('Argument cgf1 must be of CGF type')
        if not isinstance(cgf2, CGF):
            raise TypeError('Argument cgf2 must be of CGF type')

        v = 0.0
        for gto1 in cgf1.gtos:
            for gto2 in cgf2.gtos:
                 t = gto1.c * gto2.c * \
                     gto1.norm * gto2.norm * \
                     self.nuclear_gto(gto1, gto2, nucleus)
                 v += t
        return float(charge) * v

    def repulsion(self, cgf1, cgf2, cgf3, cgf4):
        """
        Calculate 1D-dipole integral between two contracted Gaussian functions
        
        cgf1: Contracted Gaussian Function 1
        cgf2: Contracted Gaussian Function 2
        cgf2: Contracted Gaussian Function 3
        cgf2: Contracted Gaussian Function 4
        """
        # verify that variables are CGFS
        if not isinstance(cgf1, CGF):
            raise TypeError('Argument cgf1 must be of CGF type')
        if not isinstance(cgf2, CGF):
            raise TypeError('Argument cgf2 must be of CGF type')

        s = 0.0
        for gto1 in cgf1.gtos:
            for gto2 in cgf2.gtos:
                for gto3 in cgf3.gtos:
                    for gto4 in cgf4.gtos:
                        s += gto1.c * gto1.norm * \
                             gto2.c * gto2.norm * \
                             gto3.c * gto3.norm * \
                             gto4.c * gto4.norm * \
                             self.__repulsion(gto1, gto2, gto3, gto4)
                        
        return s
    
    #
    # AUXILIARY FUNCTIONS
    #
    
    def overlap_gto(self, gto1, gto2):
        """
        Calculate overlap integral of two GTOs
        """
        # verify that variables are GTOs
        if not isinstance(gto1, GTO):
            raise TypeError('Argument gto1 must be of GTO type')
        if not isinstance(gto2, GTO):
            raise TypeError('Argument gto2 must be of GTO type')

        return gto1.norm * gto2.norm * \
               self.__overlap_3d(gto1.p, gto2.p, 
               gto1.alpha, gto2.alpha, 
               gto1.o, gto2.o)
    
    def dipole_gto(self, gto1, gto2, cc, cref=0.0):
        """
        Calculate dipole integral between two contracted Gaussian functions
        
        cc:   cartesian direction (0-2)
        cref: reference position (scalar)
        """
        # verify that variables are GTOs
        if not isinstance(gto1, GTO):
            raise TypeError('Argument gto1 must be of GTO type')
        if not isinstance(gto2, GTO):
            raise TypeError('Argument gto2 must be of GTO type')

        return gto1.norm * gto2.norm * \
               self.__dipole(gto1.p, gto2.p, 
                             gto1.alpha, gto2.alpha,
                             gto1.o, gto2.o,
                             cc, cref)               
    
    def kinetic_gto(self, gto1, gto2):
        """
        Calculate kinetic integral of two GTOs
        """
        # verify that variables are GTOs
        if not isinstance(gto1, GTO):
            raise TypeError('Argument gto1 must be of GTO type')
        if not isinstance(gto2, GTO):
            raise TypeError('Argument gto2 must be of GTO type')
            
        # each kinetic integral can be expanded as a series of overlap
        # integrals using Gaussian recursion formulas
        
        t0 = gto2.alpha * (2.0 * np.sum(gto2.o) + 3.0) * \
            self.overlap_gto(gto1, gto2)
        
        t1 = -2.0 * gto2.alpha**2 * ( \
            self.__overlap_3d(gto1.p, gto2.p, 
                              gto1.alpha, gto2.alpha, 
                              gto1.o, gto2.o + np.array([2,0,0])) + 
            self.__overlap_3d(gto1.p, gto2.p, 
                              gto1.alpha, gto2.alpha, 
                              gto1.o, gto2.o + np.array([0,2,0])) + 
            self.__overlap_3d(gto1.p, gto2.p, 
                              gto1.alpha, gto2.alpha, 
                              gto1.o, gto2.o + np.array([0,0,2]))
            )
            
        t2 = -0.5 * np.sum(np.array(gto2.p * (gto2.p - np.array([1,1,1]))) *
                            np.array([
                              self.__overlap_3d(gto1.p, gto2.p, 
                                                gto1.alpha, gto2.alpha, 
                                                gto1.o, gto2.o - np.array([2,0,0])) + 
                              self.__overlap_3d(gto1.p, gto2.p, 
                                                gto1.alpha, gto2.alpha, 
                                                gto1.o, gto2.o - np.array([0,2,0])) + 
                              self.__overlap_3d(gto1.p, gto2.p, 
                                                gto1.alpha, gto2.alpha, 
                                                gto1.o, gto2.o - np.array([0,0,2]))
                                    ])
                          )
            
        return t0 + gto1.norm * gto2.norm * (t1 + t2)
    
    def nuclear_gto(self, gto1, gto2, nucleus):
        """
        Calculate nuclear attraction integral for two GTOs
        """
        return self.__nuclear(gto1.p, gto1.o, gto1.alpha,
                              gto2.p, gto2.o, gto2.alpha,
                              nucleus)
    
    def repulsion_gto(self, gto1, gto2, gto3, gto4):
        """
        Calculate two-electron integral for four gtos
        """
        return self.__repulsion(gto1, gto2, gto3, gto4)

    def __dipole(self, p1, p2, alpha1, alpha2, o1, o2, cc, cref):
        """
        Calculate 1D dipole integral using coefficients of two GTOs
        
        cc:   cartesian direction (0-2)
        cref: reference position (scalar)
        """
        rab2 = np.sum(np.power(p1-p2,2))
        gamma = alpha1 + alpha2
        
        # determine new product center
        p = self.__gaussian_product_center(alpha1, p1, alpha2, p2)

        # determine correcting pre-factor
        pre = np.power(np.pi / gamma, 1.5) * \
              np.exp(-alpha1 * alpha2 * rab2 / gamma)
        
        # construct regular triple product
        w = np.zeros(3)
        for i in range(0,3):
            w[i] = self.__overlap_1d(o1[i], o2[i], 
                                     p[i] - p1[i], 
                                     p[i] - p2[i], gamma)       
            w[i] = self.__overlap_1d(o1[i], o2[i], 
                                     p[i] - p1[i], 
                                     p[i] - p2[i], gamma)
            w[i] = self.__overlap_1d(o1[i], o2[i], 
                                     p[i] - p1[i], 
                                     p[i] - p2[i], gamma)
        
        # construct adjusted triple product
        wd = np.copy(w)
        wd[cc] = self.__overlap_1d(o1[cc], o2[cc]+1, 
                                   p[cc] - p1[cc], 
                                   p[cc] - p2[cc], gamma)
        
        return pre * (np.prod(wd) + (p2[cc] - cref) * np.prod(w))
    
    def __nuclear(self, a, o1, alpha1, b, o2, alpha2, c):
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
        p = self.__gaussian_product_center(alpha1, a, alpha2, b)
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
    
    def __repulsion(self, gto1, gto2, gto3, gto4):
        rab2 = np.sum(np.power(gto1.p - gto2.p,2))
        rcd2 = np.sum(np.power(gto3.p - gto4.p,2))

        p = self.__gaussian_product_center(gto1.alpha, gto1.p, gto2.alpha, gto2.p)
        q = self.__gaussian_product_center(gto3.alpha, gto3.p, gto4.alpha, gto4.p)

        rpq2 = np.sum(np.power(p-q,2))

        gamma1 = gto1.alpha + gto2.alpha
        gamma2 = gto3.alpha + gto4.alpha
        delta = 0.25 * (1.0 / gamma1 + 1.0 / gamma2)

        b = []
        for i in range(0,3):
            b.append(self.__B_array(gto1.o[i], gto2.o[i], gto3.o[i], gto4.o[i],
                                    p[i], gto1.p[i], gto2.p[i], q[i], gto3.p[i], gto4.p[i],
                                    gamma1, gamma2, delta))

        s = 0.0
        for i in range(gto1.o[0] + gto2.o[0] + gto3.o[0] + gto4.o[0]+1):
            for j in range(gto1.o[1] + gto2.o[1] + gto3.o[1] + gto4.o[1]+1):
                for k in range(gto1.o[2] + gto2.o[2] + gto3.o[2] + gto4.o[2]+1):
                    s += b[0][i] * b[1][j] * b[2][k] * Fgamma(i+j+k, 0.25 * rpq2 / delta)

        return 2.0 * np.power(np.pi, 2.5) / (gamma1 * gamma2 * np.sqrt(gamma1+gamma2)) * \
               np.exp(-gto1.alpha * gto2.alpha * rab2 / gamma1) * \
               np.exp(-gto3.alpha * gto4.alpha * rcd2 / gamma2) * s
    
    def __overlap_3d(self, p1, p2, alpha1, alpha2, o1, o2):
        """
        Calculate three-dimensional overlap integral
        """
        rab2 = np.sum(np.power(p1-p2,2))
        gamma = alpha1 + alpha2
        p = self.__gaussian_product_center(alpha1, p1, alpha2, p2)

        pre = np.power(np.pi / gamma, 1.5) * \
              np.exp(-alpha1 * alpha2 * rab2 / gamma)
        
        wx = self.__overlap_1d(o1[0], o2[0], p[0] - p1[0], p[0] - p2[0], gamma)
        wy = self.__overlap_1d(o1[1], o2[1], p[1] - p1[1], p[1] - p2[1], gamma)
        wz = self.__overlap_1d(o1[2], o2[2], p[2] - p1[2], p[2] - p2[2], gamma)
        
        return pre * wx * wy * wz
    
    def __overlap_1d(self, l1, l2, x1, x2, gamma):
        """
        Calculate the one-dimensional component of the overlap integral
        """
        sm = 0
        
        for i in range(0, 1 + (l1 + l2) // 2):
            sm += self.__binomial_prefactor(2*i, l1, l2, x1, x2) * \
                  (1 if i == 0 else factorial2(2 * i - 1)) / \
                  np.power(2 * gamma, i)
            
        return sm
    
    def __gaussian_product_center(self, alpha1, a, alpha2, b):
        """
        Calculate the position of the product of two Gaussians
        """
        return (alpha1 * a + alpha2 * b) / (alpha1 + alpha2)
    
    def __binomial_prefactor(self, s, ia, ib, xpa, xpb):
        sm = 0
        
        for t in range(0, s+1):
            if (s - ia) <= t and t <= ib:
                sm += scipy.special.binom(ia, s-t) * scipy.special.binom(ib, t) * \
                      np.power(xpa, ia-s+t) * np.power(xpb, ib-t)
        
        return sm
    
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
        return (-1)**i * self.__binomial_prefactor(i, l1, l2, pax, pbx) * \
               (-1)**u * factorial(i) * np.power(cpx,i - 2*r - 2*u) * \
               np.power(0.25/gamma,r+u) / factorial(r) / factorial(u) / \
               factorial(i - 2*r - 2*u)

    def __B_array(self, l1, l2, l3, l4, p, a, b, q, c, d, g1, g2, delta):

        imax = l1 + l2 + l3 + l4 + 1
        arr_b = np.zeros(imax)

        for i1 in range(l1+l2+1):
            for i2 in range(l3+l4+1):
                for r1 in range(i1//2+1):
                    for r2 in range(i2//2+1):
                        for u in range((i1+i2)//2 - r1 - r2 + 1):
                            i = i1 + i2 - 2 * (r1 + r2) - u
                            arr_b[i] += self.__B_term(i1, i2, r1, r2, u,
                                                      l1, l2, l3, l4,
                                                      p, a, b, q, c, d,
                                                      g1, g2, delta)

        return arr_b

    def __B_term(self, i1, i2, r1, r2, u, l1, l2, l3, l4, 
                 px, ax, bx, qx, cx, dx, gamma1, gamma2, delta):
        return self.__fB(i1, l1, l2, px, ax, bx, r1, gamma1) * \
               np.power(-1, i2) * self.__fB(i2, l3, l4, qx, cx, dx, r2, gamma2) * \
               np.power(-1, u) * self.__fact_ratio2(i1 + i2 - 2 * (r1 + r2), u) * \
               np.power(qx - px, i1 + i2 - 2 * (r1 + r2) - 2 * u) / \
               np.power(delta, i1+i2 - 2 * (r1 + r2) - u)

    def __fB(self, i, l1, l2, p, a, b, r, g):
        return self.__binomial_prefactor(i, l1, l2, p-a, p-b) * self.__B0(i, r, g)

    def __B0(self, i, r, g):
        return self.__fact_ratio2(i,r) * np.power(4 * g, r-i)

    def __fact_ratio2(self, a, b):
        return factorial(a) / factorial(b) / factorial(a - 2*b)

    #
    # DERIVATIVE FUNCTIONS
    #
    
    def overlap_deriv(self, cgf1, cgf2, nucleus, coord):
        """
        Calculate overlap integral between two contracted Gaussian functions
        """
        # verify that variables are CGFS
        if not isinstance(cgf1, CGF):
            raise TypeError('Argument cgf1 must be of CGF type')
        if not isinstance(cgf2, CGF):
            raise TypeError('Argument cgf2 must be of CGF type')
        
        # early exit if the CGF resides on the nucleus
        cgf1_nuc = np.linalg.norm(cgf1.p - nucleus) < 1e-3
        cgf2_nuc = np.linalg.norm(cgf2.p - nucleus) < 1e-3

        
        # if both atoms are on the same nucleus or if neither atom is on
        # the nucleus, then the result for the overlap derivatives will
        # be zero
        if cgf1_nuc == cgf2_nuc:
            return 0.0
        
        s = 0.0

        for gto1 in cgf1.gtos:
            for gto2 in cgf2.gtos:
                if not cgf1_nuc:
                    t1 = 0
                else:
                     t1 = self.__overlap_deriv_gto(gto1, gto2, coord)
                if not cgf2_nuc:
                    t2 = 0
                else:
                    t2 = self.__overlap_deriv_gto(gto2, gto1, coord)

                s += gto1.c * gto2.c * \
                     gto1.norm * gto2.norm * (t1 + t2)
        return s
    
    def __overlap_deriv_gto(self, gto1, gto2, coord):
        
        orders = gto1.o.copy()
        
        if gto1.o[coord] != 0:
            orders[coord] += 1
            tplus = self.__overlap_3d(gto1.p, gto2.p, 
                                      gto1.alpha, gto2.alpha, 
                                      orders, gto2.o)
            orders[coord] -= 2
            
            tmin = self.__overlap_3d(gto1.p, gto2.p, 
                                     gto1.alpha, gto2.alpha, 
                                     orders, gto2.o)
            
            orders[coord] += 1 # recover
            
            return 2.0 * gto1.alpha * tplus - orders[coord] * tmin
        
        else: # s-type
            orders[coord] += 1
            t = self.__overlap_3d(gto1.p, gto2.p, 
                                  gto1.alpha, gto2.alpha, 
                                  orders, gto2.o)
            
            return 2.0 * gto1.alpha * t