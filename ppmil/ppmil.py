import numpy as np
import scipy
from .cgf import CGF
from .gto import GTO
from .gamma import Fgamma
import importlib.util
import warnings
from scipy.special import factorial, comb

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
        cgf2: Contracted Gaussian Function 1
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
        
        return pre * (np.product(wd) + (p2[cc] - cref) * np.product(w))
    
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
    
    def __binomial_prefactor(self, s, ia, ib, xpa, xpb):
        s = 0.0
        
        for t in range(s+1):
            if ((s-ia) <= t) and (t <= ib):
                s += comb(ia, s-t) * \
                     comb(ib,t) * \
                     np.power(xpa,ia - s + t) * \
                     np.power(xpb, ib - t)
        
        return s
    
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
                  (1 if i == 0 else scipy.special.factorial2(2 * i - 1)) / \
                  np.power(2 * gamma, i)
            
        return sm
    
    def __gaussian_product_center(self, alpha1, a, alpha2, b):
        """
        Calculate the position of the product of two Gaussians
        """
        return (alpha1 * a + alpha2 * b) / (alpha1 + alpha2)
    
    def __binomial_prefactor(self, s, ia, ib, xpa, xpb):
        sm = 0 # summation term
        
        for t in range(0, s+1):
            if (s - ia) <= t and t <= ib:
                sm += scipy.special.binom(ia, s-t) * scipy.special.binom(ib, t) * \
                      np.power(xpa, ia-s+t) * np.power(xpb, ib-t)
        
        return sm
    
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