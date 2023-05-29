import numpy as np
import scipy
from .cgf import CGF
from .gto import GTO
import importlib.util
import warnings

class PPMIL:
    def __init__(self):
        pylebedev_spec = importlib.util.find_spec('pylebedev')
        if pylebedev_spec is None:
            warnings.warn(
                "Some functionality of PPMIL depends on PyLebedev. "
                "Please install PyLebedev. See: https://ppmil.imc-tue.nl/installation.html."
            )
    
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

        s = 0.0
        for gto1 in cgf1.gtos:
            for gto2 in cgf2.gtos:
                 t = gto1.c * gto2.c * \
                     gto1.norm * gto2.norm * \
                     self.__dipole(gto1.p, gto2.p, 
                                   gto1.alpha, gto2.alpha,
                                   gto1.o, gto2.o,
                                   cc, cref)
                 #print(gto1.c, gto2.c, gto1.alpha, gto2.alpha,t)
                 s += t
        return s
    
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
        