import numpy as np
from .util.cgf import CGF
from .util.gto import GTO
from scipy.special import factorial, factorial2
from copy import deepcopy

class IntegralEvaluator:   
    def __init__(self, overlap, nuclear, eri):
        self._overlap_engine = overlap
        self._nuclear_engine = nuclear
        self._eri_engine= eri
    
    def overlap(self, cgf1:CGF, cgf2:CGF):
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
                     self.overlap_primitive(gto1, gto2)
                 s += t
        return s

    def overlap_primitive(self, gto1:GTO, gto2:GTO):
        return self._overlap_engine.overlap_primitive(gto1, gto2)
    
    def overlap_3d(self, p1, p2, alpha1, alpha2, o1, o2):
        return self._overlap_engine.overlap_3d(p1, p2, alpha1, alpha2, o1, o2)
    
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
                     gto1.norm * gto2.norm * \
                     self.kinetic_primitive(gto1, gto2)
                 s += t
        return s
    
    def kinetic_primitive(self, gto1, gto2):
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
            self.overlap_primitive(gto1, gto2)
        
        t1 = -2.0 * gto2.alpha**2 * ( \
            self.overlap_3d(gto1.p, gto2.p, 
                            gto1.alpha, gto2.alpha, 
                            gto1.o, gto2.o + np.array([2,0,0])) + 
            self.overlap_3d(gto1.p, gto2.p, 
                            gto1.alpha, gto2.alpha, 
                            gto1.o, gto2.o + np.array([0,2,0])) + 
            self.overlap_3d(gto1.p, gto2.p, 
                            gto1.alpha, gto2.alpha, 
                            gto1.o, gto2.o + np.array([0,0,2]))
            )
            
        t2 = -0.5 * np.sum(np.array(gto2.p * (gto2.p - np.array([1,1,1]))) *
                            np.array([
                              self.overlap_3d(gto1.p, gto2.p, 
                                              gto1.alpha, gto2.alpha, 
                                              gto1.o, gto2.o - np.array([2,0,0])) + 
                              self.overlap_3d(gto1.p, gto2.p, 
                                              gto1.alpha, gto2.alpha, 
                                              gto1.o, gto2.o - np.array([0,2,0])) + 
                              self.overlap_3d(gto1.p, gto2.p, 
                                              gto1.alpha, gto2.alpha, 
                                              gto1.o, gto2.o - np.array([0,0,2]))
                                    ])
                          )
            
        return t0 + t1 + t2
    
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
        assert len(nucleus) == 3

        v = 0.0
        for gto1 in cgf1.gtos:
            for gto2 in cgf2.gtos:
                 t = gto1.c * gto2.c * \
                     gto1.norm * gto2.norm * \
                     self.nuclear_primitive(gto1, gto2, nucleus)
                 v += t
        return float(charge) * v

    def nuclear_primitive(self, gto1:GTO, gto2:GTO, nuclear):
        return self._nuclear_engine.nuclear_primitive(gto1, gto2, nuclear)
        