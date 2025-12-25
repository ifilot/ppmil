import numpy as np
from .util.cgf import CGF
from .util.gto import GTO
from scipy.special import factorial, factorial2
from copy import deepcopy

class IntegralEvaluator:   
    def __init__(self, overlap, kinetic, nuclear, eri):
        self._overlap_engine = overlap
        self._kinetic_engine = kinetic
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
        # verify that variables are GTOs
        if not isinstance(gto1, GTO):
            raise TypeError('Argument gto1 must be of GTO type')
        if not isinstance(gto2, GTO):
            raise TypeError('Argument gto2 must be of GTO type')

        return self._overlap_engine.overlap_primitive(gto1, gto2)