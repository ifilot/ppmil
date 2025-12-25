# -*- coding: utf-8 -*-

import numpy as np
import scipy.special

class GTO:
    """
    Primitive Gaussian Type Orbital (GTO) of Cartesian type
    """
    def __init__(self, c, alpha, p, o):
        """
        Build GTO
        
        c - linear expansion coefficient / normalization constant
        alpha - exponent
        p - position vector, 3-vector
        o - order in the cartesian directions, 3-vector
        """
        self.c = c
        self.alpha = alpha
        self.p = np.array(p, dtype=np.float64)
        self.o = np.array(o, dtype=int)
        self.norm = self.__calculate_norm()
    
    def __str__(self):
        return '%i %i %i' % (self.o[0], self.o[1], self.o[2])

    def get_amp(self, p):
        """
        Calculate the amplitude (value) of a GTO
        """
        pp = p - self.p
        return self.norm * \
            (p[0] - self.p[0])**self.o[0] * \
            (p[1] - self.p[1])**self.o[1] * \
            (p[2] - self.p[2])**self.o[2] * \
            np.exp(-self.alpha * np.sum(np.power(pp, 2)))
                        
    def __calculate_norm(self):
        """
        Adjust the coefficient c such that the GTO becomes normalized
        """
        nom = np.power(2.0, 2.0 * np.sum(self.o) + 3./2.) * \
              np.power(self.alpha, np.sum(self.o) + 3./2.)
        
        denom = (1 if self.o[0] < 1 else scipy.special.factorial2(2 * self.o[0] - 1)) * \
                (1 if self.o[1] < 1 else scipy.special.factorial2(2 * self.o[1] - 1)) * \
                (1 if self.o[2] < 1 else scipy.special.factorial2(2 * self.o[2] - 1)) * \
                np.power(np.pi, 3/2)
                
        return np.sqrt(nom / denom)