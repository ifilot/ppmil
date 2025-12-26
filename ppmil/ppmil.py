import numpy as np
import scipy
from .cgf import CGF
from copy import deepcopy

class IntegralEvaluator:   

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
        assert len(nucleus) == 3
        
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
                    t1 = 0.0
                else:
                    t1 = self.__overlap_deriv_primitive(gto1, gto2, coord)

                if not cgf2_nuc:
                    t2 = 0.0
                else:
                    t2 = self.__overlap_deriv_primitive(gto2, gto1, coord)

                s += gto1.c * gto2.c * \
                     gto1.norm * gto2.norm * (t1 + t2)
        return s
    
    def kinetic_deriv(self, cgf1, cgf2, nucleus, coord):
        """
        Calculate overlap integral between two contracted Gaussian functions
        """
        # verify that variables are CGFS
        if not isinstance(cgf1, CGF):
            raise TypeError('Argument cgf1 must be of CGF type')
        if not isinstance(cgf2, CGF):
            raise TypeError('Argument cgf2 must be of CGF type')
        assert len(nucleus) == 3
        
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
                    t1 = 0.0
                else:
                    t1 = self.__kinetic_deriv_primitive(gto1, gto2, coord)

                if not cgf2_nuc:
                    t2 = 0.0
                else:
                    t2 = self.__kinetic_deriv_primitive(gto2, gto1, coord)

                s += gto1.c * gto2.c * \
                     gto1.norm * gto2.norm * (t1 + t2)
        return s
    
    def __overlap_deriv_primitive(self, gto1, gto2, coord):
        """
        Overlap geometric derivative for two GTOs
        
        Technically speaking, the situation wherein gto1 == gto2 should be
        avoided as this will result in aliasing errors. However, this
        situation will never arise in the computation.
        """
        if gto1.o[coord] != 0:
            gto1.o[coord] += 1 # calculate l+1 term
            tplus = self.__overlap_3d(gto1.p, gto2.p, 
                                      gto1.alpha, gto2.alpha, 
                                      gto1.o, gto2.o)
            gto1.o[coord] -= 2 # calculate l-1 term
            
            tmin = self.__overlap_3d(gto1.p, gto2.p, 
                                     gto1.alpha, gto2.alpha, 
                                     gto1.o, gto2.o)
            
            gto1.o[coord] += 1 # recover
            
            return 2.0 * gto1.alpha * tplus - gto1.o[coord] * tmin
        
        else: # s-type
            gto1.o[coord] += 1
            t = self.__overlap_3d(gto1.p, gto2.p, 
                                  gto1.alpha, gto2.alpha, 
                                  gto1.o, gto2.o)
            
            gto1.o[coord] -= 1 # recover terms
            
            return 2.0 * gto1.alpha * t
        
    def __kinetic_deriv_primitive(self, gto1, gto2, coord):
        """
        Kinetic geometric derivative for two GTOs
        
        Technically speaking, the situation wherein gto1 == gto2 should be
        avoided as this will result in aliasing errors. However, this
        situation will never arise in the computation.
        """
        if gto1.o[coord] != 0:
            gto1.o[coord] += 1 # calculate l+1 term
            tplus = self.kinetic_primitive(gto1, gto2)

            gto1.o[coord] -= 2 # calculate l-1 term
            tmin = self.kinetic_primitive(gto1, gto2)
            
            gto1.o[coord] += 1 # recover terms
            
            return 2.0 * gto1.alpha * tplus - gto1.o[coord] * tmin
        
        else: # s-type
            gto1.o[coord] += 1 # calculate l+1 term
            t = self.kinetic_primitive(gto1, gto2)

            gto1.o[coord] -= 1 # recover terms
            
            return 2.0 * gto1.alpha * t
        
    def nuclear_deriv(self, cgf1, cgf2, nuc, charge, nucderiv, coord):
        """
        Calculate geometric derivative for nuclear integrals
        
        In contrast to the analytical solutions used previously, here
        a numerical solution is used
        """
        # verify that variables are CGFS
        if not isinstance(cgf1, CGF):
            raise TypeError('Argument cgf1 must be of CGF type')
        if not isinstance(cgf2, CGF):
            raise TypeError('Argument cgf2 must be of CGF type')
        assert len(nuc) == 3
        assert len(nucderiv) == 3
        
        # early exit if the CGF resides on the nucleus
        n1 = np.linalg.norm(cgf1.p - nucderiv) < 1e-3
        n2 = np.linalg.norm(cgf2.p - nucderiv) < 1e-3
        n3 = np.linalg.norm(nuc - nucderiv) < 1e-3

        if n1 == n2 == n3:
            return 0.0

        delta = 1e-5

        # create deepcopies of objects
        cgf1m = deepcopy(cgf1)
        cgf2m = deepcopy(cgf2)
        cgf1p = deepcopy(cgf1)
        cgf2p = deepcopy(cgf2)
        nucm = deepcopy(nuc)
        nucp = deepcopy(nuc)
        
        if n1:
            cgf1m.p[coord] -= delta
            cgf1p.p[coord] += delta
            cgf1m.reset_primitive_centers()
            cgf1p.reset_primitive_centers()
        if n2:
            cgf2m.p[coord] -= delta
            cgf2p.p[coord] += delta
            cgf2m.reset_primitive_centers()
            cgf2p.reset_primitive_centers()
        if n3:
            nucm[coord] -= delta
            nucp[coord] += delta
            
        vm = self.nuclear(cgf1m, cgf2m, nucm, charge)
        vp = self.nuclear(cgf1p, cgf2p, nucp, charge)
        
        return (vp - vm) / (2.0 * delta)
    
    def repulsion_deriv(self, cgf1, cgf2, cgf3, cgf4, nucleus, coord):
        """
        Calculate geometric derivative for repulsion integral 
        """
        # verify that variables are CGFS
        if not isinstance(cgf1, CGF):
            raise TypeError('Argument cgf1 must be of CGF type')
        if not isinstance(cgf2, CGF):
            raise TypeError('Argument cgf2 must be of CGF type')
        if not isinstance(cgf3, CGF):
            raise TypeError('Argument cgf3 must be of CGF type')
        if not isinstance(cgf4, CGF):
            raise TypeError('Argument cgf4 must be of CGF type')
        assert len(nucleus) == 3
        
        n1 = np.linalg.norm(cgf1.p - nucleus) < 1e-3
        n2 = np.linalg.norm(cgf2.p - nucleus) < 1e-3
        n3 = np.linalg.norm(cgf3.p - nucleus) < 1e-3
        n4 = np.linalg.norm(cgf4.p - nucleus) < 1e-3
        
        if n1 == n2 == n3 == n4:
            return 0.0

        s = 0.0
        for gto1 in cgf1.gtos:
            for gto2 in cgf2.gtos:
                for gto3 in cgf3.gtos:
                    for gto4 in cgf4.gtos:
                        pre = gto1.c * gto2.c * gto3.c * gto4.c
                        norms = gto1.norm * gto2.norm * gto3.norm * gto4.norm
                        
                        t1 = self.__repulsion_deriv_primitive(gto1, gto2, gto3, gto4, coord) if n1 else 0.0
                        t2 = self.__repulsion_deriv_primitive(gto2, gto1, gto3, gto4, coord) if n2 else 0.0
                        t3 = self.__repulsion_deriv_primitive(gto3, gto4, gto1, gto2, coord) if n3 else 0.0
                        t4 = self.__repulsion_deriv_primitive(gto4, gto3, gto1, gto2, coord) if n4 else 0.0                        
                        
                        s += pre * norms * (t1 + t2 + t3 + t4)
        
        return s
    
    def __repulsion_deriv_primitive(self, gto1, gto2, gto3, gto4, coord):
        """
        Calculate geometric derivative for repulsion integral of four GTOs        
        """
        # create deep copy else the adjustment below *will* affect
        # the integral in the situation when gto1 == gto_i where (i != 1)
        gto1_copy = deepcopy(gto1)
        
        if gto1_copy.o[coord] != 0:
            gto1_copy.o[coord] += 1 # calculate l+1 term
            tplus = self.__repulsion(gto1_copy, gto2, gto3, gto4)
            gto1_copy.o[coord] -= 2 # calculate l-1 term
            
            tmin = self.__repulsion(gto1_copy, gto2, gto3, gto4)
            
            gto1_copy.o[coord] += 1 # recover
            
            return 2.0 * gto1_copy.alpha * tplus - gto1_copy.o[coord] * tmin
        
        else: # s-type
            gto1_copy.o[coord] += 1
            
            t = self.__repulsion(gto1_copy, gto2, gto3, gto4)
            
            gto1_copy.o[coord] -= 1 # recover terms
            
            return 2.0 * gto1_copy.alpha * t