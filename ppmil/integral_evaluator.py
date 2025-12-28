import numpy as np
from .util.cgf import CGF
from .util.gto import GTO
from .eri.teindex import teindex
from multiprocessing import Pool, cpu_count
from copy import deepcopy
from .math.math import gaussian_product_center

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
                     self._dipole(gto1.p, gto2.p, 
                                   gto1.alpha, gto2.alpha,
                                   gto1.o, gto2.o,
                                   cc, cref)
                 d += t
        return d
    
    def dipole_primitive(self, gto1, gto2, cc, cref=0.0):
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
               self._dipole(gto1.p, gto2.p, 
                            gto1.alpha, gto2.alpha,
                            gto1.o, gto2.o,
                            cc, cref)
    
    def _dipole(self, p1, p2, alpha1, alpha2, o1, o2, cc, cref):
        """
        Calculate 1D dipole integral using coefficients of two GTOs
        
        cc:   cartesian direction (0-2)
        cref: reference position (scalar)
        """
        rab2 = np.sum(np.power(p1-p2,2))
        gamma = alpha1 + alpha2
        
        # determine new product center
        p = gaussian_product_center(alpha1, p1, alpha2, p2)

        # determine correcting pre-factor
        pre = np.power(np.pi / gamma, 1.5) * \
              np.exp(-alpha1 * alpha2 * rab2 / gamma)
        
        # construct regular triple product
        w = np.zeros(3)
        for i in range(0,3):
            w[i] = self._overlap_engine._overlap_1d(o1[i], o2[i], 
                                                    p[i] - p1[i], 
                                                    p[i] - p2[i], gamma)       
            w[i] = self._overlap_engine._overlap_1d(o1[i], o2[i], 
                                                    p[i] - p1[i], 
                                                    p[i] - p2[i], gamma)
            w[i] = self._overlap_engine._overlap_1d(o1[i], o2[i], 
                                                    p[i] - p1[i], 
                                                    p[i] - p2[i], gamma)
        
        # construct adjusted triple product
        wd = np.copy(w)
        wd[cc] = self._overlap_engine._overlap_1d(o1[cc], o2[cc]+1, 
                                                  p[cc] - p1[cc], 
                                                  p[cc] - p2[cc], gamma)
        
        return pre * (np.prod(wd) + (p2[cc] - cref) * np.prod(w))
    
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
    
    
    def eri_tensor(self, cgfs, verbose=False):
        N = len(cgfs)
        tedouble, tejobs = self._build_jobs(N)

        nproc = cpu_count()
        chunks = np.array_split(tejobs, nproc)

        if verbose:
            print('Calculating electron repulsion integrals')
            print('Spawning %i threads' % nproc)
            print('Calculating %i ERI' % len(tedouble))

        with Pool(nproc) as pool:
            results = pool.map(
                IntegralEvaluator._eri_worker,
                [(self, cgfs, chunk.tolist()) for chunk in chunks]
            )

        for chunk in results:
            for idx, val in chunk:
                tedouble[idx] = val

        # expand tensor
        res = np.empty((N,N,N,N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        res[i,j,k,l] = tedouble[teindex(i,j,k,l)]

        return res
    
    @staticmethod
    def _eri_worker(args):
        self, cgfs, jobs = args
        out = []
        for idx, i, j, k, l in jobs:
            val = self.repulsion(cgfs[i], cgfs[j], cgfs[k], cgfs[l])
            out.append((idx, val))
        return out

    def repulsion(self, cgf1:CGF, cgf2:CGF, cgf3:CGF, cgf4:CGF):
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
        if not isinstance(cgf3, CGF):
            raise TypeError('Argument cgf3 must be of CGF type')
        if not isinstance(cgf4, CGF):
            raise TypeError('Argument cgf4 must be of CGF type')

        s = 0.0
        for gto1 in cgf1.gtos:
            for gto2 in cgf2.gtos:
                for gto3 in cgf3.gtos:
                    for gto4 in cgf4.gtos:
                        s += gto1.c * gto1.norm * \
                             gto2.c * gto2.norm * \
                             gto3.c * gto3.norm * \
                             gto4.c * gto4.norm * \
                             self._eri_engine.repulsion_primitive(gto1, gto2, gto3, gto4)
                        
        return s
    
    def repulsion_primitive(self, gto1:GTO, gto2:GTO, gto3:GTO, gto4:GTO):
        return self._eri_engine.repulsion_primitive(gto1, gto2, gto3, gto4)
    
    def _build_jobs(self, sz):
        # size of the packed ERI array
        max_idx = teindex(sz-1, sz-1, sz-1, sz-1)
        tedouble = [-1.0] * (max_idx + 1)

        jobs = []  # list of (idx, i, j, k, l)

        for i in range(sz):
            for j in range(sz):
                ij = i * (i + 1) // 2 + j

                for k in range(sz):
                    for l in range(sz):
                        kl = k * (k + 1) // 2 + l

                        if ij <= kl:
                            idx = teindex(i, j, k, l)

                            if idx >= len(tedouble):
                                raise RuntimeError(
                                    "Process tried to access illegal array position"
                                )

                            if tedouble[idx] < 0.0:
                                tedouble[idx] = 1.0
                                jobs.append((idx, i, j, k, l))

        return tedouble, jobs
    
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
            tplus = self.overlap_3d(gto1.p, gto2.p, 
                                      gto1.alpha, gto2.alpha, 
                                      gto1.o, gto2.o)
            gto1.o[coord] -= 2 # calculate l-1 term
            
            tmin = self.overlap_3d(gto1.p, gto2.p, 
                                     gto1.alpha, gto2.alpha, 
                                     gto1.o, gto2.o)
            
            gto1.o[coord] += 1 # recover
            
            return 2.0 * gto1.alpha * tplus - gto1.o[coord] * tmin
        
        else: # s-type
            gto1.o[coord] += 1
            t = self.overlap_3d(gto1.p, gto2.p, 
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
        nucm =  deepcopy(nuc)
        nucp =  deepcopy(nuc)
        
        if n1:
            cgf1m.p[coord] -= delta
            cgf1p.p[coord] += delta
            cgf1m.reset_gto_centers()
            cgf1p.reset_gto_centers()
        if n2:
            cgf2m.p[coord] -= delta
            cgf2p.p[coord] += delta
            cgf2m.reset_gto_centers()
            cgf2p.reset_gto_centers()
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
            tplus = self.repulsion_primitive(gto1_copy, gto2, gto3, gto4)
            gto1_copy.o[coord] -= 2 # calculate l-1 term
            
            tmin = self.repulsion_primitive(gto1_copy, gto2, gto3, gto4)
            
            gto1_copy.o[coord] += 1 # recover
            
            return 2.0 * gto1_copy.alpha * tplus - gto1_copy.o[coord] * tmin
        
        else: # s-type
            gto1_copy.o[coord] += 1
            
            t = self.repulsion_primitive(gto1_copy, gto2, gto3, gto4)
            
            gto1_copy.o[coord] -= 1 # recover terms
            
            return 2.0 * gto1_copy.alpha * t