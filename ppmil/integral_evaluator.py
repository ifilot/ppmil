import numpy as np
from .util.cgf import CGF
from .util.gto import GTO
from .eri.teindex import teindex
from multiprocessing import Pool, cpu_count

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