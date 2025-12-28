import numpy as np
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from collections import defaultdict

from ..eri.teindex import teindex
from ..eri.electron_repulsion_engine import ElectronRepulsionEngine
from ..util.cgf import CGF

@dataclass
class Shell:
    center: int
    l: int
    ao_indices: list   # indices into cgfs
    nfunc: int

class ERIEvaluator:

    def __init__(self, engine:ElectronRepulsionEngine):
        self._eri_engine = engine

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

    def eri_tensor(self, cgfs, verbose=False):
        """
        Compute the full four-index electron repulsion integral (ERI) tensor
        over a list of contracted Gaussian functions (CGFs).

        The tensor is returned in expanded form:
            ERI[i, j, k, l] = (cgfs[i] cgfs[j] | cgfs[k] cgfs[l])

        Internally, ERIs are:
        1. Packed into a 1D array using permutational symmetry
        2. Distributed across multiple worker processes
        3. Computed only once per unique (i,j,k,l) combination
        4. Expanded back into a full 4D tensor

        Parameters
        ----------
        cgfs : list
            List of contracted Gaussian functions.
        verbose : bool, optional
            If True, print progress and parallelization info.

        Returns
        -------
        numpy.ndarray
            4D array of shape (N, N, N, N) containing all ERIs.
        """
        N = len(cgfs)

        # Build the list of unique ERI jobs and the packed storage array
        tedouble, tejobs = self._build_jobs(N)

        # Number of worker processes (one per CPU core)
        nproc = cpu_count()

        # Split job list into approximately equal chunks for workers
        chunks = np.array_split(tejobs, nproc)

        if verbose:
            print('Calculating electron repulsion integrals')
            print('Spawning %i threads' % nproc)
            print('Calculating %i ERI' % len(tedouble))

        # Parallel ERI evaluation over shell quartets
        with Pool(nproc) as pool:
            results = pool.map(
                ERIEvaluator._eri_worker,
                [(self, cgfs, chunk.tolist()) for chunk in chunks]
            )

        # Collect results back into the packed ERI array
        for chunk in results:
            for idx, val in chunk:
                tedouble[idx] = val

        # Expand packed ERI storage into a full 4D tensor
        res = np.empty((N, N, N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        res[i, j, k, l] = tedouble[teindex(i, j, k, l)]

        return res
    
    @staticmethod
    def _eri_worker(args):
        """
        Worker function executed in a separate process.

        Each worker receives:
        - a reference to the parent IntegralEvaluator
        - the list of CGFs
        - a list of ERI jobs to compute

        Each job corresponds to a unique (i, j, k, l) quartet.
        The worker computes all assigned ERIs and returns them as (index, value)
        pairs, where 'index' refers to the packed ERI array position.

        Parameters
        ----------
        args : tuple
            (self, cgfs, jobs)

        Returns
        -------
        list of tuples
            Each entry is (packed_index, eri_value).
        """
        self, cgfs, jobs = args
        out = []

        for idx, i, j, k, l in jobs:
            # Compute contracted ERI for this shell quartet
            val = self.repulsion(cgfs[i], cgfs[j], cgfs[k], cgfs[l])
            out.append((idx, val))

        return out
    
    def _build_jobs(self, sz):
        """
        Construct the list of unique ERI evaluation jobs exploiting
        permutational symmetry, and allocate packed storage for results.

        ERI symmetry:
            (ij|kl) = (ji|kl) = (ij|lk) = (kl|ij)

        Using this symmetry, only ERIs with:
            (i,j) <= (k,l)
        are explicitly computed.

        Parameters
        ----------
        sz : int
            Number of contracted basis functions.

        Returns
        -------
        tedouble : list
            Packed 1D array to store unique ERI values.
        jobs : list of tuples
            Each tuple is (packed_index, i, j, k, l),
            representing one ERI to compute.
        """
        # Maximum index needed for packed ERI storage
        max_idx = teindex(sz - 1, sz - 1, sz - 1, sz - 1)

        # Initialize packed ERI array with sentinel values
        tedouble = [-1.0] * (max_idx + 1)

        jobs = []  # list of (packed_index, i, j, k, l)

        for i in range(sz):
            for j in range(sz):
                # Triangular index for (i,j)
                ij = i * (i + 1) // 2 + j

                for k in range(sz):
                    for l in range(sz):
                        # Triangular index for (k,l)
                        kl = k * (k + 1) // 2 + l

                        # Enforce (ij) <= (kl) to avoid duplicates
                        if ij <= kl:
                            idx = teindex(i, j, k, l)

                            if idx >= len(tedouble):
                                raise RuntimeError(
                                    "Process tried to access illegal array position"
                                )

                            # Only schedule this ERI if it has not been assigned yet
                            if tedouble[idx] < 0.0:
                                tedouble[idx] = 1.0  # mark as scheduled
                                jobs.append((idx, i, j, k, l))

        return tedouble, jobs

    def _build_shells(self, cgfs):
        """
        Group pure-l CGFs into shells suitable for shell-based ERI evaluation.

        Parameters
        ----------
        cgfs : list
            List of contracted Gaussian functions (AOs).
            Each CGF must have a single angular momentum l.

        Returns
        -------
        shells : list of Shell
            List of shells.
        """
        shell_dict = defaultdict(list)

        # Group AO indices by shell key
        for ao_index, cgf in enumerate(cgfs):
            l = sum(cgf.gtos[0].o)

            key = (
                tuple(float(x) for x in cgf.gtos[0].p),
                l,
                tuple(gto.alpha for gto in cgf.gtos),
                tuple(gto.c for gto in cgf.gtos),
            )

            shell_dict[key].append(ao_index)

        # Build Shell objects
        shells = []

        for key, ao_indices in shell_dict.items():
            center, l, _, _ = key

            shells.append(
                Shell(
                    center=center,
                    l=l,
                    ao_indices=ao_indices,
                    nfunc=len(ao_indices)
                )
            )

        return shells
    
    def build_shell_jobs(self, shells):
        """
        Build shell-quartet ERI jobs partitioned by shell angular-momentum type.

        Returns
        -------
        jobs_by_type : dict
            keys: (lI, lJ, lK, lL)
            values: list of (I, J, K, L) shell quartets
        """
        nshell = len(shells)
        jobs_by_type = defaultdict(list)

        def pair_index(a, b):
            if a < b:
                a, b = b, a
            return a * (a + 1) // 2 + b

        for I in range(nshell):
            for J in range(I + 1):
                IJ = pair_index(I, J)

                lI, lJ = shells[I].l, shells[J].l
                if lI < lJ:
                    lI, lJ = lJ, lI

                for K in range(I + 1):
                    max_L = J if K == I else K

                    for L in range(max_L + 1):
                        KL = pair_index(K, L)

                        if IJ < KL:
                            continue

                        lK, lL = shells[K].l, shells[L].l
                        if lK < lL:
                            lK, lL = lL, lK

                        # enforce (lI,lJ) >= (lK,lL)
                        if (lI, lJ) < (lK, lL):
                            continue

                        jobs_by_type[(lI, lJ, lK, lL)].append((I, J, K, L))

        return jobs_by_type
