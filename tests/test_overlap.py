import unittest
import numpy as np
import os
from ppmil import Molecule, IntegralEvaluator, HuzinagaOverlapEngine

class TestOverlap(unittest.TestCase):

    def testOverlap(self):
        fname = os.path.join(os.path.dirname(__file__), 'data', 'h2o.xyz')
        h2o = Molecule(xyzfile=fname)
        fname = os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json')
        cgfs, nuclei = h2o.build_basis('sto3g', fname)
        N = len(cgfs) # basis set size
        
        # build integrator engine
        integrator = IntegralEvaluator(HuzinagaOverlapEngine(), None, None, None)

        S = np.zeros((N,N))
        for i in range(N):
            for j in range(i,N):
                S[i,j] = S[j,i] = integrator.overlap(cgfs[i], cgfs[j])
        
        # test overlap integrals
        fname = os.path.join(os.path.dirname(__file__), 'data', 'overlap_h2o.txt')
        exact = np.loadtxt(fname)
        np.testing.assert_almost_equal(S, exact, 4)

if __name__ == '__main__':
    unittest.main()
