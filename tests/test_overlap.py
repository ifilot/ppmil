import unittest
import numpy as np
import os
from ppmil import Molecule, IntegralEvaluator
from ppmil import HuzinagaOverlapEngine, HellsingOverlapEngine

class TestOverlapHuzinaga(unittest.TestCase):

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

class TestOverlapHellsing(unittest.TestCase):

    def testOverlap(self):
        fname = os.path.join(os.path.dirname(__file__), 'data', 'h2o.xyz')
        h2o = Molecule(xyzfile=fname)
        fname = os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json')
        cgfs, nuclei = h2o.build_basis('sto3g', fname)
        N = len(cgfs) # basis set size
        
        # build integrator engine
        integrator = IntegralEvaluator(HellsingOverlapEngine(), None, None, None)

        S = np.zeros((N,N))
        for i in range(N):
            for j in range(i,N):
                S[i,j] = S[j,i] = integrator.overlap(cgfs[i], cgfs[j])
        
        # test overlap integrals
        fname = os.path.join(os.path.dirname(__file__), 'data', 'overlap_h2o.txt')
        exact = np.loadtxt(fname)
        try:
            np.testing.assert_allclose(S, exact, 4)
        except AssertionError:
            print("\n=== COMPUTED S ===")
            print(S)
            print("\n=== EXACT ===")
            print(exact)
            print("\n=== DIFFERENCE (S - exact) ===")
            print(S - exact)
            raise

if __name__ == '__main__':
    unittest.main()
