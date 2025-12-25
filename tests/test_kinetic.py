import unittest
import numpy as np
import os

from ppmil import Molecule, IntegralEvaluator
from ppmil import HuzinagaOverlapEngine, HellsingOverlapEngine

from ppmil import Molecule, IntegralEvaluator

class TestKinetic(unittest.TestCase):

    def testKineticHuzinaga(self):
        fname = os.path.join(os.path.dirname(__file__), 'data', 'h2o.xyz')
        h2o = Molecule(xyzfile=fname)
        fname = os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json')
        cgfs, nuclei = h2o.build_basis(fname)
        N = len(cgfs) # basis set size
        
        integrator = IntegralEvaluator(HuzinagaOverlapEngine(), None, None)

        T = np.zeros((N,N))
        for i in range(N):
            for j in range(i,N):
                T[i,j] = T[j,i] = integrator.kinetic(cgfs[i], cgfs[j])
        
        # test overlap integrals
        fname = os.path.join(os.path.dirname(__file__), 'data', 'kinetic_h2o.txt')
        exact = np.loadtxt(fname)
        np.testing.assert_almost_equal(T, exact, 4)
    
    def testKineticHellsing(self):
        fname = os.path.join(os.path.dirname(__file__), 'data', 'h2o.xyz')
        h2o = Molecule(xyzfile=fname)
        fname = os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json')
        cgfs, nuclei = h2o.build_basis(fname)
        N = len(cgfs) # basis set size
        
        integrator = IntegralEvaluator(HellsingOverlapEngine(), None, None)

        T = np.zeros((N,N))
        for i in range(N):
            for j in range(i,N):
                T[i,j] = T[j,i] = integrator.kinetic(cgfs[i], cgfs[j])
        
        # test overlap integrals
        fname = os.path.join(os.path.dirname(__file__), 'data', 'kinetic_h2o.txt')
        exact = np.loadtxt(fname)
        np.testing.assert_almost_equal(T, exact, 4)

if __name__ == '__main__':
    unittest.main()
