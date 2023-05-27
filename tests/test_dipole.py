import unittest
import numpy as np
import sys
import os

# add a reference to load the PPMIL library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ppmil import Molecule, PPMIL

class TestDipole(unittest.TestCase):

    def testDipole(self):
        fname = os.path.join(os.path.dirname(__file__), 'data', 'h2o.xyz')
        h2o = Molecule(xyzfile=fname)
        fname = os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json')
        cgfs, nuclei = h2o.build_basis('sto3g', fname)
        N = len(cgfs) # basis set size
        
        integrator = PPMIL()
        dx = np.zeros((N,N))
        dy = np.zeros((N,N))
        dz = np.zeros((N,N))
        for i in range(N):
            for j in range(i,N):
                dx[i,j] = dx[j,i] = integrator.dipole(cgfs[i], cgfs[j], 0, 0.0)
                dy[i,j] = dy[j,i] = integrator.dipole(cgfs[i], cgfs[j], 1, 0.0)
                dz[i,j] = dz[j,i] = integrator.dipole(cgfs[i], cgfs[j], 2, 0.0)
        
        # test dipole integrals
        exact_x = np.loadtxt(os.path.join(os.path.dirname(__file__), 
                                          'data', 
                                          'dipole_x.txt'))
        np.testing.assert_almost_equal(dx, exact_x, decimal=4)
        exact_y = np.loadtxt(os.path.join(os.path.dirname(__file__), 
                                          'data', 
                                          'dipole_y.txt'))
        np.testing.assert_almost_equal(dy, exact_y, decimal=4)
        exact_z = np.loadtxt(os.path.join(os.path.dirname(__file__), 
                                          'data', 
                                          'dipole_z.txt'))
        np.testing.assert_almost_equal(dz, exact_z, decimal=4)

if __name__ == '__main__':
    unittest.main()
