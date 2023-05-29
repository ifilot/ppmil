import unittest
import numpy as np
import sys
import os
import importlib.util

# add a reference to load the PPMIL library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

pylebedev_spec = importlib.util.find_spec('pylebedev')
if pylebedev_spec is None:
    sys.exit(0)

from ppmil import Molecule, PPMIL, Quadrature

class TestQuadrature(unittest.TestCase):

    def testOverlap(self):
        fname = os.path.join(os.path.dirname(__file__), 'data', 'h2o.xyz')
        h2o = Molecule(xyzfile=fname)
        fname = os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json')
        cgfs, nuclei = h2o.build_basis('sto3g', fname)
        
        # grab some gtos
        gtos = [
            cgfs[1].gtos[1],  # O 2s
            cgfs[-2].gtos[1], # H 1s
            cgfs[-1].gtos[1], # H 1s
        ]
        
        integrator = PPMIL()
        quad = Quadrature()
        
        for gto1 in gtos:
            for gto2 in gtos:
                overlap = integrator.overlap_gto(gto1, gto2)
                overlap_quad = quad.quad_overlap(gto1, gto2, 32, 7)
        
                np.testing.assert_almost_equal(overlap_quad, overlap, 8)
        
    def testDipole(self):
        fname = os.path.join(os.path.dirname(__file__), 'data', 'h2o.xyz')
        h2o = Molecule(xyzfile=fname)
        fname = os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json')
        cgfs, nuclei = h2o.build_basis('sto3g', fname)
        
        # grab some gtos
        gtos = [
            cgfs[1].gtos[1],  # O 2s
            cgfs[-2].gtos[1], # H 1s
            cgfs[-1].gtos[1], # H 1s
        ]
        
        integrator = PPMIL()
        quad = Quadrature()
        
        for gto1 in gtos:
            for gto2 in gtos:
                dx = integrator.dipole_gto(gtos[0], gtos[1], 0)
                dy = integrator.dipole_gto(gtos[0], gtos[1], 1)
                dz = integrator.dipole_gto(gtos[0], gtos[1], 2)
                dipole = np.array([dx, dy, dz])
        
                dipole_quad = quad.quad_dipole(gtos[0], gtos[1], 64, 35)
        
                np.testing.assert_almost_equal(dipole_quad, dipole, 8)

# only execute this script if PyLebedev is installed
if __name__ == '__main__':
    unittest.main()
