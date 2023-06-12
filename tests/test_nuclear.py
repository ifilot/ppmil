import unittest
import numpy as np
import sys
import os

# add a reference to load the PPMIL library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ppmil import Molecule, PPMIL, GTO

class TestNuclear(unittest.TestCase):

    def test_gto_nuclear(self):
        """
        Test nuclear attraction integral for GTOs

        V^{(c)}_ij = <gto_i | -1 / |r-Rc| | gto_j>
        """
        # construct integrator object
        integrator = PPMIL()

        # test GTO
        gto1 = GTO(0.154329, 3.425251, [0.0, 0.0, 0.0], [0, 0, 0])

        nuclear = integrator.nuclear_gto(gto1, gto1, [0.0, 0.0, 0.0])
        result = -0.9171861107748928
        np.testing.assert_almost_equal(nuclear, result, 4)
        
        nuclear = integrator.nuclear_gto(gto1, gto1, [0.0, 0.0, 1.0])
        result = -0.31049036979675293
        np.testing.assert_almost_equal(nuclear, result, 4)

    def test_cgf_nuclear(self):
        """
        Test nuclear attraction integrals for contracted Gaussians
        for the H2 molecule

        V^{(c)}_ij = <cgf_i | -Zc / |r-Rc| | cgf_j>
        """

        integrator = PPMIL()

        # build hydrogen molecule
        mol = Molecule("H2")
        mol.add_atom('H', 0.0, 0.0, 0.0)
        mol.add_atom('H', 0.0, 0.0, 1.4)
        fname = os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json')
        cgfs, nuclei = mol.build_basis('sto3g', fname)

        V1 = np.zeros((2,2))
        V1[0,0] = integrator.nuclear(cgfs[0], cgfs[0], cgfs[0].p, 1)
        V1[0,1] = V1[1,0] = integrator.nuclear(cgfs[0], cgfs[1], cgfs[0].p, 1)
        V1[1,1] = integrator.nuclear(cgfs[1], cgfs[1], cgfs[0].p, 1)

        V2 = np.zeros((2,2))
        V2[0,0] = integrator.nuclear(cgfs[0], cgfs[0], cgfs[1].p, 1)
        V2[0,1] = V2[1,0] = integrator.nuclear(cgfs[0], cgfs[1], cgfs[1].p, 1)
        V2[1,1] = integrator.nuclear(cgfs[1], cgfs[1], cgfs[1].p, 1)

        V11 = -1.2266135215759277
        V12 = -0.5974172949790955
        V22 = -0.6538270711898804
        np.testing.assert_almost_equal(V1[0,0], V11, 4)
        np.testing.assert_almost_equal(V1[1,1], V22, 4)
        np.testing.assert_almost_equal(V1[0,1], V12, 4)
        np.testing.assert_almost_equal(V2[0,0], V22, 4)
        np.testing.assert_almost_equal(V2[1,1], V11, 4)
        np.testing.assert_almost_equal(V2[0,1], V12, 4)

    def test_kinetic_h2o(self):
        """
        Test nuclear attraction integrals for contracted Gaussians
        for the H2O molecule

        V^{(c)}_ij = <cgf_i | -Zc / |r-Rc| | cgf_j>
        
        """
        fname = os.path.join(os.path.dirname(__file__), 'data', 'h2o.xyz')
        h2o = Molecule(xyzfile=fname)
        fname = os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json')
        cgfs, nuclei = h2o.build_basis('sto3g', fname)
        N = len(cgfs) # basis set size
        
        integrator = PPMIL()
        V = np.zeros((N,N))
        for i in range(N):
            for j in range(i,N):
                V[i,j] = V[j,i] = integrator.nuclear(cgfs[i], cgfs[j], 
                                                     nuclei[0][0], nuclei[0][1])
        
        # test overlap integrals
        fname = os.path.join(os.path.dirname(__file__), 'data', 'nuclear_h2o.txt')
        exact = np.loadtxt(fname)
        np.testing.assert_almost_equal(V, exact, 4)    

if __name__ == '__main__':
    unittest.main()
