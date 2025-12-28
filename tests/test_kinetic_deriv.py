import unittest
import numpy as np
import os

from ppmil import Molecule, IntegralEvaluator, HuzinagaOverlapEngine, HellsingOverlapEngine

class TestKineticDeriv(unittest.TestCase):

    def test_derivatives_h2o_subset(self):
        """
        Test Derivatives of water
        """

        # build integrator object
        integrator = IntegralEvaluator(HuzinagaOverlapEngine(), None, None)

        # build hydrogen molecule
        molfile = os.path.join(os.path.dirname(__file__), 'data', 'h2o.xyz')
        mol = Molecule(xyzfile=molfile)
        basisfile = os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json')
        cgfs, nuclei = mol.build_basis(basisfile)

        # calculate derivative towards H1 in the x-direction
        H1pos = nuclei[1][0]
        fx1 = integrator.kinetic_deriv(cgfs[2], cgfs[2], H1pos, 0) # 2px
        fx2 = integrator.kinetic_deriv(cgfs[2], cgfs[3], H1pos, 0) # 2py

        ans1 = calculate_force_finite_difference(molfile, basisfile, 1, 2, 2, 0)
        ans2 = calculate_force_finite_difference(molfile, basisfile, 1, 3, 3, 0)

        # assert that the overlap of two CGFs that spawn from
        # the same nucleus will not change in energy due to a
        # change of the nucleus coordinates
        np.testing.assert_almost_equal(fx1, ans1, 4)
        np.testing.assert_almost_equal(fx2, ans2, 4)

        # # assert that the cross-terms will change
        fx3 = integrator.kinetic_deriv(cgfs[2], cgfs[5], nuclei[1][0], 0) # 2px
        fx4 = integrator.kinetic_deriv(cgfs[2], cgfs[5], nuclei[1][0], 0) # 2px

        ans3 = calculate_force_finite_difference(molfile, basisfile, 1, 2, 5, 0)
        ans4 = calculate_force_finite_difference(molfile, basisfile, 1, 2, 5, 0)

        np.testing.assert_almost_equal(fx3, ans3, 4)
        self.assertFalse(fx3 == 0.0)
        np.testing.assert_almost_equal(fx4, ans4, 4)
        self.assertFalse(fx4 == 0.0)
        
    def test_derivatives_h2o_fulltest(self):
        """
        Test Derivatives of water
        """
        # build integrator object
        integrator = IntegralEvaluator(HuzinagaOverlapEngine(), None, None)

        # build hydrogen molecule
        molfile = os.path.join(os.path.dirname(__file__), 'data', 'h2o.xyz')
        mol = Molecule(xyzfile=molfile)
        basisfile = os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json')
        cgfs, nuclei = mol.build_basis(basisfile)

        # load results from file
        fname = os.path.join(os.path.dirname(__file__), 'data', 'kinetic_deriv_h2o.txt')
        vals = np.loadtxt(fname).reshape((len(cgfs), len(cgfs), 3, 3))
        for i in range(0, len(cgfs)): # loop over cgfs
            for j in range(0, len(cgfs)): # loop over cgfs
                for k in range(0,3): # loop over nuclei
                    for l in range(0,3): # loop over directions
                        force = integrator.kinetic_deriv(cgfs[i], cgfs[j], nuclei[k][0], l)
                        np.testing.assert_almost_equal(force, vals[i,j,k,l], 4)

    def test_derivatives_h2o_fulltest_hellsing(self):
        """
        Test Derivatives of water
        """
        # build integrator object
        integrator = IntegralEvaluator(HellsingOverlapEngine(), None, None)

        # build hydrogen molecule
        molfile = os.path.join(os.path.dirname(__file__), 'data', 'h2o.xyz')
        mol = Molecule(xyzfile=molfile)
        basisfile = os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json')
        cgfs, nuclei = mol.build_basis(basisfile)

        # load results from file
        fname = os.path.join(os.path.dirname(__file__), 'data', 'kinetic_deriv_h2o.txt')
        vals = np.loadtxt(fname).reshape((len(cgfs), len(cgfs), 3, 3))
        for i in range(0, len(cgfs)): # loop over cgfs
            for j in range(0, len(cgfs)): # loop over cgfs
                for k in range(0,3): # loop over nuclei
                    for l in range(0,3): # loop over directions
                        force = integrator.kinetic_deriv(cgfs[i], cgfs[j], nuclei[k][0], l)
                        np.testing.assert_almost_equal(force, vals[i,j,k,l], 4)
                        

def calculate_force_finite_difference(molfile, basisfile, 
                                      nuc_id, cgf_id1, cgf_id2, coord):
    # build integrator object
    integrator = IntegralEvaluator(HuzinagaOverlapEngine(), None, None)

    # distance
    diff = 0.00001

    vals = np.zeros(2)
    for i,v in enumerate([-1,1]):
        mol = Molecule(xyzfile=molfile)
        mol.atoms[nuc_id][1][coord] += v * diff / 2
        cgfs, nuclei = mol.build_basis(basisfile)
        vals[i] = integrator.kinetic(cgfs[cgf_id1], cgfs[cgf_id2])

    return (vals[1] - vals[0]) / diff

if __name__ == '__main__':
    unittest.main()