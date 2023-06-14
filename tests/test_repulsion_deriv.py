import unittest
from copy import copy, deepcopy
import numpy as np
import os, sys

# add a reference to load the PPMIL library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ppmil import Molecule, PPMIL

class TestRepulsionDerivatives(unittest.TestCase):
   
   def test_derivatives_h2o_subset(self):
       """
       Test Derivatives of water
       """

       # build integrator object
       integrator = PPMIL()

       # build hydrogen molecule
       molfile = os.path.join(os.path.dirname(__file__), 'data', 'h2o.xyz')
       mol = Molecule(xyzfile=molfile)
       basisfile = os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json')
       cgfs, nuclei = mol.build_basis('sto3g', basisfile)

       # calculate derivative towards H1 in the x-direction
       Opos = nuclei[0][0]
       H1pos = nuclei[1][0]
       fx1 = integrator.repulsion_deriv(cgfs[2], cgfs[2], cgfs[2], cgfs[2], H1pos, 0) # 2px
       fx2 = integrator.repulsion_deriv(cgfs[2], cgfs[3], cgfs[3], cgfs[3], H1pos, 0) # 2py

       ans1 = calculate_force_finite_difference(molfile, basisfile, 2, 2, 2, 2, 1, 0)
       ans2 = calculate_force_finite_difference(molfile, basisfile, 2, 3, 3, 3, 1, 0)

       # assert that the overlap of two CGFs that spawn from
       # the same nucleus will not change in energy due to a
       # change of the nucleus coordinates
       np.testing.assert_almost_equal(fx1, ans1, 4)
       np.testing.assert_almost_equal(fx2, ans2, 4)

       # assert that the cross-terms will change
       fx3 = integrator.repulsion_deriv(cgfs[3], cgfs[3], cgfs[5], cgfs[5], Opos, 0)
       fx4 = integrator.repulsion_deriv(cgfs[3], cgfs[3], cgfs[5], cgfs[5], H1pos, 0)
       fx5 = integrator.repulsion_deriv(cgfs[5], cgfs[3], cgfs[5], cgfs[3], Opos, 0)
       fx6 = integrator.repulsion_deriv(cgfs[3], cgfs[5], cgfs[3], cgfs[5], H1pos, 0)

       ans3 = calculate_force_finite_difference(molfile, basisfile, 3, 3, 5, 5, 0, 0)
       ans4 = calculate_force_finite_difference(molfile, basisfile, 3, 3, 5, 5, 1, 0)
       ans5 = calculate_force_finite_difference(molfile, basisfile, 5, 3, 5, 3, 0, 0)
       ans6 = calculate_force_finite_difference(molfile, basisfile, 3, 5, 3, 5, 1, 0)

       np.testing.assert_almost_equal(fx3, ans3, 4)
       self.assertFalse(fx3 == 0.0)
       # np.testing.assert_almost_equal(fx4, ans4, 4)
       # self.assertFalse(fx4 == 0.0) 
       # np.testing.assert_almost_equal(fx5, ans5, 4)
       # self.assertFalse(fx5 == 0.0)
       # np.testing.assert_almost_equal(fx6, ans6, 4)
       # self.assertFalse(fx6 == 0.0) 
   
    # def test_derivatives_h2o_fulltest(self):
    #     """
    #     Test Derivatives of water
    #     """
    #     # build integrator object
    #     integrator = PPMIL()

    #     # build hydrogen molecule
    #     molfile = os.path.join(os.path.dirname(__file__), 'data', 'h2o.xyz')
    #     mol = Molecule(xyzfile=molfile)
    #     basisfile = os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json')
    #     cgfs, nuclei = mol.build_basis('sto3g', basisfile)
    #     O = nuclei[0][0]
    #     Ochg = nuclei[0][1]

    #     # load results from file
    #     fname = os.path.join(os.path.dirname(__file__), 'data', 'repulsion_deriv_h2o.txt')
    #     #vals = np.loadtxt(fname).reshape((len(cgfs), len(cgfs), (len(cgfs), (len(cgfs), 3, 3))
    #     for i in range(0, len(cgfs)): # loop over cgfs
    #         for j in range(0, len(cgfs)): # loop over cgfs
    #             for k in range(0, len(cgfs)): # loop over cgfs
    #                 for l in range(0, len(cgfs)): # loop over cgfs
    #                     for m in range(0,3):  # loop over nuclei
    #                         for n in range(0,3):  # loop over directions
    #                             force = integrator.repulsion_deriv(cgfs[i], cgfs[j], cgfs[k], cgfs[l], nuclei[m][0], n)
    #                             val = calculate_force_finite_difference(molfile, basisfile, i, j, k, l, m, n)
    #                             np.testing.assert_almost_equal(force, val, 4)
                        

def calculate_force_finite_difference(molfile, basisfile, 
                                      cgf_id1, cgf_id2, cgf_id3, cgf_id4, 
                                      nuc_id, coord):
    # build integrator object
    integrator = PPMIL()

    # distance
    diff = 0.00001

    vals = np.zeros(2)
    for i,v in enumerate([-1,1]):
        mol = Molecule(xyzfile=molfile)
        mol.atoms[nuc_id][1][coord] += v * diff / 2
        cgfs, nuclei = mol.build_basis('sto3g', basisfile)
        vals[i] = integrator.repulsion(cgfs[cgf_id1], cgfs[cgf_id2], 
                                       cgfs[cgf_id3], cgfs[cgf_id4])

    return (vals[1] - vals[0]) / diff

if __name__ == '__main__':
    unittest.main()