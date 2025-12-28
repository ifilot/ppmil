import unittest
import numpy as np
import os

from ppmil import Molecule, GTO
from ppmil import IntegralEvaluator, HuzinagaElectronRepulsionEngine, HellsingElectronRepulsionEngine
from ppmil.eri.teindex import teindex

class TestRepulsion(unittest.TestCase):

    def test_gto_repulsion(self):
        """
        Test two-electron integrals for primitive GTOs

        (ij|kl) = <gto_i gto_j | r_ij | gto_k gto_l>
        """

        # construct integrator object
        integrator = IntegralEvaluator(None, None, HuzinagaElectronRepulsionEngine())

        # test GTO
        gto1 = GTO(0.154329, 3.425251, [0.0, 0.0, 0.0], [0, 0, 0])
        repulsion = integrator.repulsion_primitive(gto1, gto1, gto1, gto1)
        result = 0.20141123130697272
        np.testing.assert_almost_equal(repulsion, result, 4)

    def test_cgf_repulsion(self):
        """
        Test two-electron integrals for contracted Gaussians

        (ij|kl) = <cgf_i cgf_j | r_ij | cgf_k cgf_l>
        """

        # construct integrator object
        integrator = IntegralEvaluator(None, None, HuzinagaElectronRepulsionEngine())

        # build hydrogen molecule
        mol = Molecule("H2")
        mol.add_atom('H', 0.0, 0.0, 0.0)
        mol.add_atom('H', 0.0, 0.0, 1.4)
        fname = os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json')
        cgfs, nuclei = mol.build_basis(fname)

        T1111 = integrator.repulsion(cgfs[0], cgfs[0], cgfs[0], cgfs[0])
        T1122 = integrator.repulsion(cgfs[0], cgfs[0], cgfs[1], cgfs[1])
        T1112 = integrator.repulsion(cgfs[0], cgfs[0], cgfs[0], cgfs[1])
        T2121 = integrator.repulsion(cgfs[1], cgfs[0], cgfs[1], cgfs[0])
        T1222 = integrator.repulsion(cgfs[0], cgfs[1], cgfs[1], cgfs[1])
        T2211 = integrator.repulsion(cgfs[1], cgfs[1], cgfs[0], cgfs[0])

        np.testing.assert_almost_equal(T1111, 0.7746056914329529, 4)
        np.testing.assert_almost_equal(T1122, 0.5696758031845093, 4)
        np.testing.assert_almost_equal(T1112, 0.4441076656879812, 4)
        np.testing.assert_almost_equal(T2121, 0.2970285713672638, 4)

        # test similarity between two-electron integrals
        np.testing.assert_almost_equal(T1222, T1112, 4)
        np.testing.assert_almost_equal(T1122, T2211, 4)
    
    def test_repulsion_h2o(self):
        """
        Test two-electron integrals for contracted Gaussians

        (ij|kl) = <cgf_i cgf_j | r_ij | cgf_k cgf_l>
        """

        # construct integrator object
        integrator = IntegralEvaluator(None, None, HuzinagaElectronRepulsionEngine())

        # build hydrogen molecule
        mol = Molecule("H2O")
        mol.add_atom('O', 0.00000, -0.07579, 0.00000)
        mol.add_atom('H', 0.86681, 0.60144, 0.00000)
        mol.add_atom('H',  -0.86681, 0.60144, 0.00000)
        fname = os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json')
        cgfs, nuclei = mol.build_basis(fname)
        
        N = len(cgfs)
        fname = os.path.join(os.path.dirname(__file__), 'data', 'repulsion_h2o.txt')
        vals = np.loadtxt(fname).reshape((N,N,N,N))
        res = integrator.eri_tensor(cgfs)
        np.testing.assert_almost_equal(res, vals, 4)

    def test_two_electron_indices(self):
        """
        Test unique two-electron indices
        """
        np.testing.assert_almost_equal(teindex(1,1,2,1), teindex(1,1,1,2), 4)
        np.testing.assert_almost_equal(teindex(1,1,2,1), teindex(2,1,1,1), 4)
        np.testing.assert_almost_equal(teindex(1,2,1,1), teindex(2,1,1,1), 4)
        np.testing.assert_almost_equal(teindex(1,1,1,2), teindex(1,1,2,1), 4)
        self.assertNotEqual(teindex(1,1,1,1), teindex(1,1,2,1))
        self.assertNotEqual(teindex(1,1,2,1), teindex(1,1,2,2))

    def test_gto_repulsion_hellsing(self):
        """
        Test two-electron integrals for primitive GTOs

        (ij|kl) = <gto_i gto_j | r_ij | gto_k gto_l>
        """

        # construct integrator object
        integrator = IntegralEvaluator(None, None, HellsingElectronRepulsionEngine())

        # test GTO
        gto1 = GTO(0.154329, 3.425251, [0.0, 0.0, 0.0], [0, 0, 0])
        repulsion = integrator.repulsion_primitive(gto1, gto1, gto1, gto1)
        result = 0.20141123130697272
        np.testing.assert_almost_equal(repulsion, result, 4)

    def test_cgf_repulsion_hellsing(self):
        """
        Test two-electron integrals for contracted Gaussians

        (ij|kl) = <cgf_i cgf_j | r_ij | cgf_k cgf_l>
        """

        # construct integrator object
        integrator = IntegralEvaluator(None, None, HellsingElectronRepulsionEngine())

        # build hydrogen molecule
        mol = Molecule("H2")
        mol.add_atom('H', 0.0, 0.0, 0.0)
        mol.add_atom('H', 0.0, 0.0, 1.4)
        fname = os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json')
        cgfs, nuclei = mol.build_basis(fname)

        T1111 = integrator.repulsion(cgfs[0], cgfs[0], cgfs[0], cgfs[0])
        T1122 = integrator.repulsion(cgfs[0], cgfs[0], cgfs[1], cgfs[1])
        T1112 = integrator.repulsion(cgfs[0], cgfs[0], cgfs[0], cgfs[1])
        T2121 = integrator.repulsion(cgfs[1], cgfs[0], cgfs[1], cgfs[0])
        T1222 = integrator.repulsion(cgfs[0], cgfs[1], cgfs[1], cgfs[1])
        T2211 = integrator.repulsion(cgfs[1], cgfs[1], cgfs[0], cgfs[0])

        np.testing.assert_almost_equal(T1111, 0.7746056914329529, 4)
        np.testing.assert_almost_equal(T1122, 0.5696758031845093, 4)
        np.testing.assert_almost_equal(T1112, 0.4441076656879812, 4)
        np.testing.assert_almost_equal(T2121, 0.2970285713672638, 4)

        # test similarity between two-electron integrals
        np.testing.assert_almost_equal(T1222, T1112, 4)
        np.testing.assert_almost_equal(T1122, T2211, 4)

    def test_repulsion_h2o_hellsing(self):
        """
        Test two-electron integrals for contracted Gaussians

        (ij|kl) = <cgf_i cgf_j | r_ij | cgf_k cgf_l>
        """

        # construct integrator object
        integrator = IntegralEvaluator(None, None, HellsingElectronRepulsionEngine())

        # build hydrogen molecule
        mol = Molecule("H2O")
        mol.add_atom('O', 0.00000, -0.07579, 0.00000)
        mol.add_atom('H', 0.86681, 0.60144, 0.00000)
        mol.add_atom('H',  -0.86681, 0.60144, 0.00000)
        fname = os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json')
        cgfs, nuclei = mol.build_basis(fname)
        
        N = len(cgfs)
        fname = os.path.join(os.path.dirname(__file__), 'data', 'repulsion_h2o.txt')
        vals = np.loadtxt(fname).reshape((N,N,N,N))
        res = integrator.eri_tensor(cgfs)
        np.testing.assert_almost_equal(res, vals, 4)

    def test_repulsion_h2o_hellsing_kernel(self):
        """
        Test two-electron integrals for contracted Gaussians

        (ij|kl) = <cgf_i cgf_j | r_ij | cgf_k cgf_l>
        """

        # construct integrator object
        integrator = IntegralEvaluator(None, None, HellsingElectronRepulsionEngine(True))

        # build hydrogen molecule
        mol = Molecule("H2O")
        mol.add_atom('O', 0.00000, -0.07579, 0.00000)
        mol.add_atom('H', 0.86681, 0.60144, 0.00000)
        mol.add_atom('H',  -0.86681, 0.60144, 0.00000)
        fname = os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json')
        cgfs, nuclei = mol.build_basis(fname)
        
        N = len(cgfs)
        fname = os.path.join(os.path.dirname(__file__), 'data', 'repulsion_h2o.txt')
        vals = np.loadtxt(fname).reshape((N,N,N,N))
        res = integrator.eri_tensor(cgfs)
        np.testing.assert_almost_equal(res, vals, 4)

if __name__ == '__main__':
    unittest.main()
