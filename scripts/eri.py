import numpy as np
import os
import time

from ppmil import Molecule, GTO
from ppmil import IntegralEvaluator, HuzinagaElectronRepulsionEngine
from ppmil.eri.teindex import teindex

def main():
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

    st = time.perf_counter()
    res = integrator.eri_tensor(cgfs)
    np.testing.assert_almost_equal(res, vals, decimal=4)
    end = time.perf_counter()
    print('Time elapsed (Huzinaga): %.2f s' % (end - st))

if __name__ == '__main__':
    main()