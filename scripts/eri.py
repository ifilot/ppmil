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
    mol = Molecule('benzene', os.path.join(os.path.dirname(__file__), 'data', 'co.xyz'))
    cgfs, nuclei = mol.build_basis(os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json'))

    st = time.perf_counter()
    res = integrator.eri_tensor(cgfs, verbose=True)
    end = time.perf_counter()
    print('Time elapsed (Huzinaga): %.2f s' % (end - st))

if __name__ == '__main__':
    main()