import numpy as np
import os
import time

from ppmil import Molecule, GTO
from ppmil import ERIEvaluator, HellsingElectronRepulsionEngine
from ppmil.eri.teindex import teindex

def main():
    integrator = ERIEvaluator(HellsingElectronRepulsionEngine)

    # build hydrogen molecule
    mol = Molecule('h2o', os.path.join(os.path.dirname(__file__), 'data', 'h2o.xyz'))
    cgfs, nuclei = mol.build_basis(os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json'))
    res = integrator.eri_tensor(cgfs, verbose=True)
    print(res[0][0][0][0])

if __name__ == '__main__':
    main()