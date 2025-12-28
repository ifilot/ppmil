import numpy as np
import os
import time

from ppmil import Molecule, GTO
from ppmil import ERIEvaluator, HellsingElectronRepulsionEngine
from ppmil.eri.teindex import teindex

def main():
    # construct integrator object
    integrator = ERIEvaluator(HellsingElectronRepulsionEngine(True))

    # build hydrogen molecule
    mol = Molecule('co', os.path.join(os.path.dirname(__file__), 'data', 'co.xyz'))
    cgfs, nuclei = mol.build_basis(os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json'))
    shells = integrator._build_shells(cgfs)
    jobsets = integrator.build_shell_jobs(shells)
    
    print('Shells')
    for shell in shells:
        print(shell)

    print('Jobs')
    for k,v in jobsets.items():
        print(k,v)

    st = time.perf_counter()
    res = integrator.eri_tensor(cgfs, verbose=True)
    end = time.perf_counter()
    print('Time elapsed (Hellsing): %.2f s' % (end - st))

if __name__ == '__main__':
    main()