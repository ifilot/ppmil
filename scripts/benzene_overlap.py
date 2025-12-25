from ppmil import Molecule, IntegralEvaluator
from ppmil import HuzinagaOverlapEngine, HellsingOverlapEngine
from ppmil import HuzinagaNuclearEngine, HellsingNuclearEngine
import numpy as np
import os
import time

mol = Molecule('benzene', os.path.join(os.path.dirname(__file__), 'data', 'benzene.xyz'))
cgfs, nuclei = mol.build_basis(os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json'))

integrator1 = IntegralEvaluator(HuzinagaOverlapEngine(), HuzinagaNuclearEngine(), None)
integrator2 = IntegralEvaluator(HellsingOverlapEngine(True), HellsingNuclearEngine(), None)

N = len(cgfs)
S1 = np.empty((N,N))
S2 = np.empty((N,N))

st = time.perf_counter()
for i,c1 in enumerate(cgfs):
    for j,c2 in enumerate(cgfs):
        S1[i,j] = integrator1.nuclear(c1, c2, [0,0,0], 1)
end = time.perf_counter()
print('Time elapsed (Huzinaga): %.2f s' % (end - st))

st = time.perf_counter()
for i,c1 in enumerate(cgfs):
    for j,c2 in enumerate(cgfs):
        S2[i,j] = integrator2.nuclear(c1, c2, [0,0,0], 1)
end = time.perf_counter()
print('Time elapsed (Helsing): %.2f s' % (end - st))

print('Max difference: ', np.max(np.abs(S1 - S2)))