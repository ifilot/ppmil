from ppmil.erieval.obara_saika import ObaraSaika
from ppmil import Molecule, GTO, IntegralEvaluator, HuzinagaElectronRepulsionEngine
import os
from ppmil.math.gamma_numba import Fgamma

# build hydrogen molecule
mol = Molecule('co', os.path.join(os.path.dirname(__file__), 'data', 'co.xyz'))
cgfs, nuclei = mol.build_basis(os.path.join(os.path.dirname(__file__), 'data', 'sto3g.json'))

print(Fgamma(0, 0.0))
print(Fgamma(1, 0.0))

eval = ObaraSaika()
buf = ObaraSaika.shell_psss(
    cgfs[2], cgfs[0], cgfs[5], cgfs[5]
)
print(buf)

integrator = IntegralEvaluator(None, None, HuzinagaElectronRepulsionEngine())

for i in range(2,5):
    print(integrator.repulsion(cgfs[i], cgfs[0], cgfs[5], cgfs[5]))