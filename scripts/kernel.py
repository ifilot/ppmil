from ppmil import Molecule, IntegralEvaluator
from ppmil import HellsingNuclearEngine
import numpy as np

engine = HellsingNuclearEngine(True, 2)

print('(0|0)')
engine.print_kernel_coefficients(0,0)
print()

print('(1|0)')
engine.print_kernel_coefficients(1,0)
print()

print('(0|1)')
engine.print_kernel_coefficients(0,1)
print()

print('(1|1)')
engine.print_kernel_coefficients(1,1)
print()