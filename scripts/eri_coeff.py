import numpy as np
import os
import time

from ppmil import Molecule, GTO
from ppmil import IntegralEvaluator, HellsingElectronRepulsionEngine
from ppmil.eri.teindex import teindex

def main():
    # construct integrator object
    integrator = IntegralEvaluator(None, None, HellsingElectronRepulsionEngine())
    floats, ints = integrator._eri_engine.calculate_coefficients(2,2,2,2)
    print(ints.shape)
    print(ints.T)
    print(floats)

if __name__ == '__main__':
    main()