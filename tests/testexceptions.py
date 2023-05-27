import unittest
import numpy as np
import sys
import os

# add a reference to load the PPMIL library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ppmil import Molecule, PPMIL, CGF

class TestExceptions(unittest.TestCase):

    def testInvalidType(self):        
        integrator = PPMIL()
        
        with self.assertRaises(TypeError) as context:
            integrator.overlap(1,1)
        
        self.assertTrue('Argument cgf1 must be of CGF type' in str(context.exception))
        
        cgf = CGF()
        
        with self.assertRaises(TypeError) as context:
            integrator.overlap(cgf,1)
        
        self.assertTrue('Argument cgf2 must be of CGF type' in str(context.exception))

if __name__ == '__main__':
    unittest.main()
