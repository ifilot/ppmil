import unittest

from ppmil import IntegralEvaluator, CGF

class TestExceptions(unittest.TestCase):

    def testInvalidType(self):        
        integrator = IntegralEvaluator(None, None, None, None)
        
        with self.assertRaises(TypeError) as context:
            integrator.overlap(1,1)
        
        self.assertTrue('Argument cgf1 must be of CGF type' in str(context.exception))
        
        cgf = CGF()
        
        with self.assertRaises(TypeError) as context:
            integrator.overlap(cgf,1)
        
        self.assertTrue('Argument cgf2 must be of CGF type' in str(context.exception))

if __name__ == '__main__':
    unittest.main()
