import unittest
import numpy as np

from ppmil.math.gamma_numba import Fgamma

class TestGammaFunction(unittest.TestCase):

    def test_Fgamma(self):        
        np.testing.assert_almost_equal(Fgamma(0, 0), 1.0, 4)
        np.testing.assert_almost_equal(Fgamma(0, 0.5), 0.855624, 4)
        np.testing.assert_almost_equal(Fgamma(0, 1.0), 0.746824, 4)
        
        np.testing.assert_almost_equal(Fgamma(1, 1.0), 0.189472, 4)
        np.testing.assert_almost_equal(Fgamma(2, 1.0), 0.100269, 4)
        np.testing.assert_almost_equal(Fgamma(3, 1.0), 0.0667323, 4)
        np.testing.assert_almost_equal(Fgamma(4, 1.0), 0.0496232, 4)
        np.testing.assert_almost_equal(Fgamma(5, 1.0), 0.0393649, 4)

if __name__ == '__main__':
    unittest.main()
