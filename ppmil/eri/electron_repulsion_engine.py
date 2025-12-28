from abc import abstractmethod
from scipy.special import factorial
import numpy as np

from ..util.gto import GTO

class ElectronRepulsionEngine:
    """
    Abstract class for Two Electron Integral Engine
    """
    def __init__(self):
        self._fact = np.array(
            [factorial(i) for i in range(10)],
            dtype=np.float64
        )

    @abstractmethod
    def repulsion_primitive(self, g1:GTO, g2:GTO, g3:GTO, g4:GTO):
        pass